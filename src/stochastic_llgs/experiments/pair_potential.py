"""Forced-pair production scan for the inter-skyrmion
potential V(r). Demag on.

Builds a flat grid
`[(r_init, ens_idx) for r_init in r_init_list for ens_idx in
range(n_ens)]`. Each trajectory writes its own NPZ with the
pair-separation time series `r(t)`; V(r) is reconstructed
downstream by integrating `<dr/dt>(r)` across the ensemble.

Execution modes
---------------
- Serial / multiprocess via `SWEEP_NPROC` (default 1).
- SLURM array: each task runs `grid[SLURM_ARRAY_TASK_ID]`.

Run with:
    python -m src.stochastic_llgs.experiments.pair_potential
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import multiprocessing as mp
import os
# Third-party
import numpy as np
# Local
from src.phase_diagram.params_helper import make_params
from src.simulator.demag import precompute_demag_kernels
from src.simulator.main import topological_charge
from src.simulator.pulses import ConstantPulse
from src.stochastic_llgs.diagnostics import (
    skyrmion_center_pbc,
)
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
from src.stochastic_llgs.io import save_trajectory
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.thermal_field import sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def skyrmion_at_position(nx, ny, a, R, dw, polarity, cx, cy):
    """Single Neel skyrmion placed at `(cx, cy)`.

    Mirrors `src.simulator.initial_conditions.skyrmion_profile`
    but the core position is configurable (the original centers
    at the lattice midpoint).
    """
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
    )
    x = jj * a - float(cx)
    y = ii * a - float(cy)
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    if int(polarity) == -1:
        theta = np.pi - theta
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    m = np.zeros((ny, nx, 3))
    m[..., 0] = sin_t * np.cos(phi)
    m[..., 1] = sin_t * np.sin(phi)
    m[..., 2] = cos_t
    return m


# -----------------------------------------------------------------------------
def two_skyrmion_pair_ic(nx, ny, a, R, dw, polarity_top,
                         c1, c2):
    """Build a SAF magnetization pair containing two skyrmions
    at positions `c1` and `c2`."""
    m_top_1 = skyrmion_at_position(
        nx, ny, a, R, dw, polarity_top, c1[0], c1[1])
    m_top_2 = skyrmion_at_position(
        nx, ny, a, R, dw, polarity_top, c2[0], c2[1])
    # Overlay the two single-skyrmion fields by keeping, per cell,
    # whichever has the deeper core (lower m_z for the top layer).
    pick1 = (
        m_top_1[..., 2] <= m_top_2[..., 2]
    )[..., np.newaxis]
    m_top = np.where(pick1, m_top_1, m_top_2)
    pol_bot = -int(polarity_top)
    m_bot_1 = skyrmion_at_position(
        nx, ny, a, R, dw, pol_bot, c1[0], c1[1])
    m_bot_2 = skyrmion_at_position(
        nx, ny, a, R, dw, pol_bot, c2[0], c2[1])
    # Bottom layer has opposite polarity: keep the higher-m_z core.
    pick1b = (
        m_bot_1[..., 2] >= m_bot_2[..., 2]
    )[..., np.newaxis]
    m_bot = np.where(pick1b, m_bot_1, m_bot_2)
    return m_top, m_bot


# -----------------------------------------------------------------------------
def _run_one_point(args):
    """Worker function: run one forced-pair trajectory and save."""
    cfg, point = args
    r_init = float(point['r_init'])
    ens_idx = int(point['ens_idx'])
    cell_idx = int(point['cell_idx'])
    # Decorrelate seeds: distinct stride per ensemble member and
    # per grid cell so no two trajectories share an RNG stream.
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * cell_idx
    nx = int(cfg['nx'])
    ny = int(cfg['ny'])
    dt = float(cfg['dt'])
    n_relax = int(cfg['n_relax'])
    n_drive = int(cfg['n_drive'])
    sample_every = int(cfg['sample_every'])
    tol_norm = float(cfg['tol_norm'])
    use_demag = bool(cfg['use_demag'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    p = make_params(nx=nx, ny=ny, J_current=0.0)
    p.dt = dt
    p.pulse = ConstantPulse(0.0)
    attach_thermal(
        p, T=float(cfg['T_sub']),
        R_th=float(cfg['R_th']), seed=seed)
    kernels = (precompute_demag_kernels(
        p, kind='slab', accuracy=None, tol_conv=None)
        if use_demag else None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    L_x = nx * p.a
    L_y = ny * p.a
    # Place the two cores symmetrically about box center, r_init
    # apart along x, on the same midline (separation drives V(r)).
    cy0 = L_y / 2.0
    c1 = (L_x / 2.0 - 0.5 * r_init, cy0)
    c2 = (L_x / 2.0 + 0.5 * r_init, cy0)
    m_top, m_bot = two_skyrmion_pair_ic(
        nx, ny, p.a, p.skyrmion_R, p.skyrmion_dw,
        polarity_top=1, c1=c1, c2=c2)
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for _ in range(n_relax):
        h_top = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels,
            h_top, h_bot, tol_norm,
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_samples = (n_drive + sample_every - 1) // sample_every
    t_sample = np.empty(n_samples, dtype=float)
    r_pair = np.empty(n_samples, dtype=float)
    Q_arr = np.empty(n_samples, dtype=float)
    alive_both = True
    s_idx = 0
    for step in range(1, n_drive + 1):
        h_top = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels,
            h_top, h_bot, tol_norm,
        )
        if step % sample_every == 0:
            # Split the lattice into left/right halves and locate
            # one core in each; their distance is the pair sep r(t).
            mid = nx // 2
            m_left = m_top[:, :mid, :]
            m_right = m_top[:, mid:, :]
            w_l = float((1.0 - m_left[..., 2]).sum())
            w_r = float((1.0 - m_right[..., 2]).sum())
            if w_l > 0.0 and w_r > 0.0:
                cx_l, cy_l = skyrmion_center_pbc(
                    m_left, p.a, core_polarity=+1)
                cx_r, cy_r = skyrmion_center_pbc(
                    m_right, p.a, core_polarity=+1)
                # Shift right-half x back into full-lattice coords.
                cx_r += mid * p.a
                r_now = math.hypot(cx_r - cx_l, cy_r - cy_l)
            else:
                r_now = float('nan')
                alive_both = False
            t_sample[s_idx] = step * dt
            r_pair[s_idx] = r_now
            Q_arr[s_idx] = float(
                topological_charge(m_top, p.a))
            s_idx += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    payload = {
        't_sample': t_sample[:s_idx],
        'r_pair': r_pair[:s_idx],
        'Q': Q_arr[:s_idx],
        'alive_both': bool(alive_both),
        'r_init': r_init,
        'ens_idx': ens_idx,
    }
    out_name = (
        f'r{r_init*1e9:06.1f}nm_ens{ens_idx:03d}.npz'
    )
    out_path = os.path.join(cfg['out_dir'], out_name)
    base_config = {
        'T_sub': float(cfg['T_sub']),
        'R_th': float(cfg['R_th']),
        'r_init': r_init,
        'nx': nx, 'ny': ny, 'dt': dt,
        'n_relax': n_relax, 'n_drive': n_drive,
        'sample_every': sample_every,
        'seed': seed,
        'tol_norm': tol_norm,
        'use_demag': use_demag,
    }
    save_trajectory(out_path, payload, base_config)
    return (
        f'  r_init={r_init*1e9:5.1f} nm  ens={ens_idx:03d}  '
        f'alive_both={alive_both}'
    )


# -----------------------------------------------------------------------------
def main():
    """Build the r_init x ensemble grid and dispatch it."""
    # =========================== User Configuration =========================
    r_init_list     = [180e-9, 240e-9, 320e-9, 400e-9]
    t_sub           = 300.0
    r_th            = 0.0
    nx              = 256
    ny              = 256
    dt              = 5.0e-14
    n_relax         = 4000           # 200 ps shape-relax
    n_drive         = 20000          # 1 ns drift
    sample_every    = 100            # 5 ps cadence
    n_ens           = 30
    seed_base       = 411
    tol_norm        = 5.0e-3
    use_demag       = True
    out_dir         = 'output/stochastic_llgs/pair_potential'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid = []
    for cell_idx, r_init in enumerate(r_init_list):
        for ens_idx in range(int(n_ens)):
            grid.append({
                'r_init': float(r_init),
                'ens_idx': int(ens_idx),
                'cell_idx': int(cell_idx),
            })
    cfg = {
        'T_sub': float(t_sub), 'R_th': float(r_th),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'n_relax': int(n_relax), 'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'seed_base': int(seed_base),
        'tol_norm': float(tol_norm),
        'use_demag': bool(use_demag),
        'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'pair_potential: SLURM_ARRAY_TASK_ID='
                f'{task_id} out of range '
                f'[0, {len(grid) - 1}].'
            )
        grid = [grid[idx]]
        n_proc = 1
    args_list = [(cfg, point) for point in grid]
    print(
        f'pair_potential: {len(args_list)} trajectories '
        f'(lattice={nx}x{ny}, demag={use_demag}, T={t_sub} K), '
        f'n_proc={n_proc}'
    )
    if n_proc <= 1:
        for args in args_list:
            print(_run_one_point(args), flush=True)
    else:
        with mp.Pool(n_proc) as pool:
            for result in pool.imap_unordered(
                    _run_one_point, args_list):
                print(result, flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
