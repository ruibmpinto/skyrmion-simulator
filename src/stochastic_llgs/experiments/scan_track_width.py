"""Skyrmion length-scale vs track-width (T_sub, j) scan.

Drives a SAF skyrmion at steady current on a rectangular box
sized to the elliptically deformed skyrmion under drive,
`L_x = 4 D_1` by `L_y = 4 D_2` (the transverse extent `L_y` is
the track width). For each `(T_sub, j)` cell an ensemble of
`n_ens` independent trajectories is run; the elliptical axes
`D_1`, `D_2` recorded per trajectory quantify how the skyrmion
length scale grows with current density and temperature
relative to the fixed track width.

Set-A (D = 0.85e-3) has quality factor Q approximately 1, so the
local-anisotropy model cannot confine the skyrmion (it expands
to the box). The skyrmion is only stable with full NEWELL FFT
demag, exactly as the S47 deformation sweep. The equilibrium is
therefore relaxed ONCE with the deterministic convergence-stop
`relax()` under newell demag, written to `m_eq.npz`, and every
trajectory is seeded from that field and driven with newell
demag in the stochastic stepper.

Box from the stored S47 deformation data: equilibrium diameter
~187 nm, maximum coherent (intact) deformation D_1 ~ 245 nm,
D_2 ~ 174 nm. Rounded up to D_1 = 250 nm, D_2 = 175 nm gives
a rectangular box L_x = 700 nm (nx = 350, along motion) by
L_y = 1000 nm (ny = 500, transverse = track width) at a = 2 nm.

One designated realization per cell (`ens_idx == 0`) additionally
dumps the full magnetization fields: the initial / relaxed /
final configurations plus an `m_z` animation-frame stack, split
into a separate `anim_*.npz` so the per-trajectory files stay
light for aggregation.

Execution modes
---------------
- Serial         : run every grid point in sequence.
- Multiprocess   : set `SWEEP_NPROC=N` to dispatch via `mp.Pool`.
- SLURM array    : set `SLURM_ARRAY_TASK_ID=I`; the script runs
                   only `grid[I]` and ignores `SWEEP_NPROC`. The
                   equilibrium `m_eq.npz` must already exist
                   (relax it once before dispatching the array).

Run with:
    python -m src.stochastic_llgs.experiments.scan_track_width
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import multiprocessing as mp
import os
# Third-party
import numpy as np
# Local
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.pulses import ConstantPulse
from src.stochastic_llgs.io import save_trajectory
from src.stochastic_llgs.joule_heating import T_of_j
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.experiments.run_single import (
    trajectory_worker, _equilibrate,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _relax_equilibrium(cfg):
    """Relax the SAF skyrmion to its newell-demag equilibrium.

    Deterministic convergence-stop relaxation at the production
    box; the result seeds every trajectory.

    Parameters
    ----------
    cfg : dict
        Shared run configuration (box, demag, relax tolerances).

    Returns
    -------
    m_eq_top : numpy.ndarray(3d)
        Relaxed top-layer field, shape (ny, nx, 3).
    m_eq_bot : numpy.ndarray(3d)
        Relaxed bottom-layer field.
    """
    p = make_params(nx=int(cfg['nx']), ny=int(cfg['ny']),
                    D=float(cfg['D']))
    p.dt = float(cfg['dt'])
    kernels = precompute_demag_kernels(
        p, kind=str(cfg['demag_kind']),
        accuracy=float(cfg['demag_accuracy']),
        tol_conv=float(cfg['demag_tol_conv']))
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
    m_eq_top, m_eq_bot, conv, n_used, _E, tau = relax(
        m_top=m_top, m_bot=m_bot, p=p, kernels=kernels,
        max_steps=int(cfg['relax_max_steps']),
        alpha_relax=float(cfg['relax_alpha']),
        tol_torque=float(cfg['relax_tol_torque']),
        tol_dE=float(cfg['relax_tol_dE']),
        check_every=int(cfg['relax_check_every']),
        print_every=0, mask=None)
    print(
        f'm_eq relax: converged={conv} n_steps={n_used} '
        f'tau_max={tau:.2e} T'
    )
    return m_eq_top, m_eq_bot


# -----------------------------------------------------------------------------
def _thermalize(cfg, t_idx, T_sub, ens, m_eq_top, m_eq_bot):
    """Equilibrate the shared T=0 field at temperature `T_sub`.

    Runs the J=0 noisy dynamics from the cold equilibrium until the
    LCC size plateaus. Computed once per (T_sub, ens) and reused by the
    six current drives of that temperature/realization.

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Thermally equilibrated fields at `T_sub`.
    """
    p = make_params(nx=int(cfg['nx']), ny=int(cfg['ny']),
                    D=float(cfg['D']))
    p.dt = float(cfg['dt'])
    kernels = precompute_demag_kernels(
        p, kind=str(cfg['demag_kind']),
        accuracy=float(cfg['demag_accuracy']),
        tol_conv=float(cfg['demag_tol_conv']))
    # Decorrelated thermalization seed per (T, ens); distinct from the
    # drive seed (which also folds in the current).
    seed = int(cfg['seed_base']) + 1000 * int(ens) \
        + 1000_000 * int(t_idx)
    attach_thermal(p, T=float(T_sub), R_th=float(cfg['R_th']), seed=seed)
    p.pulse = ConstantPulse(0.0)
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    m_top = np.array(m_eq_top, dtype=float)
    m_bot = np.array(m_eq_bot, dtype=float)
    m_top, m_bot, n_used, conv, d1, d2 = _equilibrate(
        m_top, m_bot, p, kernels, float(cfg['dt']), rng, sigma,
        float(cfg['tol_norm']), dict(cfg['equil']))
    print(
        f'  thermalize T={T_sub:5.1f} ens={ens:03d}: '
        f'n={n_used} conv={conv} '
        f'D1={d1*1e9:.1f} D2={d2*1e9:.1f} nm', flush=True)
    return m_top, m_bot


# -----------------------------------------------------------------------------
def _run_one_point(args):
    """Worker function: run one trajectory and save its NPZ.

    Each trajectory is seeded from the shared equilibrium field
    and driven with newell demag. The `ens_idx == 0` member of
    each cell dumps full-field data, split into a dedicated
    `anim_*.npz`.

    Parameters
    ----------
    args : tuple(dict, dict)
        (config, point). `config` carries the shared run
        parameters and the equilibrium field; `point` carries
        the cell coordinates and ensemble index.

    Returns
    -------
    msg : str
        Short progress line for the orchestrator to print.
    """
    cfg, point = args
    T_sub = float(point['T_sub'])
    j = float(point['j_current'])
    ens_idx = int(point['ens_idx'])
    t_idx = int(point['t_idx'])
    # Decorrelate seeds: distinct stride per ensemble member and
    # per grid cell so no two trajectories share an RNG stream.
    # Stage offset 1e9 keeps the drive streams disjoint from the
    # equilibrate-stage seeds (cell_idx overlaps t_idx there).
    seed = 1_000_000_000 + int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * int(point['cell_idx'])
    # Only the first ensemble member dumps full fields.
    dump_fields = bool(ens_idx == 0)
    snapshot_every = int(cfg['snapshot_every']) if dump_fields else 0
    # Seed from the shared per-(T, ens) thermalized state; no
    # equilibration in the drive (equil=None, n_relax=0).
    m_th_top, m_th_bot = cfg['thermal_cache'][(t_idx, ens_idx)]
    base_config = {
        'T_sub': T_sub, 'R_th': float(cfg['R_th']),
        'j_current': j,
        'nx': int(cfg['nx']), 'ny': int(cfg['ny']),
        'dt': float(cfg['dt']),
        'n_relax': 0,
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'seed': seed, 'traj_idx': ens_idx,
        'tol_norm': float(cfg['tol_norm']),
        'use_demag': True,
        'q_threshold': float(cfg['q_threshold']),
        'k_consecutive': int(cfg['k_consecutive']),
        'param_overrides': {'D': float(cfg['D'])},
        'dump_fields': dump_fields,
        'snapshot_every': snapshot_every,
        'demag_kind': str(cfg['demag_kind']),
        'demag_accuracy': float(cfg['demag_accuracy']),
        'demag_tol_conv': float(cfg['demag_tol_conv']),
        'm_init_top': m_th_top,
        'm_init_bot': m_th_bot,
        'equil': None,
    }
    payload = trajectory_worker(base_config)
    T_eff = T_of_j(j, T_sub=T_sub, R_th=cfg['R_th'])
    payload['T_sub'] = T_sub
    payload['j_current'] = j
    payload['ens_idx'] = ens_idx
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Split the heavy full-field dumps into a separate file so the
    # per-trajectory NPZ consumed by aggregation stays light.
    if dump_fields:
        dump_keys = [
            k for k in list(payload)
            if k.startswith('field_') or k.startswith('anim_')
        ]
        anim_payload = {k: payload.pop(k) for k in dump_keys}
        anim_name = f'anim_T{T_sub:05.1f}_j{j:.2e}.npz'
        anim_path = os.path.join(cfg['out_dir'], anim_name)
        save_trajectory(anim_path, anim_payload, base_config)
    out_name = (
        f'T{T_sub:05.1f}_j{j:.2e}_ens{ens_idx:03d}.npz'
    )
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trajectory(out_path, payload, base_config)
    return (
        f'  T_sub={T_sub:5.1f}  j={j:.2e}  ens={ens_idx:03d}  '
        f'T_eff={T_eff:5.1f}  '
        f'alive={payload["alive_at_end"]}  '
        f'v={payload["velocity"]:.1f}'
    )


# -----------------------------------------------------------------------------
def main():
    """Build the (T_sub, j) x ensemble grid and dispatch it."""
    # =========================== User Configuration =========================
    # Rectangular box: L_x = 700 nm (along motion), L_y = 1000 nm
    # (transverse = track width) at a = 2 nm.
    nx              = 350
    ny              = 500
    # Current density: intact -> deformation -> burst onset under
    # steady drive.
    j_list          = [
        0.5e11, 1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e11,
    ]
    # Substrate temperatures spanning the ~200 K Tomasello regime
    # split.
    t_sub_list      = [
        10.0, 50.0, 100.0, 130.0, 160.0, 200.0,
    ]
    r_th            = 0.0            # isolate substrate T
    dt              = 5.0e-14
    # n_relax is unused under equilibrate-to-plateau (equil set below).
    n_relax         = 0
    n_drive         = 40000          # 2 ns
    sample_every    = 200            # 10 ps between samples
    snapshot_every  = 400            # 20 ps between anim frames
    n_ens           = 100
    # Thermal equilibration at each T before driving: run the J=0
    # noisy dynamics until the LCC size plateaus (per trajectory,
    # seeded from the shared T=0 m_eq).
    equil_check_every = 200          # 10 ps between size checks
    equil_window      = 10           # checks per window (100 ps)
    equil_tol         = 0.02         # relative size-change tolerance
    equil_k_consec    = 3            # consecutive stable windows
    equil_max_steps   = 300000       # 15 ns cap
    seed_base       = 101
    tol_norm        = 5.0e-3
    q_threshold     = 0.5
    k_consecutive   = 10
    # Racetrack: periodic x, free top/bottom (y) for both demag and
    # exchange/DMI over the full box; the track width is the transverse
    # box extent L_y = ny*a. DMI just below D_c for a stable compact
    # skyrmion. Must match the C++ campaign drivers.
    demag_kind      = 'racetrack'
    dmi             = 0.47e-3
    demag_accuracy  = 4.0
    demag_tol_conv  = 0.02
    # Deterministic equilibrium relaxation (matches S47).
    relax_max_steps = 200000
    relax_alpha     = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE    = 1.0e-8
    relax_check_every = 1000
    out_dir         = 'output/stochastic_llgs/scan_track_width'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    cfg = {
        'R_th': float(r_th), 'nx': int(nx), 'ny': int(ny),
        'dt': float(dt), 'n_relax': int(n_relax),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'snapshot_every': int(snapshot_every),
        'seed_base': int(seed_base),
        'tol_norm': float(tol_norm),
        'q_threshold': float(q_threshold),
        'k_consecutive': int(k_consecutive),
        'demag_kind': str(demag_kind),
        'D': float(dmi),
        'demag_accuracy': float(demag_accuracy),
        'demag_tol_conv': float(demag_tol_conv),
        'equil': {
            'check_every': int(equil_check_every),
            'window': int(equil_window),
            'tol': float(equil_tol),
            'k_consec': int(equil_k_consec),
            'max_steps': int(equil_max_steps),
        },
        'relax_max_steps': int(relax_max_steps),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
        'relax_check_every': int(relax_check_every),
        'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Equilibrium: load the cached newell-relaxed field, or relax
    # once and cache it. Every trajectory seeds from this field.
    m_eq_path = os.path.join(out_dir, 'm_eq.npz')
    if os.path.isfile(m_eq_path):
        z = np.load(m_eq_path)
        m_eq_top = z['m_eq_top']
        m_eq_bot = z['m_eq_bot']
        print(f'loaded equilibrium from {m_eq_path}')
    else:
        m_eq_top, m_eq_bot = _relax_equilibrium(cfg)
        np.savez_compressed(
            m_eq_path, m_eq_top=m_eq_top, m_eq_bot=m_eq_bot)
        print(f'saved equilibrium to {m_eq_path}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Thermal cache: equilibrate once per (T, ens) and reuse across the
    # current drives of that temperature (the J=0 equilibrium does not
    # depend on the drive current).
    thermal_cache = {}
    for t_idx, T_sub in enumerate(t_sub_list):
        for ens_idx in range(int(n_ens)):
            thermal_cache[(t_idx, ens_idx)] = _thermalize(
                cfg, t_idx, float(T_sub), ens_idx, m_eq_top, m_eq_bot)
    cfg['thermal_cache'] = thermal_cache
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flat grid: one entry per trajectory.
    grid = []
    cell_idx = 0
    for t_idx, T_sub in enumerate(t_sub_list):
        for j in j_list:
            for ens_idx in range(int(n_ens)):
                grid.append({
                    'T_sub': float(T_sub),
                    'j_current': float(j),
                    'ens_idx': int(ens_idx),
                    'cell_idx': int(cell_idx),
                    't_idx': int(t_idx),
                })
            cell_idx += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # HPC array dispatch / local multiprocess fallback.
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_track_width: SLURM_ARRAY_TASK_ID={task_id} '
                f'out of range [0, {len(grid) - 1}].'
            )
        grid = [grid[idx]]
        n_proc = 1
    args_list = [(cfg, point) for point in grid]
    print(
        f'scan_track_width: {len(args_list)} trajectories '
        f'(lattice={nx}x{ny}, n_ens={n_ens}, '
        f'demag={demag_kind}), n_proc={n_proc}'
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
