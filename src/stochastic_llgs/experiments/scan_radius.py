"""Radius-dependence scan: vary (D, H_z) at fixed T, j.
Demag on.

Builds a flat grid
`[(D, H_z, ens_idx) for (D, H_z) in cells for ens_idx in
range(n_ens)]`. Each trajectory writes its own NPZ; v(R) and
P_surv(R) are aggregated downstream.

Execution modes
---------------
- Serial / multiprocess via `SWEEP_NPROC` (default 1).
- SLURM array: each task runs `grid[SLURM_ARRAY_TASK_ID]`.

Run with:
    python -m src.stochastic_llgs.experiments.scan_radius
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
from src.stochastic_llgs.io import save_trajectory
from src.stochastic_llgs.experiments.run_single import (
    trajectory_worker,
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


def _run_one_point(args):
    """Worker function: run one trajectory at (D, H_z) and save."""
    cfg, point = args
    D = float(point['D'])
    H_z = float(point['H_z'])
    ens_idx = int(point['ens_idx'])
    cell_idx = int(point['cell_idx'])
    # Decorrelate seeds: distinct stride per ensemble member and
    # per grid cell so no two trajectories share an RNG stream.
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * cell_idx
    base_config = {
        'T_sub': float(cfg['T_sub']),
        'R_th': float(cfg['R_th']),
        'j_current': float(cfg['j_current']),
        'nx': int(cfg['nx']), 'ny': int(cfg['ny']),
        'dt': float(cfg['dt']),
        'n_relax': int(cfg['n_relax']),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'seed': seed, 'traj_idx': ens_idx,
        'tol_norm': float(cfg['tol_norm']),
        'use_demag': bool(cfg['use_demag']),
        'q_threshold': float(cfg['q_threshold']),
        'k_consecutive': int(cfg['k_consecutive']),
        'param_overrides': {
            'D': D,
            'H_ext': np.array(
                [0.0, 0.0, H_z], dtype=float),
        },
        'dump_fields': False, 'snapshot_every': 0,
        'demag_kind': 'none',
        'demag_accuracy': None, 'demag_tol_conv': None,
        'm_init_top': None, 'm_init_bot': None,
        'equil': None,
    }
    payload = trajectory_worker(base_config)
    payload['D'] = D
    payload['H_z'] = H_z
    payload['ens_idx'] = ens_idx
    out_name = (
        f'D{D*1e3:.3f}_Hz{H_z:+.3f}_ens{ens_idx:03d}.npz'
    )
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trajectory(out_path, payload, base_config)
    # Late-time mean diameter (second half) as the steady-state
    # radius proxy for the progress line; NaN if no live track.
    if np.isfinite(payload['velocity']):
        diam = payload['diameter']
        half = max(1, diam.size // 2)
        d_mean = float(np.nanmean(diam[half:]))
    else:
        d_mean = float('nan')
    return (
        f'  D={D:.2e}  H_z={H_z:+.3f}  ens={ens_idx:03d}  '
        f'alive={payload["alive_at_end"]}  '
        f'<d>={d_mean*1e9:5.1f} nm  '
        f'v={payload["velocity"]:.1f}'
    )


# -----------------------------------------------------------------------------
def main():
    """Build the (D, H_z) x ensemble grid and dispatch it."""
    # =========================== User Configuration =========================
    # (D, H_z) cells targeting R values around
    # {40, 60, 80, 120} nm; retune after a first run.
    dh_cells = [
        (0.4e-3,  0.10),
        (0.5e-3,  0.05),
        (0.62e-3, 0.00),
        (0.8e-3, -0.05),
    ]
    t_sub           = 300.0
    j_current       = 4.0e11
    r_th            = 0.0
    nx              = 256
    ny              = 256
    dt              = 5.0e-14
    n_relax         = 10000          # 500 ps
    n_drive         = 40000          # 2 ns
    sample_every    = 200
    n_ens           = 30
    seed_base       = 311
    tol_norm        = 5.0e-3
    use_demag       = True
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = 'output/stochastic_llgs/scan_radius'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid = []
    for cell_idx, (D, H_z) in enumerate(dh_cells):
        for ens_idx in range(int(n_ens)):
            grid.append({
                'D': float(D), 'H_z': float(H_z),
                'ens_idx': int(ens_idx),
                'cell_idx': int(cell_idx),
            })
    cfg = {
        'T_sub': float(t_sub),
        'j_current': float(j_current),
        'R_th': float(r_th),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'n_relax': int(n_relax), 'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'seed_base': int(seed_base),
        'tol_norm': float(tol_norm),
        'use_demag': bool(use_demag),
        'q_threshold': float(q_threshold),
        'k_consecutive': int(k_consecutive),
        'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_radius: SLURM_ARRAY_TASK_ID={task_id} '
                f'out of range [0, {len(grid) - 1}].'
            )
        grid = [grid[idx]]
        n_proc = 1
    args_list = [(cfg, point) for point in grid]
    print(
        f'scan_radius: {len(args_list)} trajectories '
        f'(lattice={nx}x{ny}, demag={use_demag}, '
        f'T={t_sub} K, j={j_current:.2e}), n_proc={n_proc}'
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
