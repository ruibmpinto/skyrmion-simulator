"""Zero-drive Arrhenius scan: tau(T) at j = 0. Demag off.

Builds a flat grid `[(T_sub, ens_idx) for T_sub in t_subs for
ens_idx in range(n_ens)]`. Each trajectory writes its own NPZ;
the Arrhenius fit `log(tau) vs 1/T` is performed by a
downstream analysis script that reads all per-task files.

Execution modes
---------------
- Serial / multiprocess via `SWEEP_NPROC` (default 1).
- SLURM array: each task runs `grid[SLURM_ARRAY_TASK_ID]`.

Run with:
    python -m src.stochastic_llgs.experiments.scan_arrhenius
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
    """Worker function: run one zero-drive trajectory and save."""
    cfg, point = args
    T_sub = float(point['T_sub'])
    ens_idx = int(point['ens_idx'])
    # Decorrelate seeds: distinct stride per ensemble member and
    # per grid cell so no two trajectories share an RNG stream.
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * int(point['cell_idx'])
    base_config = {
        'T_sub': T_sub, 'R_th': float(cfg['R_th']),
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
        'param_overrides': {},
        'dump_fields': False, 'snapshot_every': 0,
        'demag_kind': 'none',
        'demag_accuracy': None, 'demag_tol_conv': None,
        'm_init_top': None, 'm_init_bot': None,
        'equil': None,
    }
    payload = trajectory_worker(base_config)
    flip_idx = int(payload['flip_index'])
    # Convert the sample-index flip to a physical dwell time tau;
    # NaN means the skyrmion survived the whole observation window.
    if flip_idx == -1:
        t_flip = float('nan')
    else:
        t_flip = flip_idx * int(cfg['sample_every']) \
            * float(cfg['dt'])
    payload['T_sub'] = T_sub
    payload['ens_idx'] = ens_idx
    payload['t_flip'] = t_flip
    out_name = (
        f'T{T_sub:05.1f}_ens{ens_idx:03d}.npz'
    )
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trajectory(out_path, payload, base_config)
    flip_tag = 'survived' if flip_idx == -1 \
        else f't_flip={t_flip*1e9:.2f}ns'
    return (
        f'  T={T_sub:5.1f}  ens={ens_idx:03d}  '
        f'flip_index={flip_idx:6d}  {flip_tag}'
    )


# -----------------------------------------------------------------------------
def main():
    """Build the (T_sub) x ensemble grid and dispatch it."""
    # =========================== User Configuration =========================
    t_sub_list      = [
        10.0, 25.0, 50.0, 100.0, 150.0,
        200.0, 250.0, 300.0, 350.0,
    ]
    r_th            = 0.0            # j = 0 means no heating
    j_current       = 0.0
    nx              = 256
    ny              = 256
    dt              = 5.0e-14
    n_relax         = 4000           # 200 ps thermalization
    n_drive         = 200000         # 10 ns observation window
    sample_every    = 100            # 5 ps cadence
    n_ens           = 200
    seed_base       = 211
    tol_norm        = 5.0e-3
    use_demag       = False
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = 'output/stochastic_llgs/scan_arrhenius'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    grid = []
    for cell_idx, T_sub in enumerate(t_sub_list):
        for ens_idx in range(int(n_ens)):
            grid.append({
                'T_sub': float(T_sub),
                'ens_idx': int(ens_idx),
                'cell_idx': int(cell_idx),
            })
    cfg = {
        'R_th': float(r_th), 'j_current': float(j_current),
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
                f'scan_arrhenius: SLURM_ARRAY_TASK_ID='
                f'{task_id} out of range '
                f'[0, {len(grid) - 1}].'
            )
        grid = [grid[idx]]
        n_proc = 1
    args_list = [(cfg, point) for point in grid]
    print(
        f'scan_arrhenius: {len(args_list)} trajectories '
        f'(lattice={nx}x{ny}, n_ens={n_ens}, '
        f'demag={use_demag}), n_proc={n_proc}'
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
