"""(T_sub, j) production scan for the stochastic SAF skyrmion
simulator. Demag off.

Builds a flat grid of trajectories
`[(T_sub, j, ens_idx) for T_sub in t_subs for j in js for
ens_idx in range(n_ens)]`. Each trajectory writes its own NPZ;
aggregation across the ensemble is performed downstream by a
plot/analysis script that reads all per-task files.

Execution modes
---------------
- Serial         : run every grid point in sequence.
- Multiprocess   : set `SWEEP_NPROC=N` to dispatch via `mp.Pool`.
- SLURM array    : set `SLURM_ARRAY_TASK_ID=I`; the script runs
                   only `grid[I]` and ignores `SWEEP_NPROC`. This
                   matches the pattern used by `sweep_S41_v_time`,
                   `sweep_S49_TSH`, etc.

Run with:
    python -m src.stochastic_llgs.experiments.scan_tj
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools
import multiprocessing as mp
import os
# Third-party
# Local
from src.stochastic_llgs.io import save_trajectory
from src.stochastic_llgs.joule_heating import T_of_j
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
    """Worker function: run one trajectory and save its NPZ.

    Parameters
    ----------
    args : tuple(dict, dict)
        (config, point). `config` carries the shared run
        parameters; `point` carries the cell coordinates
        and ensemble index.

    Returns
    -------
    msg : str
        Short progress line for the orchestrator to print.
    """
    cfg, point = args
    T_sub = float(point['T_sub'])
    j = float(point['j_current'])
    ens_idx = int(point['ens_idx'])
    # Decorrelate seeds: distinct stride per ensemble member and
    # per grid cell so no two trajectories share an RNG stream.
    seed = int(cfg['seed_base']) + 1000 * ens_idx \
        + 1000_000 * int(point['cell_idx'])
    base_config = {
        'T_sub': T_sub, 'R_th': float(cfg['R_th']),
        'j_current': j,
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
    T_eff = T_of_j(j, T_sub=T_sub, R_th=cfg['R_th'])
    payload['T_sub'] = T_sub
    payload['j_current'] = j
    payload['ens_idx'] = ens_idx
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
    t_sub_list      = [
        10.0, 25.0, 50.0, 100.0, 150.0,
        200.0, 250.0, 300.0, 350.0,
    ]
    j_list          = [
        1.0e11, 2.0e11, 4.0e11, 8.0e11, 1.6e12,
    ]
    r_th            = 0.0
    nx              = 256
    ny              = 256
    dt              = 5.0e-14
    n_relax         = 10000          # 500 ps
    n_drive         = 40000          # 2 ns
    sample_every    = 200            # 10 ps between samples
    n_ens           = 30
    seed_base       = 101
    tol_norm        = 5.0e-3
    use_demag       = False
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = 'output/stochastic_llgs/scan_tj'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flat grid: one entry per trajectory.
    cells = list(
        itertools.product(t_sub_list, j_list)
    )
    grid = []
    for cell_idx, (T_sub, j) in enumerate(cells):
        for ens_idx in range(int(n_ens)):
            grid.append({
                'T_sub': float(T_sub),
                'j_current': float(j),
                'ens_idx': int(ens_idx),
                'cell_idx': int(cell_idx),
            })
    cfg = {
        'R_th': float(r_th), 'nx': int(nx), 'ny': int(ny),
        'dt': float(dt), 'n_relax': int(n_relax),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'seed_base': int(seed_base),
        'tol_norm': float(tol_norm),
        'use_demag': bool(use_demag),
        'q_threshold': float(q_threshold),
        'k_consecutive': int(k_consecutive),
        'out_dir': out_dir,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # HPC array dispatch / local multiprocess fallback.
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'scan_tj: SLURM_ARRAY_TASK_ID={task_id} out '
                f'of range [0, {len(grid) - 1}].'
            )
        grid = [grid[idx]]
        n_proc = 1
    args_list = [(cfg, point) for point in grid]
    print(
        f'scan_tj: {len(args_list)} trajectories '
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
