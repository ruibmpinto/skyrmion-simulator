"""Pulse-shape driving sweep at finite temperature (reference/pilot).

No counterpart in Pham et al.; this sweep is new. It drives a
pre-equilibrated SAF skyrmion with each of six current-pulse shapes --
square, half-sine, the three triangular asymmetries and Gaussian -- at
matched peak amplitude and duration, on the racetrack geometry the
finite-T campaign uses (periodic x, free y, D = 0.72 mJ/m^2, Set A,
H_k,top = 36 mT).

This is the Python reference for `sweep_pulse_shape.cpp`: same seeds,
same conventions, same stepper, so a point run in both languages is
directly comparable. Production ensembles go through the C++ driver;
use this to cross-check it and to scout small grids.

The simulated window is the pulse duration plus a settle interval, so
the skyrmion is observed coming to rest once the drive ends.

Execution modes, as in the S41-S49 scripts:

- Serial loop (default, `SWEEP_NPROC = 1`).
- `multiprocessing.Pool` when `SWEEP_NPROC > 1`.
- Single grid point when `SLURM_ARRAY_TASK_ID` is set.

All run-time values are top-of-`main()` variables; no argparse.

Run with:
    SWEEP_NPROC=4 python -m scripts.sweep_pulse_shape_finiteT
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools
import math
import multiprocessing as mp
import os
# Local
from src.orchestrator.pulsed_run import run_pulsed_point, shape_names

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
    """Worker entry point: integrate one (cfg, point) pair.

    Parameters
    ----------
    args : tuple(dict, dict)
        `(cfg, point)` as built in `main()`.

    Returns
    -------
    summary : str
        One-line console summary printed by the parent.
    """
    cfg, point = args
    return run_pulsed_point(cfg, point)


# -----------------------------------------------------------------------------
def main():
    """Build the (shape, peak J, T_sub) x ensemble grid and dispatch it."""
    # =========================== User Configuration =========================
    # Box and material: the campaign case this study seeds from.
    nx = 350
    ny = 500
    dmi = 0.72e-3                  # J/m^2
    k_top = 1.3106e6               # J/m^3, H_k,top = 36 mT (Set A)
    box_tag = 'hk36_D0p72_350x500'
    # Campaign directory holding m_eq / m_thermal. Read-only here.
    seed_root = ('/Volumes/T7/skyrmion_simulator/output/'
                 'stochastic_llgs/scan_track_width/campaign')
    out_root = 'output/sweeps_driving_T/pulse_shape_py'
    # T_sub = 0 is the deterministic baseline, the rest are thermal.
    t_sub_list = [0.0, 10.0, 50.0, 100.0]
    # Peak current densities inside the no-labyrinth window; narrow or
    # extend using the pulsed-stability pilot table.
    peak_j_list = [0.5e11, 1.0e11, 2.0e11, 3.0e11]
    n_ens = 4                      # reference/pilot only, not production
    dt = 5.0e-14
    t_pulse = 500.0e-12
    t_settle = 500.0e-12
    # Gaussian arm width, absolute so it is independent of the window.
    gauss_fwhm = 250.0e-12
    sample_dt = 10.0e-12
    r_th = 0.0
    tol_norm = 5.0e-3
    # Racetrack: periodic x, free y, matching the campaign.
    demag_kind = 'racetrack'
    demag_accuracy = 4.0
    demag_tol_conv = 0.02
    seed_base = 202
    # ======================= End User Configuration =========================
    n_drive = int(math.ceil((t_pulse + t_settle)/dt))
    sample_every = int(math.ceil(sample_dt/dt))
    cfg = {
        'nx': nx, 'ny': ny, 'dt': dt,
        'D': dmi, 'K_top': k_top,
        'demag_kind': demag_kind,
        'demag_accuracy': demag_accuracy,
        'demag_tol_conv': demag_tol_conv,
        'n_drive': n_drive,
        'sample_every': sample_every,
        'tol_norm': tol_norm,
        'r_th': r_th,
        't_pulse': t_pulse,
        'gauss_fwhm': gauss_fwhm,
        'seed_dir': os.path.join(seed_root, box_tag),
        'out_dir': os.path.join(out_root, box_tag),
        'seed_base': seed_base,
        'sweep_tag': 'pulse_shape',
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flat per-trajectory grid. A deterministic point has exactly one
    # member; ensemble copies of a noise-free run would be duplicates.
    grid = []
    cell_idx = 0
    for shape, peak_j, t_sub in itertools.product(
            shape_names(), peak_j_list, t_sub_list):
        members = 1 if t_sub == 0.0 else n_ens
        for ens in range(members):
            grid.append({'shape': shape, 'peak_j': peak_j,
                         'T_sub': t_sub, 'ens': ens,
                         'cell_idx': cell_idx})
        cell_idx += 1
    print(f'pulse-shape grid: {len(grid)} trajectories over {cell_idx} '
          f'cells (box {nx}x{ny}, tag {box_tag})')
    print(f'  window {1e12*(t_pulse + t_settle):.0f} ps = '
          f'{n_drive} steps, sampling every {sample_every} steps')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dispatch: single array task, worker pool, or serial loop.
    task_env = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_env is not None and task_env != '':
        idx = int(task_env)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'SLURM_ARRAY_TASK_ID={idx} outside '
                f'[0, {len(grid) - 1}].')
        print(_run_one_point((cfg, grid[idx])), flush=True)
        return
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    if n_proc < 1:
        raise RuntimeError(f'SWEEP_NPROC must be >= 1, got {n_proc}.')
    args_list = [(cfg, point) for point in grid]
    if n_proc == 1:
        for args in args_list:
            print(_run_one_point(args), flush=True)
        return
    with mp.Pool(processes=n_proc) as pool:
        for summary in pool.imap_unordered(_run_one_point, args_list):
            print(summary, flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
