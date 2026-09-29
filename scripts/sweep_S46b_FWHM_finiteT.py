"""Figure S46b at finite temperature: Gaussian FWHM at fixed current.

Finite-T counterpart of `scripts/sweep_S46b_FWHM.py`, which stays as the
T = 0 reference in its original geometry and is not modified. This sweep
re-runs the FWHM axis at a single peak current on the finite-T campaign
geometry (350x500 racetrack, periodic x, free y, D = 0.72 mJ/m^2,
Set A) across temperatures, T = 0 included.

It differs from S43a only in holding the current fixed and sampling the
width more finely, which is what the reference did: S43a crosses width
with current, S46b isolates the width response. The reference's
J = 8.9e11 is far outside the no-labyrinth window at finite T, so the
current is 2e11 instead, chosen from the measured stability maps as the
largest value that stays compact at every temperature in range -- a
deliberate departure to record in the report.

The window is held fixed while the FWHM varies, so every point has the
same observation span and the same settle interval after the pulse; only
the width of the drive changes.

Reference/pilot path: production ensembles run through the C++ driver
`sweep_pulse_shape.cpp`, which reaches the same grid far faster and also
writes the per-configuration field-dump streams the GIFs are built from.

Execution modes and configuration follow the S41-S49 convention:
serial, `SWEEP_NPROC` pool, or a single `SLURM_ARRAY_TASK_ID` point; all
run-time values are top-of-`main()` variables, no argparse.

Run with:
    SWEEP_NPROC=4 python -m scripts.sweep_S46b_FWHM_finiteT
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
from src.orchestrator.pulsed_run import run_pulsed_point

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
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
    """Build the (peak J, T_sub) x ensemble grid and dispatch it."""
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
    out_root = 'output/sweeps_driving_T/S46b_FWHM_py'
    t_sub_list = [0.0, 10.0, 50.0, 100.0]
    # Peak current densities inside the no-labyrinth window; narrow or
    # extend using the pulsed-stability pilot table.
    # Single current: the width response in isolation. 2e11 is the
    # largest current that keeps the state COMPACT at every temperature
    # up to 100 K -- from docs/stability_box_size_hk36.pdf (Fig. 4,
    # converged L_x = 2800 nm box): P(S) = 1.00 / 0.99 / 0.88 at
    # T = 10 / 50 / 100 K with only 0.01 labyrinth in the worst cell.
    # 3e11 is already fully elongated by 10 K and carries 0.07 L at
    # 100 K; 4e11 is entirely labyrinth at 100 K. The reference's
    # 8.9e11 is far outside the window -- a deliberate departure to
    # record in the report. DC 2 ns figures, so conservative for a
    # pulse, which delivers less charge at the same peak.
    peak_j_list = [2.0e11]
    n_ens = 4                      # reference/pilot only, not production
    dt = 5.0e-14
    # Fixed window; the FWHM axis is swept per point below, so the
    # observation span and settle interval are the same everywhere.
    t_window = 1000.0e-12
    gauss_fwhm_list = [
        100.0e-12, 150.0e-12, 200.0e-12, 300.0e-12, 400.0e-12,
        500.0e-12, 600.0e-12, 700.0e-12,
    ]
    # cfg carries a width too, but every point overrides it; keeping it
    # explicit means make_pulse can never see an implicit value.
    gauss_fwhm = 500.0e-12
    t_settle = 500.0e-12           # observed after the pulse
    sample_dt = 10.0e-12
    r_th = 0.0
    tol_norm = 5.0e-3
    demag_kind = 'racetrack'       # periodic x, free y
    demag_accuracy = 4.0
    demag_tol_conv = 0.02
    seed_base = 707
    # ======================= End User Configuration =========================
    n_drive = int(math.ceil((t_window + t_settle)/dt))
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
        't_pulse': t_window,
        'gauss_fwhm': gauss_fwhm,
        'seed_dir': os.path.join(seed_root, box_tag),
        'out_dir': os.path.join(out_root, box_tag),
        'seed_base': seed_base,
        'sweep_tag': 'S46b_FWHM',
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flat per-trajectory grid. A deterministic point has exactly one
    # member; ensemble copies of a noise-free run would be duplicates.
    grid = []
    cell_idx = 0
    for fwhm, peak_j, t_sub in itertools.product(
            gauss_fwhm_list, peak_j_list, t_sub_list):
        members = 1 if t_sub == 0.0 else n_ens
        for ens in range(members):
            grid.append({'shape': 'gaussian', 'peak_j': peak_j,
                         'T_sub': t_sub, 'ens': ens,
                         'gauss_fwhm': fwhm, 'cell_idx': cell_idx})
        cell_idx += 1
    print(f'S46b finite-T grid: {len(grid)} trajectories over '
          f'{cell_idx} cells (box {nx}x{ny}, tag {box_tag})')
    print(f'  FWHM {[f"{1e12*f:.0f}" for f in gauss_fwhm_list]} ps x '
          f'{len(peak_j_list)} J x {len(t_sub_list)} T, '
          f'window {1e12*(t_window + t_settle):.0f} ps = {n_drive} steps')
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
