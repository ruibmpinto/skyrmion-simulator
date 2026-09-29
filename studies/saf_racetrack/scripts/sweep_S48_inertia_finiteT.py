"""Figure S48 at finite temperature: inertia from a pulse with an
off-phase.

Finite-T counterpart of `studies/saf_racetrack/scripts/sweep_S48_inertia.py`,
which stays as the T = 0 reference and is not modified. Two deliberate
departures from
it, both required here:

- The H_RKKY axis is dropped, since this study does not vary the
  interlayer coupling. What remains is the inertial response against
  temperature and current.
- Set A replaces the reference's Set B (K_top = 1.294e6, alpha = 0.216,
  gamma = 175.9e9), so this sweep sits at the same 36 mT anisotropy and
  damping as the rest of the study. It will therefore NOT reproduce
  Pham S48 quantitatively; that deviation belongs in the report.

Why this sweep cannot reuse the campaign: it measures the rise AND the
fall -- the onset transient, the decay time 1/tau after the current
stops, and any ringing. The campaign drove DC for 2 ns and never turned
off, so the fall is simply absent from that data. Here a square pulse
occupies the first part of the window and the rest is unforced, which
is what makes the decay measurable.

Reference/pilot path: production ensembles run through the C++ driver
`sweep_pulse_shape.cpp` with the square shape.

Execution modes and configuration follow the S41-S49 convention:
serial, `SWEEP_NPROC` pool, or a single `SLURM_ARRAY_TASK_ID` point; all
run-time values are top-of-`main()` variables, no argparse.

Run with:
    SWEEP_NPROC=4 python -m \
    studies.saf_racetrack.scripts.sweep_S48_inertia_finiteT
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
from studies.saf_racetrack.orchestrator.pulsed_run import run_pulsed_point

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
    seed_root = ('output/'
                 'stochastic_llgs/scan_track_width/campaign')
    out_root = 'output/sweeps_driving_T/S48_inertia_py'
    t_sub_list = [0.0, 10.0, 50.0, 100.0]
    # Peak current densities inside the no-labyrinth window; narrow or
    # extend using the pulsed-stability pilot table.
    peak_j_list = [0.5e11, 1.0e11, 2.0e11, 3.0e11]
    n_ens = 4                      # reference/pilot only, not production
    dt = 5.0e-14
    # Square pulse on for t_window, then an equally long unforced
    # interval: the decay after switch-off is the quantity of interest,
    # so the off-phase is as long as the on-phase.
    t_window = 1000.0e-12
    t_settle = 1000.0e-12
    # Unused by the square shape but required by make_pulse, so a
    # Gaussian can never be selected with an implicit width.
    gauss_fwhm = 500.0e-12
    # Finer sampling than the other sweeps: 1/tau and any ringing are
    # read off the transient, which 10 ps steps would smear.
    sample_dt = 2.0e-12
    r_th = 0.0
    tol_norm = 5.0e-3
    demag_kind = 'racetrack'       # periodic x, free y
    demag_accuracy = 4.0
    demag_tol_conv = 0.02
    seed_base = 505
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
        'sweep_tag': 'S48_inertia',
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flat per-trajectory grid. A deterministic point has exactly one
    # member; ensemble copies of a noise-free run would be duplicates.
    grid = []
    cell_idx = 0
    for peak_j, t_sub in itertools.product(peak_j_list, t_sub_list):
        members = 1 if t_sub == 0.0 else n_ens
        for ens in range(members):
            grid.append({'shape': 'square', 'peak_j': peak_j,
                         'T_sub': t_sub, 'ens': ens,
                         'cell_idx': cell_idx})
        cell_idx += 1
    print(f'S48 finite-T grid: {len(grid)} trajectories over {cell_idx} '
          f'cells (box {nx}x{ny}, tag {box_tag})')
    print(f'  square pulse on for {1e12*t_window:.0f} ps then '
          f'{1e12*t_settle:.0f} ps unforced = {n_drive} steps, '
          f'sampling every {sample_every} steps')
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
