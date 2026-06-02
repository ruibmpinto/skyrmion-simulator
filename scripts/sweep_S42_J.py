"""Sweep driver for Pham et al. (2024) Figure S42.

Nine Gaussian current pulses with FWHM = 500 ps at varying
peak J. For each J in `J_values`, one trace is written to
`output/sweeps_S41_S49/S42/J_<J0>.npz`.

The script supports three execution modes, selected at the top
of `main()`:

- Serial loop (default, `SWEEP_NPROC = 1`): one grid point
  after another.
- `multiprocessing.Pool` parallelism (`SWEEP_NPROC > 1`):
  grid points dispatched to a worker pool of that size. Each
  worker computes its own demag kernels and parameter
  namespace, so the workers do not share mutable state.
- HPC array task (`SLURM_ARRAY_TASK_ID` set): the script runs
  only the single grid point matching the task id. Always
  serial within a single task.

`SWEEP_NPROC` is read from the environment so the script
itself does not need editing to switch modes:

    SWEEP_NPROC=6 python -m scripts.sweep_S42_J

A top-of-`main()` flag `use_full_demag` selects between full
FFT magnetostatic demag and the local-K_eff approximation.

All run-time values are top-of-`main()` variables; no argparse.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import multiprocessing as mp
import os
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import GaussianPulse
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.orchestrator.driver import run_one
from src.orchestrator.integrators import (
    step_demag_deterministic,
    step_deterministic,
)
from src.orchestrator.io import save_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _make_saf_skyrmion_ic(p):
    """Module-level IC factory so worker processes can pickle it."""
    return saf_skyrmion(
        p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw)


def _run_one_point(args):
    """Worker entry point: integrate one (cfg, point) pair.

    Parameters
    ----------
    args : tuple(dict, dict)
        (cfg, point) where `cfg` is the shared configuration
        dict built in `main()` and `point` is the grid-point-
        specific dict.

    Returns
    -------
    summary : str
        One-line console summary suitable for printing once
        the worker finishes. Workers cannot share stdout
        cleanly, so the parent prints each return value as it
        completes.
    """
    cfg, point = args
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fresh parameter namespace per worker so mutations do not
    # leak across grid points.
    p = default_params()
    p.D = cfg['D']
    p.nx = cfg['nx']
    p.ny = cfg['ny']
    p.dt = cfg['dt']
    _precompute(p)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Demag treatment + IC + integrator step.
    if cfg['use_full_demag']:
        kernels = precompute_demag_kernels(
            p,
            kind=cfg['demag_kind'],
            accuracy=(cfg['demag_newell_accuracy']
                      if cfg['demag_kind'] == 'newell' else None),
            tol_conv=(cfg['demag_newell_tol_conv']
                      if cfg['demag_kind'] == 'newell' else None),
        )
        step = step_demag_deterministic(kernels)
        # Pre-relax with full demag (convergence-stop).
        m_top0, m_bot0 = _make_saf_skyrmion_ic(p)
        (m_top_eq, m_bot_eq,
         conv, n_relax_used, E_final, tau_max) = relax(
            m_top=m_top0,
            m_bot=m_bot0,
            p=p,
            kernels=kernels,
            max_steps=cfg['relax_max_steps'],
            alpha_relax=cfg['relax_alpha'],
            tol_torque=cfg['relax_tol_torque'],
            tol_dE=cfg['relax_tol_dE'],
            check_every=cfg['relax_check_every'],
        )
        ic_factory = (
            lambda p, _eq=(m_top_eq, m_bot_eq):
            (_eq[0].copy(), _eq[1].copy()))
        n_relax_for_driver = 0
        n_relax_metadata = int(n_relax_used)
        relax_extra = {
            'relax_mode': 'convergence_stop',
            'relax_converged': bool(conv),
            'relax_tau_max_final': float(tau_max),
            'relax_E_final': float(E_final),
        }
    else:
        step = step_deterministic()
        ic_factory = _make_saf_skyrmion_ic
        n_relax_for_driver = cfg['n_relax_fixed']
        n_relax_metadata = int(cfg['n_relax_fixed'])
        relax_extra = {'relax_mode': 'fixed_time'}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gaussian drive pulse for this J0.
    pulse = GaussianPulse(
        J0=point['J0'],
        t_center=cfg['t_center'],
        FWHM=cfg['FWHM'],
    )
    trace = run_one(
        p=p,
        pulse=pulse,
        n_relax=n_relax_for_driver,
        n_drive=cfg['n_drive'],
        sample_every=cfg['sample_every'],
        step_drive=step,
        step_relax=step,
        ic_factory=ic_factory,
        record_snapshot_at=None,
        print_every=10000)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    metadata = {
        'figure': 'S42',
        'demag': ('full_fft' if cfg['use_full_demag']
                  else 'local_keff'),
        'J0': float(point['J0']),
        'FWHM': float(cfg['FWHM']),
        'sigma': float(cfg['sigma']),
        't_center': float(cfg['t_center']),
        'tail_sigmas': float(cfg['tail_sigmas']),
        'D': float(cfg['D']),
        'nx': int(cfg['nx']),
        'ny': int(cfg['ny']),
        'dt': float(cfg['dt']),
        'n_relax': int(n_relax_metadata),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
    }
    metadata.update(relax_extra)
    _D_tag = f'D{int(round(cfg["D"]*1e5)):03d}e-3'
    out_path = os.path.join(
        cfg['out_dir'],
        f'J_{point["J0"]:.2e}_{_D_tag}.npz')
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PBC-aware diagnostic.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=cfg['nx'] * p.a, L_y=cfg['ny'] * p.a,
    )
    v_avg_proxy = (
        float(cx[-1] - cx[0])
        / (2.0 * cfg['tail_sigmas'] * cfg['sigma']))
    return (
        f'  J0 = {point["J0"]:.2e}: '
        f'v_avg(proxy) = {v_avg_proxy:.0f} m/s -> {out_path}')


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration (edit here).
    # Demag treatment: True = full FFT demag + convergence-stop
    # relax; False = local K_eff approximation + fixed-time relax.
    use_full_demag = True
    # Demag formulation: 'slab' uses the analytic thin-film shape
    # factor; 'newell' uses mumax3-style finite-prism numerical
    # integration (slower ~30s precompute, captures finite-cell
    # corrections at cell-scale wavelengths).
    demag_kind = 'newell'
    demag_newell_accuracy = 4.0
    demag_newell_tol_conv = 0.02
    # Sweep grid (A/m^2). Paper S42 uses ~9 points up to 8.9e11.
    J_values = [
        1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e11,
        6.0e11, 7.0e11, 8.0e11, 8.9e11,
    ]
    # Gaussian pulse parameters.
    FWHM = 500.0e-12          # 500 ps full width at half max.
    tail_sigmas = 3.0
    # Set A material constants.
    D = 0.85e-3               # J/m^2
    nx = 256
    ny = 256
    dt = 5.0e-14              # s
    # Fixed-time relax (used only when use_full_demag is False).
    relax_time = 500.0e-12
    n_relax_fixed = int(math.ceil(relax_time / dt))
    # Convergence-stop relax tolerances (used only when full demag).
    relax_max_steps = 200_000
    relax_alpha = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    # Drive window = 6 sigma + 200 ps tail.
    sigma = FWHM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    t_center = tail_sigmas * sigma
    drive_time = 2.0 * tail_sigmas * sigma + 200.0e-12
    n_drive = int(math.ceil(drive_time / dt))
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    out_dir = 'output/sweeps_S41_S49/S42'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Execution model: serial (default), Pool (SWEEP_NPROC > 1),
    # or HPC array task (SLURM_ARRAY_TASK_ID).
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(J_values):
            raise RuntimeError(
                f'sweep_S42_J: SLURM_ARRAY_TASK_ID={task_id} '
                f'out of range [0, {len(J_values) - 1}].')
        J_values = [J_values[idx]]
        # An HPC array task runs only its assigned point, so a
        # local pool would be redundant.
        n_proc = 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg = {
        'use_full_demag': bool(use_full_demag),
        'demag_kind': str(demag_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'D': float(D),
        'nx': int(nx),
        'ny': int(ny),
        'dt': float(dt),
        'FWHM': float(FWHM),
        'sigma': float(sigma),
        't_center': float(t_center),
        'tail_sigmas': float(tail_sigmas),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'n_relax_fixed': int(n_relax_fixed),
        'relax_max_steps': int(relax_max_steps),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
        'relax_check_every': int(relax_check_every),
        'out_dir': out_dir,
    }
    args_list = [(cfg, {'J0': float(J0)}) for J0 in J_values]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    os.makedirs(out_dir, exist_ok=True)
    demag_tag = 'full_fft' if use_full_demag else 'local_keff'
    print(
        f'S42 sweep: nx={nx}, FWHM={FWHM*1e12:.0f} ps, '
        f'sigma={sigma*1e12:.1f} ps, t_center={t_center*1e12:.1f} ps')
    print(f'  demag = {demag_tag}, n_drive = {n_drive} steps, '
          f'n_proc = {n_proc}')
    if n_proc <= 1:
        for args in args_list:
            print(_run_one_point(args))
    else:
        # `spawn` is the default on macOS for Python 3.8+; it
        # requires the worker function to be importable at the
        # module level (which `_run_one_point` is).
        with mp.Pool(n_proc) as pool:
            for result in pool.imap_unordered(
                    _run_one_point, args_list):
                print(result)


# =============================================================================
if __name__ == '__main__':
    main()
