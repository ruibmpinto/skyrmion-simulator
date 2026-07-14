"""Sweep driver for Pham et al. (2024) Figure S41.

Single DC trajectory: J = 1e11 A/m^2 square pulse for 2 ns,
sampled at fine resolution so that v_inst(t) can be plotted
alongside v_avg = Delta x / t_pulse. The trace is written to
`output/sweeps_S41_S49/S41/run.npz`.

A top-of-`main()` flag `use_full_demag` selects the demag
treatment:
- `True`  : full FFT magnetostatic demag via
  `step_demag_deterministic(kernels)`. The initial condition
  is pre-relaxed by the convergence-stop
  `phase_diagram.relaxation.relax` so the drive starts from
  the true equilibrium.
- `False` : local `K_eff = K - mu0 Ms^2/2` approximation via
  `step_deterministic()`. Relaxation is the fixed-time
  Phase 1 inside `run_one`.

All run-time values are top-of-`main()` variables; no argparse.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import SquarePulse
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


def main():
    """Run one S41 DC trajectory (2 ns square pulse at J=1e11).

    Samples finely so v_inst(t) can be recovered; the trace is
    written to `output/sweeps_S41_S49/S41`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration.
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
    # Drive parameters (paper S41).
    J0 = 1.0e11               # Amperes/m^2
    t_pulse = 2.0e-9          # Square-pulse duration (s)
    # DMI: paper-adjusted (Set A).
    D = 0.85e-3               # J/m^2
    # Lattice size: 256 x 256 to match the paper.
    nx = 256
    ny = 256
    # Time integration.
    dt = 5.0e-14              # s; matches default_params.
    # Phase 2: drive for the pulse + a 500 ps post-pulse tail so
    # the instantaneous velocity decays back to zero (visible in
    # the figure).
    drive_time = t_pulse + 500.0e-12
    n_drive = int(math.ceil(drive_time / dt))
    # Sample every 5 ps for a smooth v_inst(t) curve.
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    # Fixed-time relax (used only when use_full_demag is False).
    relax_time = 500.0e-12
    n_relax_fixed = int(math.ceil(relax_time / dt))
    # Convergence-stop relax (used only when use_full_demag is True).
    relax_max_steps = 200_000        # safety cap (= 10 ns at dt)
    relax_alpha = 1.0                # over-damped quench
    relax_tol_torque = 1.0e-5        # T
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    # Output path. Encode the demag kind and DMI value in the
    # filename so re-running with a different D (or switching
    # demag formulations) does not clobber a previous run.
    out_dir = 'output/sweeps_S41_S49/S41'
    out_path = os.path.join(
        out_dir, f'run_{demag_kind}_D{int(round(D*1e5)):03d}e-3.npz')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # HPC array index: S41 has a single trajectory; reject any
    # non-zero array index loudly.
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None and int(task_id) != 0:
        raise RuntimeError(
            f'sweep_S41_v_time: SLURM_ARRAY_TASK_ID={task_id} '
            f'out of range; S41 has one trajectory (index 0).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the Set A parameter namespace.
    p = default_params()
    p.D = D
    p.nx = nx
    p.ny = ny
    p.dt = dt
    _precompute(p)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the integrator step and, when using full demag, the
    # pre-relaxed IC. The two branches diverge here and merge
    # back at the `run_one` call below.
    if use_full_demag:
        # Precompute kernels once; depends only on (nx, ny, a,
        # t_Co, d_Ru, Ms, mu0).
        kernels = precompute_demag_kernels(
            p,
            kind=demag_kind,
            accuracy=(demag_newell_accuracy
                      if demag_kind == 'newell' else None),
            tol_conv=(demag_newell_tol_conv
                      if demag_kind == 'newell' else None),
        )
        # Demag-aware step used for both relax and drive.
        step = step_demag_deterministic(kernels)
        # Convergence-stop relax (over-damped quench) with full
        # demag. relax() temporarily mutates p (SOT, alpha,
        # gamma_p) and restores everything on exit.
        m_top0, m_bot0 = saf_skyrmion(
            nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        print(f'S41: relaxing with full demag (convergence-stop)...')
        (m_top_eq, m_bot_eq,
         conv, n_relax_used, E_final, tau_max) = relax(
            m_top=m_top0,
            m_bot=m_bot0,
            p=p,
            kernels=kernels,
            max_steps=relax_max_steps,
            alpha_relax=relax_alpha,
            tol_torque=relax_tol_torque,
            tol_dE=relax_tol_dE,
            check_every=relax_check_every,
            print_every=10000,
        )
        print(
            f'  relax: converged={conv}, '
            f'n_steps={n_relax_used} '
            f'({n_relax_used*dt*1e12:.0f} ps), '
            f'tau_max={tau_max:.2e} T, E={E_final:.3e} J')
        if not conv:
            # Surface non-convergence loudly; run still proceeds.
            print(
                f'  WARN: relax did not satisfy '
                f'tol_torque={relax_tol_torque:.1e} T within '
                f'max_steps={relax_max_steps}.')

        def ic_factory(p):
            # Fresh copy per call so the driver does not mutate
            # the stored equilibrium spin field.
            return m_top_eq.copy(), m_bot_eq.copy()
        # The pre-relaxation already settled the IC; run_one
        # skips its own Phase 1.
        n_relax_for_driver = 0
        # Total number of relax steps actually executed (here:
        # the convergence-stop count). Stored as `n_relax` to
        # match the metadata schema of the other sweep scripts.
        n_relax_metadata = int(n_relax_used)
        relax_meta = {
            'relax_mode': 'convergence_stop',
            'relax_converged': bool(conv),
            'relax_tau_max_final': float(tau_max),
            'relax_E_final': float(E_final),
            'relax_alpha': float(relax_alpha),
            'relax_tol_torque': float(relax_tol_torque),
            'relax_tol_dE': float(relax_tol_dE),
        }
    else:
        # Local-K_eff path (no demag). Fixed-time relax inside run_one. 
        # No kernels.
        step = step_deterministic()

        def ic_factory(p):
            return saf_skyrmion(
                p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        n_relax_for_driver = n_relax_fixed
        n_relax_metadata = int(n_relax_fixed)
        relax_meta = {
            'relax_mode': 'fixed_time',
            'relax_time_ps': float(relax_time * 1e12),}
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive pulse: square pulse starting at drive-phase t = 0.
    pulse = SquarePulse(J0=J0, t_start=0.0, t_end=t_pulse)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run drive.
    os.makedirs(out_dir, exist_ok=True)
    print(
        f'S41: drive with '
        f'{"full_fft" if use_full_demag else "local_keff"} demag, '
        f'J0={J0:.2e}, t_pulse={t_pulse*1e9:.1f} ns')
    print(f'  n_drive = {n_drive} steps '
          f'({n_drive*dt*1e12:.1f} ps)')
    print(f'  sample_every = {sample_every} steps '
          f'({sample_every*dt*1e12:.2f} ps)')
    trace = run_one(
        p=p,
        pulse=pulse,
        n_relax=n_relax_for_driver,
        n_drive=n_drive,
        sample_every=sample_every,
        step_drive=step,
        step_relax=step,
        ic_factory=ic_factory,
        record_snapshot_at=None,
        print_every=10000,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist with full metadata.
    metadata = {
        'figure': 'S41',
        'demag': 'full_fft' if use_full_demag else 'local_keff',
        'J0': float(J0),
        't_pulse': float(t_pulse),
        'D': float(D),
        'nx': int(nx),
        'ny': int(ny),
        'dt': float(dt),
        'n_relax': int(n_relax_metadata),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
    }
    metadata.update(relax_meta)
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cheap end-of-run diagnostic with PBC-aware unwrap.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * p.a, L_y=ny * p.a, periodic_y=True)
    # v_avg over the pulse window using the unwrapped centroid.
    t = trace['t']
    i_start = int(0)
    i_end = int(((len(t) - 1) * t_pulse) // (n_drive * dt))
    if i_end >= len(cx):
        i_end = len(cx) - 1
    dx_pulse = float(cx[i_end] - cx[i_start])
    v_avg = dx_pulse / t_pulse
    d_top_initial = float(trace['d_top'][0])
    d_top_final = float(trace['d_top'][-1])
    Q_initial = float(trace['Q_top'][0])
    Q_final = float(trace['Q_top'][-1])
    print(f'  wrote {out_path}')
    print(f'  trace has {len(trace["t"])} samples')
    print(f'  d_top[0]   = {d_top_initial*1e9:6.1f} nm '
          f'(paper expects ~215 nm)')
    print(f'  d_top[end] = {d_top_final*1e9:6.1f} nm')
    print(f'  Q_top[0]   = {Q_initial:+.4f}')
    print(f'  Q_top[end] = {Q_final:+.4f}')
    print(f'  v_avg over pulse window = {v_avg:.1f} m/s '
          f'(paper expects ~117 m/s)')


# =============================================================================
if __name__ == '__main__':
    main()
