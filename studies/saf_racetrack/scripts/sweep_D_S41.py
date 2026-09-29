"""S41 trajectory swept over the DMI constant D, to locate the
Newell-effective Bogdanov--Hubert threshold and the D that
reproduces the paper's ~215 nm skyrmion diameter.

Each array index picks one D value from `D_values`, runs the
same S41 protocol (256 x 256 Newell demag, convergence-stop
over-damped relax, 2 ns square pulse at J=1e11 A/m^2), and
writes the trajectory to
`output/sweeps_S41_S49/S41_D_sweep/D_<X>mJm2.npz`.

Dispatched as a SLURM array job; the dispatch is mandatory
because each D point is multi-hour, so running the whole list
serially in a single task is impractical.

All other run-time values are top-of-`main()` variables, no
argparse.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.relaxation import relax
from skyrmion_simulator.simulator.demag import precompute_demag_kernels
from skyrmion_simulator.simulator.fields import effective_field
from skyrmion_simulator.simulator.initial_conditions import saf_skyrmion
from skyrmion_simulator.simulator.integrator import rhs_local_keff, rk4_step
from skyrmion_simulator.simulator.parameters import _precompute, default_params
from skyrmion_simulator.simulator.pulses import ConstantPulse, SquarePulse
from skyrmion_simulator.stochastic_llgs.diagnostics import unwrap_trajectory
from studies.saf_racetrack.orchestrator.driver import run_one
from studies.saf_racetrack.orchestrator.integrators import (
    step_demag_deterministic,
    step_deterministic,
)
from studies.saf_racetrack.orchestrator.io import save_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def main():
    """Run the S41 pulse protocol at one DMI value `D`.

    The SLURM array index selects one `D` from `D_values`; the
    trajectory is written to `output/sweeps_S41_S49/S41_D_sweep`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sweep grid: DMI values to scan. Pham 2024 nominal is
    # 0.62 mJ/m^2 (in-plane PMA, sputter sample); we use
    # 0.85 mJ/m^2 to match the experimental ~215 nm size
    # with our finite-prism (Newell) kernel, which raises
    # the effective D_c above the slab estimate.
    D_values = [
        0.62e-3,
        0.72e-3,
        0.80e-3,
        0.85e-3,
        0.90e-3,
        0.95e-3,
        1.00e-3,
    ]
    # Field model. 'keff' uses the local K_eff path (slab demag
    # folded into anisotropy via p.C_anis_*, no explicit FFT
    # convolution). 'newell' / 'slab' use the explicit-demag
    # path with bare K plus a precomputed FFT kernel.
    field_kind = 'keff'
    # demag_kind = 'newell'
    demag_newell_accuracy = 4.0
    demag_newell_tol_conv = 0.02
    # Drive parameters (paper S41).
    J0 = 1.0e11
    t_pulse = 2.0e-9
    # Lattice and time integration.
    nx = 256
    ny = 256
    dt = 5.0e-14
    drive_time = t_pulse + 500.0e-12
    n_drive = int(math.ceil(drive_time / dt))
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    # Convergence-stop relax.
    relax_max_steps = 200_000
    relax_alpha = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    out_dir = 'output/sweeps_S41_S49/S41_D_sweep'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # HPC array index dispatch: each task picks one D value.
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is None:
        raise RuntimeError(
            'sweep_D_S41: SLURM_ARRAY_TASK_ID must be set '
            '(this script is meant for SLURM array dispatch).')
    idx = int(task_id)
    if idx < 0 or idx >= len(D_values):
        raise RuntimeError(
            f'sweep_D_S41: SLURM_ARRAY_TASK_ID={task_id} out '
            f'of range [0, {len(D_values) - 1}].')
    D = D_values[idx]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the parameter namespace at this D.
    p = default_params()
    p.D = D
    p.nx = nx
    p.ny = ny
    p.dt = dt
    _precompute(p)
    print(f'D-sweep task {idx}/{len(D_values)-1}: '
          f'D = {D*1e3:.3f} mJ/m^2', flush=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the step callable + relaxed IC for either path.
    if field_kind == 'keff':
        # Local-K_eff path: no FFT demag kernel; anisotropy
        # prefactor p.C_anis_* already contains the slab demag.
        kernels = None
        step = step_deterministic()
        m_top0, m_bot0 = saf_skyrmion(
            nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        print(f'  relaxing with K_eff (over-damped quench, '
              f'no demag kernel) at D={D*1e3:.3f} mJ/m^2...',
              flush=True)
        # Inline relax: save & override damping/SOT, integrate
        # with alpha=relax_alpha, restore on exit. Mirrors the
        # convention in src/skyrmion_simulator/simulator/relaxation.relax() but
        # uses the K_eff RHS instead of the demag-aware pair.
        _alpha_save = p.alpha
        _gamma_p_save = p.gamma_p
        _H_DL_save = p.H_DL
        _H_FL_save = p.H_FL
        _pulse_save = getattr(p, 'pulse', None)
        p.alpha = float(relax_alpha)
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        p.H_DL = 0.0
        p.H_FL = 0.0
        p.pulse = ConstantPulse(0.0)
        m_top_eq = m_top0.copy()
        m_bot_eq = m_bot0.copy()
        tau_max = float('inf')
        conv = False
        n_relax_used = relax_max_steps
        try:
            t = 0.0
            for k in range(1, relax_max_steps + 1):
                m_top_eq, m_bot_eq = rk4_step(
                    rhs_local_keff,
                    m_top_eq, m_bot_eq, t, p.dt, p)
                t += p.dt
                if k % relax_check_every == 0:
                    # Maximum tangential torque |m x (m x H)|.
                    H_t = effective_field(
                        m_top_eq, m_bot_eq,
                        p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY)
                    H_b = effective_field(
                        m_bot_eq, m_top_eq,
                        p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY)
                    tau_t = np.cross(
                        m_top_eq, np.cross(m_top_eq, H_t))
                    tau_b = np.cross(
                        m_bot_eq, np.cross(m_bot_eq, H_b))
                    tau_max = float(max(
                        np.sqrt((tau_t * tau_t).sum(-1)).max(),
                        np.sqrt((tau_b * tau_b).sum(-1)).max()))
                    if k % 10000 == 0:
                        print(f'    relax step {k}/'
                              f'{relax_max_steps} '
                              f'({k*p.dt*1e12:.0f} ps), '
                              f'tau_max={tau_max:.2e} T',
                              flush=True)
                    if tau_max < relax_tol_torque:
                        conv = True
                        n_relax_used = k
                        break
        finally:
            p.alpha = _alpha_save
            p.gamma_p = _gamma_p_save
            p.H_DL = _H_DL_save
            p.H_FL = _H_FL_save
            p.pulse = _pulse_save
        # K_eff path skips the energy-trend check, so E_final is
        # not produced; emit nan so the metadata schema stays
        # uniform with the demag-path runs.
        E_final = float('nan')
    else:
        # Explicit-demag path. Precompute kernel (slab or Newell)
        # and use the demag-aware relax().
        kernels = precompute_demag_kernels(
            p, kind=field_kind,
            accuracy=demag_newell_accuracy,
            tol_conv=demag_newell_tol_conv)
        step = step_demag_deterministic(kernels)
        m_top0, m_bot0 = saf_skyrmion(
            nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        print(f'  relaxing with {field_kind} demag '
              f'(convergence-stop) at D={D*1e3:.3f} mJ/m^2...',
              flush=True)
        (m_top_eq, m_bot_eq,
         conv, n_relax_used, E_final, tau_max) = relax(
            m_top=m_top0, m_bot=m_bot0, p=p, kernels=kernels,
            max_steps=relax_max_steps,
            alpha_relax=relax_alpha,
            tol_torque=relax_tol_torque,
            tol_dE=relax_tol_dE,
            check_every=relax_check_every,
            print_every=10000,)
    print(
        f'  relax: converged={conv}, '
        f'n_steps={n_relax_used} '
        f'({n_relax_used*dt*1e12:.0f} ps), '
        f'tau_max={tau_max:.2e} T, E={E_final:.3e} J',
        flush=True)
    if not conv:
        print(f'  WARN: relax did not satisfy '
              f'tol_torque={relax_tol_torque:.1e} T within '
              f'max_steps={relax_max_steps}.', flush=True)

    # Hand the relaxed pair to the driver as a fresh-copy IC.
    def ic_factory(p, _eq=(m_top_eq, m_bot_eq)):
        return _eq[0].copy(), _eq[1].copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive.
    # Square current pulse on [0, t_pulse].
    pulse = SquarePulse(J0=J0, t_start=0.0, t_end=t_pulse)
    # Ensure output directory exists.
    os.makedirs(out_dir, exist_ok=True)
    # Per-D output filename keeps D and field_kind in the basename
    # so K_eff and Newell sweeps coexist in the same directory.
    out_path = os.path.join(
        out_dir, f'D_{D*1e3:.3f}mJm2_{field_kind}.npz')

    print(f'  drive: J0={J0:.2e}, t_pulse={t_pulse*1e9:.1f} ns, '
          f'n_drive={n_drive}', flush=True)
    # Skip relax (n_relax=0): IC is already the equilibrium.
    trace = run_one(
        p=p, pulse=pulse,
        n_relax=0, n_drive=n_drive,
        sample_every=sample_every,
        step_drive=step, step_relax=step,
        ic_factory=ic_factory,
        record_snapshot_at=None,
        print_every=10000,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist with full metadata, mirroring sweep_S41_v_time.py.
    # Full run metadata: physics + numerics + relax outcome.
    metadata = {
        'figure': 'S41_D_sweep',
        'demag': 'local_keff' if field_kind == 'keff' else field_kind,
        'field_kind': str(field_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'J0': float(J0),
        't_pulse': float(t_pulse),
        'D': float(D),
        'nx': int(nx),
        'ny': int(ny),
        'dt': float(dt),
        'n_relax': int(n_relax_used),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'relax_mode': 'convergence_stop',
        'relax_converged': bool(conv),
        'relax_tau_max_final': float(tau_max),
        'relax_E_final': float(E_final),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
    }
    # Write npz with trace arrays + metadata dict.
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cheap end-of-run diagnostic with PBC-aware unwrap.
    # Unwrap PBC jumps so centroid is monotone in real space.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * p.a, L_y=ny * p.a, periodic_y=True)
    t = trace['t']
    # Pulse window indices: start at IC, end at the t_pulse mark.
    i_start = 0
    i_end = int(((len(t) - 1) * t_pulse) // (n_drive * dt))
    if i_end >= len(cx):
        i_end = len(cx) - 1
    # Displacement during the pulse and resulting average speed.
    dx_pulse = float(cx[i_end] - cx[i_start])
    v_avg = dx_pulse / t_pulse
    # Pre- and post-pulse skyrmion size and topological charge.
    d_top_initial = float(trace['d_top'][0])
    d_top_final = float(trace['d_top'][-1])
    Q_initial = float(trace['Q_top'][0])
    Q_final = float(trace['Q_top'][-1])
    print(f'  wrote {out_path}', flush=True)
    print(f'  D = {D*1e3:.3f} mJ/m^2: '
          f'd_top[0]={d_top_initial*1e9:6.1f} nm, '
          f'd_top[end]={d_top_final*1e9:6.1f} nm, '
          f'Q={Q_initial:+.4f}, '
          f'v_avg={v_avg:.1f} m/s', flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
