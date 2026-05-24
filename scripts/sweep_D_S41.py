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
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import SquarePulse
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.sweeps.driver import run_one
from src.sweeps.integrators import step_demag_deterministic
from src.sweeps.io import save_trace

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
    # Demag (Newell finite-prism, same as production S41).
    demag_kind = 'newell'
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
    # Build the Newell kernel and do the convergence-stop
    # relax from the analytic IC, same as the production
    # sweep_S41_v_time.py.
    kernels = precompute_demag_kernels(
        p, kind=demag_kind,
        accuracy=demag_newell_accuracy,
        tol_conv=demag_newell_tol_conv)
    step = step_demag_deterministic(kernels)
    m_top0, m_bot0 = saf_skyrmion(
        nx, ny, p.a, p.skyrmion_R, p.skyrmion_dw)
    print(f'  relaxing with Newell demag '
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

    def ic_factory(p, _eq=(m_top_eq, m_bot_eq)):
        return _eq[0].copy(), _eq[1].copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive.
    pulse = SquarePulse(J0=J0, t_start=0.0, t_end=t_pulse)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(
        out_dir, f'D_{D*1e3:.3f}mJm2.npz')
    print(f'  drive: J0={J0:.2e}, t_pulse={t_pulse*1e9:.1f} ns, '
          f'n_drive={n_drive}', flush=True)
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
    metadata = {
        'figure': 'S41_D_sweep',
        'demag': 'full_fft',
        'demag_kind': str(demag_kind),
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
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cheap end-of-run diagnostic with PBC-aware unwrap.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * p.a, L_y=ny * p.a)
    t = trace['t']
    i_start = 0
    i_end = int(((len(t) - 1) * t_pulse) // (n_drive * dt))
    if i_end >= len(cx):
        i_end = len(cx) - 1
    dx_pulse = float(cx[i_end] - cx[i_start])
    v_avg = dx_pulse / t_pulse
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
