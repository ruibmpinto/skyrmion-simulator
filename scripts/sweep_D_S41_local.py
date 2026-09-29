"""Local multi-worker D-sweep for Pham et al. (2024) Figure S41.

Mirror of `sweep_D_S41.py` but designed for laptop / workstation
runs: loops over the whole D grid with optional
`multiprocessing.Pool` parallelism set via the `SWEEP_NPROC`
environment variable. No SLURM dispatch.

Output filenames carry the field_kind tag so K_eff and demag
sweeps coexist in the same directory.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import multiprocessing as mp
import os
# Third-party
import numpy as np
# Local
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.fields import effective_field
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rhs_local_keff, rk4_step
from src.simulator.parameters import _precompute, default_params
from src.simulator.pulses import ConstantPulse, SquarePulse
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
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _run_one_D(args):
    """Relax + drive at one D value. `args` is the (cfg, D)
    tuple; cfg holds every numerical parameter so the function
    is picklable for `multiprocessing.Pool`."""
    cfg, D = args
    field_kind = cfg['field_kind']
    nx = cfg['nx']
    ny = cfg['ny']
    dt = cfg['dt']
    # Build the parameter namespace at this D.
    p = default_params()
    p.D = D
    p.nx = nx
    p.ny = ny
    p.dt = dt
    _precompute(p)
    print(f'D = {D*1e3:.3f} mJ/m^2: starting', flush=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the step callable + relaxed IC for either path.
    if field_kind == 'keff':
        kernels = None
        step = step_deterministic()
        m_top0, m_bot0 = saf_skyrmion(
            nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        print(f'  D={D*1e3:.3f}: relaxing with K_eff '
              f'(over-damped, no kernel)...', flush=True)
        _alpha_save = p.alpha
        _gamma_p_save = p.gamma_p
        _H_DL_save = p.H_DL
        _H_FL_save = p.H_FL
        _pulse_save = getattr(p, 'pulse', None)
        p.alpha = float(cfg['relax_alpha'])
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
        p.H_DL = 0.0
        p.H_FL = 0.0
        p.pulse = ConstantPulse(0.0)
        m_top_eq = m_top0.copy()
        m_bot_eq = m_bot0.copy()
        tau_max = float('inf')
        conv = False
        n_relax_used = cfg['relax_max_steps']
        try:
            t = 0.0
            for k in range(1, cfg['relax_max_steps'] + 1):
                m_top_eq, m_bot_eq = rk4_step(
                    rhs_local_keff,
                    m_top_eq, m_bot_eq, t, p.dt, p)
                t += p.dt
                if k % cfg['relax_check_every'] == 0:
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
                    if tau_max < cfg['relax_tol_torque']:
                        conv = True
                        n_relax_used = k
                        break
        finally:
            p.alpha = _alpha_save
            p.gamma_p = _gamma_p_save
            p.H_DL = _H_DL_save
            p.H_FL = _H_FL_save
            p.pulse = _pulse_save
        E_final = float('nan')
    else:
        kernels = precompute_demag_kernels(
            p, kind=field_kind,
            accuracy=cfg['demag_newell_accuracy'],
            tol_conv=cfg['demag_newell_tol_conv'])
        step = step_demag_deterministic(kernels)
        m_top0, m_bot0 = saf_skyrmion(
            nx, ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
        print(f'  D={D*1e3:.3f}: relaxing with {field_kind} '
              f'demag (convergence-stop)...', flush=True)
        (m_top_eq, m_bot_eq,
         conv, n_relax_used, E_final, tau_max) = relax(
            m_top=m_top0, m_bot=m_bot0, p=p, kernels=kernels,
            max_steps=cfg['relax_max_steps'],
            alpha_relax=cfg['relax_alpha'],
            tol_torque=cfg['relax_tol_torque'],
            tol_dE=cfg['relax_tol_dE'],
            check_every=cfg['relax_check_every'],
            print_every=0)
    print(
        f'  D={D*1e3:.3f}: relax converged={conv}, '
        f'n_steps={n_relax_used} '
        f'({n_relax_used*dt*1e12:.0f} ps), '
        f'tau_max={tau_max:.2e} T',
        flush=True)

    # Hand the relaxed pair to the driver as a fresh-copy IC.
    def ic_factory(p, _eq=(m_top_eq, m_bot_eq)):
        return _eq[0].copy(), _eq[1].copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive.
    pulse = SquarePulse(
        J0=cfg['J0'], t_start=0.0, t_end=cfg['t_pulse'])
    out_path = os.path.join(
        cfg['out_dir'],
        f'D_{D*1e3:.3f}mJm2_{field_kind}.npz')
    trace = run_one(
        p=p, pulse=pulse,
        n_relax=0, n_drive=cfg['n_drive'],
        sample_every=cfg['sample_every'],
        step_drive=step, step_relax=step,
        ic_factory=ic_factory,
        record_snapshot_at=None,
        print_every=0,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist with full metadata.
    metadata = {
        'figure': 'S41_D_sweep',
        'demag': ('local_keff' if field_kind == 'keff'
                  else field_kind),
        'field_kind': str(field_kind),
        'demag_newell_accuracy': float(
            cfg['demag_newell_accuracy']),
        'demag_newell_tol_conv': float(
            cfg['demag_newell_tol_conv']),
        'J0': float(cfg['J0']),
        't_pulse': float(cfg['t_pulse']),
        'D': float(D),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'n_relax': int(n_relax_used),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
        'relax_mode': 'convergence_stop',
        'relax_converged': bool(conv),
        'relax_tau_max_final': float(tau_max),
        'relax_E_final': float(E_final),
        'relax_alpha': float(cfg['relax_alpha']),
        'relax_tol_torque': float(cfg['relax_tol_torque']),
        'relax_tol_dE': float(cfg['relax_tol_dE']),
    }
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cheap end-of-run diagnostic.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * p.a, L_y=ny * p.a, periodic_y=True)
    t_arr = trace['t']
    i_end = int(((len(t_arr) - 1) * cfg['t_pulse'])
                // (cfg['n_drive'] * dt))
    if i_end >= len(cx):
        i_end = len(cx) - 1
    dx_pulse = float(cx[i_end] - cx[0])
    v_avg = dx_pulse / cfg['t_pulse']
    d0 = float(trace['d_top'][0])
    d1 = float(trace['d_top'][-1])
    Q0 = float(trace['Q_top'][0])
    return (
        f'  D = {D*1e3:.3f} mJ/m^2: '
        f'd_top[0]={d0*1e9:6.1f} nm, '
        f'd_top[end]={d1*1e9:6.1f} nm, '
        f'Q={Q0:+.4f}, '
        f'v_avg={v_avg:.1f} m/s -> {out_path}')


def main():
    """Sweep the DMI constant `D` over the full grid locally.

    Loops all of `D_values` (optional `multiprocessing.Pool` via
    `SWEEP_NPROC`), writing one S41 trajectory per `D` to
    `output/sweeps_S41_S49/S41_D_sweep`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sweep grid: DMI values to scan. Same as sweep_D_S41.py.
    D_values = [
        0.62e-3,
        0.72e-3,
        0.80e-3,
        0.85e-3,
        0.90e-3,
        0.95e-3,
        1.00e-3,
    ]
    # Field model. 'keff' (no FFT demag) or 'newell' / 'slab'.
    field_kind = 'keff'
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
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cfg = {
        'field_kind': str(field_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'J0': float(J0),
        't_pulse': float(t_pulse),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'n_drive': int(n_drive),
        'sample_every': int(sample_every),
        'relax_max_steps': int(relax_max_steps),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
        'relax_check_every': int(relax_check_every),
        'out_dir': out_dir,
    }
    args_list = [(cfg, D) for D in D_values]
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    print(f'sweep_D_S41_local: {len(args_list)} D values, '
          f'field_kind={field_kind}, n_proc={n_proc}',
          flush=True)
    if n_proc <= 1:
        for args in args_list:
            print(_run_one_D(args), flush=True)
    else:
        with mp.Pool(n_proc) as pool:
            for result in pool.imap_unordered(
                    _run_one_D, args_list):
                print(result, flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
