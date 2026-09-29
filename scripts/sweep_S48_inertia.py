"""Sweep driver for Pham et al. (2024) Figure S48.

DC square pulse at J = 1e11 A/m^2 for 2 ns, sweeping H_RKKY
with Set B material constants (alpha = 0.216, gamma = 175.9e9,
D = 0.62 mJ/m^2). Same execution-model machinery
(serial / multiprocessing Pool / HPC array).
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
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _make_saf_skyrmion_ic(p):
    """Module-level IC factory so worker processes can pickle it."""
    return saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)


def _run_one_point(args):
    """Worker: integrate one H_RKKY grid point (Set B); returns a
    one-line console summary."""
    cfg, point = args
    # Fresh parameter namespace; Set B alpha/gamma/D, H_RKKY varies.
    p = default_params()
    p.D = cfg['D']
    p.alpha = cfg['alpha']
    p.gamma = cfg['gamma']
    # Set B keeps the measured Hk_top = 12.4 mT (S48 caption).
    p.K_top = 1.294e6
    p.H_RKKY = float(point['H_RKKY'])
    p.nx = cfg['nx']
    p.ny = cfg['ny']
    p.dt = cfg['dt']
    _precompute(p)
    # Demag / relax branch: full FFT demag + convergence-stop, or
    # local K_eff + fixed-time relax.
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
        m_top0, m_bot0 = _make_saf_skyrmion_ic(p)
        (m_top_eq, m_bot_eq,
         conv, n_relax_used, E_final, tau_max) = relax(
            m_top=m_top0, m_bot=m_bot0, p=p, kernels=kernels,
            max_steps=cfg['relax_max_steps'],
            alpha_relax=cfg['relax_alpha'],
            tol_torque=cfg['relax_tol_torque'],
            tol_dE=cfg['relax_tol_dE'],
            check_every=cfg['relax_check_every'],
            print_every=10000)
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
    # DC square drive pulse on [0, t_pulse].
    pulse = SquarePulse(
        J0=cfg['J0'], t_start=0.0, t_end=cfg['t_pulse'])
    trace = run_one(
        p=p, pulse=pulse,
        n_relax=n_relax_for_driver, n_drive=cfg['n_drive'],
        sample_every=cfg['sample_every'],
        step_drive=step, step_relax=step,
        ic_factory=ic_factory, record_snapshot_at=None,
        print_every=10000)
    # Persist trace + full run metadata; tag filename by H_RKKY, D.
    metadata = {
        'figure': 'S48',
        'demag': ('full_fft' if cfg['use_full_demag']
                  else 'local_keff'),
        'J0': float(cfg['J0']),
        't_pulse': float(cfg['t_pulse']),
        'H_RKKY': float(point['H_RKKY']),
        'alpha': float(cfg['alpha']),
        'gamma': float(cfg['gamma']),
        'D': float(cfg['D']),
        'nx': int(cfg['nx']), 'ny': int(cfg['ny']),
        'dt': float(cfg['dt']),
        'n_relax': int(n_relax_metadata),
        'n_drive': int(cfg['n_drive']),
        'sample_every': int(cfg['sample_every']),
    }
    metadata.update(relax_extra)
    _D_tag = f'D{int(round(cfg["D"]*1e5)):03d}e-3'
    out_name = (
        f'HRKKY_{point["H_RKKY"]*1000:.0f}mT_{_D_tag}.npz')
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # Steady-state v proxy in the pulse middle (0.5-1.5 ns).
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=cfg['nx'] * p.a, L_y=cfg['ny'] * p.a, periodic_y=True)
    import numpy as np
    t_arr = trace['t']
    win = (t_arr > 0.5e-9) & (t_arr < 1.5e-9)
    if np.any(win):
        v_steady = (
            float(cx[win][-1] - cx[win][0])
            / float(t_arr[win][-1] - t_arr[win][0]))
    else:
        v_steady = float('nan')
    return (
        f'  H_RKKY = {point["H_RKKY"]*1000:>4.0f} mT: '
        f'v_steady proxy = {v_steady:>5.0f} m/s -> {out_name}')


def main():
    """Sweep the RKKY field `H_RKKY` under a 2 ns DC pulse at
    J = 1e11 (Set B constants), writing one trace per point to
    `output/sweeps_S41_S49/S48`.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    use_full_demag = True
    # Demag formulation: 'slab' uses the analytic thin-film shape
    # factor; 'newell' uses mumax3-style finite-prism numerical
    # integration (slower ~30s precompute, captures finite-cell
    # corrections at cell-scale wavelengths).
    demag_kind = 'newell'
    demag_newell_accuracy = 4.0
    demag_newell_tol_conv = 0.02
    H_RKKY_values = [
        0.100, 0.150, 0.200, 0.250, 0.300,
        0.350, 0.400, 0.450, 0.500,
    ]
    J0 = 1.0e11
    t_pulse = 2.0e-9
    # Set B (paper Figure S48 caption).
    alpha = 0.216
    gamma = 175.9e9
    D = 0.62e-3
    nx = 256
    ny = 256
    dt = 5.0e-14
    relax_time = 500.0e-12
    n_relax_fixed = int(math.ceil(relax_time / dt))
    relax_max_steps = 200_000
    relax_alpha = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    drive_time = t_pulse + 1.0e-9
    n_drive = int(math.ceil(drive_time / dt))
    sample_dt = 1.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    out_dir = 'output/sweeps_S41_S49/S48'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(H_RKKY_values):
            raise RuntimeError(
                f'sweep_S48_inertia: '
                f'SLURM_ARRAY_TASK_ID={task_id} out of range '
                f'[0, {len(H_RKKY_values) - 1}].')
        H_RKKY_values = [H_RKKY_values[idx]]
        n_proc = 1
    cfg = {
        'use_full_demag': bool(use_full_demag),
        'demag_kind': str(demag_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'J0': float(J0),
        't_pulse': float(t_pulse),
        'alpha': float(alpha),
        'gamma': float(gamma),
        'D': float(D),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
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
    args_list = [
        (cfg, {'H_RKKY': H}) for H in H_RKKY_values]
    os.makedirs(out_dir, exist_ok=True)
    demag_tag = 'full_fft' if use_full_demag else 'local_keff'
    print(
        f'S48 sweep: nx={nx}, demag={demag_tag}, '
        f'Set B (alpha={alpha}, gamma={gamma:.2e}, D={D*1e3:.2f}), '
        f'{len(args_list)} grid points, n_proc={n_proc}')
    if n_proc <= 1:
        for args in args_list:
            print(_run_one_point(args))
    else:
        with mp.Pool(n_proc) as pool:
            for result in pool.imap_unordered(
                    _run_one_point, args_list):
                print(result)


# =============================================================================
if __name__ == '__main__':
    main()
