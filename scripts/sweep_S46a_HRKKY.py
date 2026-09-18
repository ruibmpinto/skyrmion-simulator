"""Sweep driver for Pham et al. (2024) Figure S46, panel A.

Gaussian pulse at fixed J = 8.9e11 A/m^2 and FWHM = 500 ps,
sweeping H_RKKY from 200 to 950 mT. Same execution-model
machinery as `sweep_S42_J.py`.
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
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)


def _run_one_point(args):
    """Worker: integrate one H_RKKY grid point; returns a
    one-line console summary."""
    cfg, point = args
    # Fresh parameter namespace; H_RKKY varies per-point.
    p = default_params()
    p.D = cfg['D']
    p.nx = cfg['nx']
    p.ny = cfg['ny']
    p.dt = cfg['dt']
    p.H_RKKY = float(point['H_RKKY'])
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
    # Fixed Gaussian drive pulse (J0, FWHM constant across sweep).
    pulse = GaussianPulse(
        J0=cfg['J0'], t_center=cfg['t_center'], FWHM=cfg['FWHM'])
    trace = run_one(
        p=p, pulse=pulse,
        n_relax=n_relax_for_driver, n_drive=cfg['n_drive'],
        sample_every=cfg['sample_every'],
        step_drive=step, step_relax=step,
        ic_factory=ic_factory, record_snapshot_at=None,
        print_every=10000)
    # Persist trace + full run metadata; tag filename by H_RKKY, D.
    metadata = {
        'figure': 'S46a',
        'demag': ('full_fft' if cfg['use_full_demag']
                  else 'local_keff'),
        'J0': float(cfg['J0']),
        'FWHM': float(cfg['FWHM']),
        'sigma': float(cfg['sigma']),
        't_center': float(cfg['t_center']),
        'tail_sigmas': float(cfg['tail_sigmas']),
        'H_RKKY': float(point['H_RKKY']),
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
    # PBC-aware v_avg proxy + peak diameter for the summary line.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=cfg['nx'] * p.a, L_y=cfg['ny'] * p.a, periodic_y=True)
    v_avg_proxy = (
        float(cx[-1] - cx[0])
        / (2.0 * cfg['tail_sigmas'] * cfg['sigma']))
    d_max = float(np.max(trace['d_top']))
    return (
        f'  H_RKKY = {point["H_RKKY"]*1000:>4.0f} mT: '
        f'v_avg(proxy) = {v_avg_proxy:>5.0f} m/s, '
        f'max d_top = {d_max*1e9:.0f} nm -> {out_name}')


def main():
    """Sweep the RKKY field `H_RKKY` (200-950 mT) at fixed
    J = 8.9e11 and FWHM = 500 ps, writing one trace per point to
    `output/sweeps_S41_S49/S46a`.
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
        0.200, 0.250, 0.300, 0.350, 0.400, 0.450,
        0.500, 0.550, 0.600, 0.650, 0.700, 0.750,
        0.800, 0.850, 0.900, 0.950,
    ]
    J0 = 8.9e11
    FWHM = 500.0e-12
    tail_sigmas = 3.0
    sigma = FWHM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    t_center = tail_sigmas * sigma
    # D = 0.85e-3             # legacy repo calibration (old kernels)
    # Paper Set A (supp. 1.1): D = 0.76 mJ/m^2, H_k,top = 36 mT.
    D = 0.76e-3
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
    drive_time = 2.0 * tail_sigmas * sigma + 200.0e-12
    n_drive = int(math.ceil(drive_time / dt))
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    out_dir = 'output/sweeps_S41_S49/S46a'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(H_RKKY_values):
            raise RuntimeError(
                f'sweep_S46a_HRKKY: SLURM_ARRAY_TASK_ID={task_id} '
                f'out of range [0, {len(H_RKKY_values) - 1}].')
        H_RKKY_values = [H_RKKY_values[idx]]
        n_proc = 1
    cfg = {
        'use_full_demag': bool(use_full_demag),
        'demag_kind': str(demag_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'J0': float(J0),
        'D': float(D),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'FWHM': float(FWHM), 'sigma': float(sigma),
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
    args_list = [
        (cfg, {'H_RKKY': H}) for H in H_RKKY_values]
    os.makedirs(out_dir, exist_ok=True)
    demag_tag = 'full_fft' if use_full_demag else 'local_keff'
    print(f'S46a sweep: nx={nx}, demag={demag_tag}, '
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
