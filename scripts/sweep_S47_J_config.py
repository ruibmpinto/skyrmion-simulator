"""Sweep driver for Pham et al. (2024) Figure S47.

Three drive configurations crossed with a 10-point J sweep,
plus a spin snapshot at the pulse peak for the highest J of
each configuration. Same execution-model machinery as the
other sweeps (serial, multiprocessing Pool via SWEEP_NPROC,
or SLURM_ARRAY_TASK_ID).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import itertools
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
    """Worker: integrate one (config, J) grid point; returns a
    one-line console summary."""
    cfg, point = args
    # Per-point Gaussian sigma + drive window; FWHM and H_RKKY
    # both come from the config selected for this point.
    FWHM = float(point['FWHM'])
    H_RKKY = float(point['H_RKKY'])
    sigma = FWHM / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    t_center = cfg['tail_sigmas'] * sigma
    drive_time = 2.0 * cfg['tail_sigmas'] * sigma + 200.0e-12
    n_drive = int(math.ceil(drive_time / cfg['dt']))
    # Record a spin snapshot at the pulse peak only for flagged
    # points (highest J of each config).
    record_snapshot_at = (
        t_center if point['record_snapshot'] else None)
    # Fresh parameter namespace; H_RKKY varies per config.
    p = default_params()
    p.D = cfg['D']
    p.nx = cfg['nx']; p.ny = cfg['ny']; p.dt = cfg['dt']
    p.H_RKKY = H_RKKY
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
    # Gaussian drive pulse for this J0.
    pulse = GaussianPulse(
        J0=point['J0'], t_center=t_center, FWHM=FWHM)
    trace = run_one(
        p=p, pulse=pulse,
        n_relax=n_relax_for_driver, n_drive=n_drive,
        sample_every=cfg['sample_every'],
        step_drive=step, step_relax=step,
        ic_factory=ic_factory,
        record_snapshot_at=record_snapshot_at,
        print_every=10000)
    # Persist trace + full run metadata; tag filename by cfg, J, D.
    metadata = {
        'figure': 'S47',
        'demag': ('full_fft' if cfg['use_full_demag']
                  else 'local_keff'),
        'cfg_idx': int(point['cfg_idx']),
        'J0': float(point['J0']),
        'FWHM': float(FWHM),
        'sigma': float(sigma),
        't_center': float(t_center),
        'tail_sigmas': float(cfg['tail_sigmas']),
        'H_RKKY': float(H_RKKY),
        'D': float(cfg['D']),
        'nx': int(cfg['nx']), 'ny': int(cfg['ny']),
        'dt': float(cfg['dt']),
        'n_relax': int(n_relax_metadata),
        'n_drive': int(n_drive),
        'sample_every': int(cfg['sample_every']),
        'snapshot_requested_at': (
            float(record_snapshot_at)
            if record_snapshot_at is not None else -1.0),
    }
    metadata.update(relax_extra)
    _D_tag = f'D{int(round(cfg["D"]*1e5)):03d}e-3'
    out_name = (
        f'cfg{point["cfg_idx"]}'
        f'_J_{point["J0"]:.2e}_{_D_tag}.npz')
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # PBC-aware v_avg proxy, peak diameter, snapshot flag.
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=cfg['nx'] * p.a, L_y=cfg['ny'] * p.a, periodic_y=True)
    v_avg_proxy = (
        float(cx[-1] - cx[0])
        / (2.0 * cfg['tail_sigmas'] * sigma))
    d_max = float(np.max(trace['d_top']))
    has_snap = trace['snapshot_t'] is not None
    return (
        f'  cfg={point["cfg_idx"]} '
        f'(FWHM={FWHM*1e12:>4.0f} ps, '
        f'H_RKKY={H_RKKY*1e3:>4.0f} mT), '
        f'J={point["J0"]:.2e}: v_avg(proxy)={v_avg_proxy:>5.0f} '
        f'm/s, d_max={d_max*1e9:.0f} nm, snap={has_snap} '
        f'-> {out_name}')


def main():
    """Sweep `J` (10 points) across three drive configurations,
    plus a spin snapshot at the pulse peak for each
    configuration's highest `J`; traces written to
    `output/sweeps_S41_S49/S47`.
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
    configs = [
        {'idx': 0, 'FWHM': 500.0e-12, 'H_RKKY': 0.205},
        {'idx': 1, 'FWHM': 500.0e-12, 'H_RKKY': 0.410},
        {'idx': 2, 'FWHM': 100.0e-12, 'H_RKKY': 0.205},
    ]
    J_values = [
        0.5e11, 1.0e11, 1.5e11, 2.0e11, 3.0e11,
        4.0e11, 5.0e11, 6.0e11, 8.0e11, 8.9e11,
    ]
    tail_sigmas = 3.0
    D = 0.85e-3
    nx = 256; ny = 256; dt = 5.0e-14
    relax_time = 500.0e-12
    n_relax_fixed = int(math.ceil(relax_time / dt))
    relax_max_steps = 200_000
    relax_alpha = 1.0
    relax_tol_torque = 1.0e-5
    relax_tol_dE = 1.0e-8
    relax_check_every = 1_000
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    out_dir = 'output/sweeps_S41_S49/S47'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cross every config with every J; flag the highest-J point of
    # each config for a peak-pulse spin snapshot.
    J_max = float(max(J_values))
    grid = []
    for cfg_dict, J0 in itertools.product(configs, J_values):
        grid.append({
            'cfg_idx': cfg_dict['idx'],
            'FWHM': cfg_dict['FWHM'],
            'H_RKKY': cfg_dict['H_RKKY'],
            'J0': float(J0),
            'record_snapshot': abs(J0 - J_max) < 1e-3,
        })
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'sweep_S47_J_config: '
                f'SLURM_ARRAY_TASK_ID={task_id} out of range '
                f'[0, {len(grid) - 1}].')
        grid = [grid[idx]]
        n_proc = 1
    cfg = {
        'use_full_demag': bool(use_full_demag),
        'demag_kind': str(demag_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'D': float(D),
        'nx': int(nx), 'ny': int(ny), 'dt': float(dt),
        'tail_sigmas': float(tail_sigmas),
        'sample_every': int(sample_every),
        'n_relax_fixed': int(n_relax_fixed),
        'relax_max_steps': int(relax_max_steps),
        'relax_alpha': float(relax_alpha),
        'relax_tol_torque': float(relax_tol_torque),
        'relax_tol_dE': float(relax_tol_dE),
        'relax_check_every': int(relax_check_every),
        'out_dir': out_dir,
    }
    args_list = [(cfg, point) for point in grid]
    os.makedirs(out_dir, exist_ok=True)
    demag_tag = 'full_fft' if use_full_demag else 'local_keff'
    print(f'S47 sweep: nx={nx}, demag={demag_tag}, '
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
