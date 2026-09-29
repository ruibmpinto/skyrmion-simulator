"""Sweep driver for Pham et al. (2024) Figure S49.

Two parts:

(1) Analytic Thiele evaluation (no time integration). Runs
    serially in `main()`, always.
(2) Numerical LLGS verification with the TSH torque enabled
    in `llgs_rhs`. Grid is (R/Delta, lambda_sq) and each point
    is dispatched via the standard execution-model machinery
    (serial / multiprocessing Pool / HPC array).

Set B material constants (paper S49 caption).
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
from src.simulator.pulses import ConstantPulse, SquarePulse
from src.simulator.topological_torque import (
    sot_thiele_speed,
    tsh_thiele_speed,
)
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


def _build_set_B_params(D, alpha, gamma, lambda_sq,
                        nx, ny, dt, R, Delta):
    """Build a Set B parameter namespace with imposed skyrmion
    R/Delta and TSH coupling lambda_sq."""
    p = default_params()
    p.D = D
    p.alpha = alpha
    p.gamma = gamma
    # Set B keeps the measured Hk_top = 12.4 mT (S49 caption).
    p.K_top = 1.294e6
    p.nx = nx
    p.ny = ny
    p.dt = dt
    p.lambda_sq = float(lambda_sq)
    p.skyrmion_R = float(R)
    p.skyrmion_dw = float(Delta)
    _precompute(p)
    return p


def _make_saf_skyrmion_ic(p):
    """Module-level IC factory so worker processes can pickle it."""
    return saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)


def _run_analytic(R_over_Delta_values, Delta, lambda_sq_values,
                  J0_for_b_j, out_path):
    """Build the analytic v_SOT and v_TSH curves."""
    p = _build_set_B_params(
        D=0.62e-3, alpha=0.216, gamma=175.9e9,
        lambda_sq=0.0, nx=256, ny=256, dt=5.0e-14,
        R=80.0e-9, Delta=Delta)
    p.pulse = ConstantPulse(J0_for_b_j)
    # SOT speed vs R; TSH speed vs R for each lambda_sq.
    R_arr = np.array(R_over_Delta_values, dtype=float) * Delta
    v_sot = np.array(
        [sot_thiele_speed(R, Delta, p) for R in R_arr])
    v_tsh = {
        lam: np.array(
            [tsh_thiele_speed(R, Delta, lam, p) for R in R_arr])
        for lam in lambda_sq_values
    }
    trace = {
        't': np.array([0.0]),
        'R_over_Delta': np.array(
            R_over_Delta_values, dtype=float),
        'v_SOT': v_sot,
        'snapshot_m_top': None,
        'snapshot_m_bot': None,
        'snapshot_t': None,
    }
    # Store each TSH curve under a lambda-tagged key (nm^2).
    for lam, v in v_tsh.items():
        key = f'v_TSH_lam_{lam*1e18:.0f}nm2'
        trace[key] = v
    metadata = {
        'figure': 'S49_analytic',
        'Delta': float(Delta),
        'J0': float(J0_for_b_j),
        'R_over_Delta_values': [
            float(r) for r in R_over_Delta_values],
        'lambda_sq_values_nm2': [
            float(lam * 1e18) for lam in lambda_sq_values],
        'alpha': 0.216,
        'gamma': 175.9e9,
        'D': 0.62e-3,
    }
    save_trace(path=out_path, trace=trace, metadata=metadata)
    print(f'Saved analytic curves -> {out_path}')


def _run_one_point(args):
    """Worker: LLGS-verify one (R/Delta, lambda_sq) grid point;
    returns a one-line console summary."""
    cfg, point = args
    # Imposed skyrmion size R and TSH coupling for this point.
    rd = float(point['R_over_Delta'])
    lambda_sq = float(point['lambda_sq'])
    R = rd * cfg['Delta']
    p = _build_set_B_params(
        D=cfg['D'], alpha=cfg['alpha'], gamma=cfg['gamma'],
        lambda_sq=lambda_sq,
        nx=cfg['nx'], ny=cfg['ny'], dt=cfg['dt'],
        R=R, Delta=cfg['Delta'])
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
    # Persist trace + full run metadata; tag by R/Delta, lambda, D.
    metadata = {
        'figure': 'S49_llgs',
        'demag': ('full_fft' if cfg['use_full_demag']
                  else 'local_keff'),
        'J0': float(cfg['J0']),
        't_pulse': float(cfg['t_pulse']),
        'R': float(R),
        'Delta': float(cfg['Delta']),
        'R_over_Delta': float(rd),
        'lambda_sq_nm2': float(lambda_sq * 1e18),
        'lambda_sq_m2': float(lambda_sq),
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
        f'llgs_R{rd:.1f}'
        f'_lam{lambda_sq*1e18:.0f}nm2_{_D_tag}.npz')
    out_path = os.path.join(cfg['out_dir'], out_name)
    save_trace(path=out_path, trace=trace, metadata=metadata)
    # Steady-state v proxy in the pulse middle (0.5-1.5 ns).
    t_arr = trace['t']
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=cfg['nx'] * p.a, L_y=cfg['ny'] * p.a, periodic_y=True)
    win = (t_arr > 0.5e-9) & (t_arr < 1.5e-9)
    if np.any(win):
        v_steady = (
            float(cx[win][-1] - cx[win][0])
            / float(t_arr[win][-1] - t_arr[win][0]))
    else:
        v_steady = float('nan')
    return (
        f'  R/Delta={rd:.1f}, '
        f'lambda_sq={lambda_sq*1e18:.0f} nm^2: '
        f'v_steady = {v_steady:6.1f} m/s -> {out_name}')


def main():
    """Run the S49 TSH study: analytic Thiele curves plus a
    numerical LLGS grid over `(R/Delta, lambda_sq)`, writing both
    to `output/sweeps_S41_S49/S49`.
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
    Delta = 24.5e-9
    R_over_Delta_values = [
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0,
    ]
    lambda_sq_values_tsh = [3.0e-18, 50.0e-18]
    lambda_sq_values_llgs = [0.0, 3.0e-18, 50.0e-18]
    J0 = 8.0e11
    t_pulse = 2.0e-9
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
    drive_time = t_pulse + 500.0e-12
    n_drive = int(math.ceil(drive_time / dt))
    sample_dt = 5.0e-12
    sample_every = int(math.ceil(sample_dt / dt))
    out_dir = 'output/sweeps_S41_S49/S49'
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytic part (no LLGS, always runs).
    _D_tag_top = f'D{int(round(D*1e5)):03d}e-3'
    _run_analytic(
        R_over_Delta_values=R_over_Delta_values,
        Delta=Delta,
        lambda_sq_values=lambda_sq_values_tsh,
        J0_for_b_j=J0,
        out_path=os.path.join(
            out_dir, f'analytic_{_D_tag_top}.npz'),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LLGS numerical part with optional multiprocessing.
    grid = [
        {'R_over_Delta': rd, 'lambda_sq': lam}
        for (rd, lam) in itertools.product(
            R_over_Delta_values, lambda_sq_values_llgs)
    ]
    n_proc = int(os.environ.get('SWEEP_NPROC', '1'))
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID')
    if task_id is not None:
        idx = int(task_id)
        if idx < 0 or idx >= len(grid):
            raise RuntimeError(
                f'sweep_S49_TSH (llgs): '
                f'SLURM_ARRAY_TASK_ID={task_id} out of range '
                f'[0, {len(grid) - 1}].')
        grid = [grid[idx]]
        n_proc = 1
    cfg = {
        'use_full_demag': bool(use_full_demag),
        'demag_kind': str(demag_kind),
        'demag_newell_accuracy': float(demag_newell_accuracy),
        'demag_newell_tol_conv': float(demag_newell_tol_conv),
        'J0': float(J0), 't_pulse': float(t_pulse),
        'alpha': float(alpha), 'gamma': float(gamma),
        'D': float(D), 'Delta': float(Delta),
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
    args_list = [(cfg, point) for point in grid]
    demag_tag = 'full_fft' if use_full_demag else 'local_keff'
    print(
        f'S49 LLGS: nx={nx}, demag={demag_tag}, '
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
