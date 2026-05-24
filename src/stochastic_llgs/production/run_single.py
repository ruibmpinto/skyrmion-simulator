"""Single-trajectory production runner.

Builds a SAF skyrmion, optionally relaxes it under
`J = 0`, then drives it with the existing SOT model in
`src.simulator.integrator.llgs_rhs` at the user-specified
substrate temperature `T_sub` and current density `j`. The
effective lattice temperature is set by uniform Joule
heating, `T = T_sub + R_th * j**2`. Sample every
`sample_every` steps: skyrmion center (PBC wrapped),
topological charge, top-layer diameter, and the
end-of-step norm drift.

The worker `trajectory_worker(config)` is module-level
(picklable) so it can be dispatched through
`src.stochastic_llgs.ensemble.run_ensemble`. `main()` runs
a single trajectory standalone.

Functions
---------
trajectory_worker
    Run one SAF skyrmion stochastic trajectory; return a
    payload dict.
main
    Standalone single-trajectory driver, no argparse.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import copy
import os
import time
# Third-party
import numpy as np
# Local
from src.phase_diagram.params_helper import make_params
from src.simulator.analysis import skyrmion_diameter
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.main import topological_charge
from src.simulator.pulses import ConstantPulse
from src.stochastic_llgs.diagnostics import (
    detect_annihilation,
    hall_angle,
    skyrmion_center_pbc,
    unwrap_trajectory,
)
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
from src.stochastic_llgs.io import save_trajectory
from src.stochastic_llgs.joule_heating import T_of_j
from src.stochastic_llgs.parameters_thermal import attach_thermal
from src.stochastic_llgs.thermal_field import sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
_REQUIRED_KEYS = (
    'T_sub', 'R_th', 'j_current', 'nx', 'ny',
    'dt', 'n_relax', 'n_drive', 'sample_every',
    'seed', 'tol_norm', 'use_demag',
    'q_threshold', 'k_consecutive', 'param_overrides',
)


def _validate_config(config):
    """Hard-fail if any required key is missing or invalid."""
    missing = [k for k in _REQUIRED_KEYS if k not in config]
    if missing:
        raise RuntimeError(
            f'trajectory_worker: config missing keys: '
            f'{missing}.'
        )
    if config['use_demag'] not in (True, False):
        raise RuntimeError(
            f'trajectory_worker: use_demag must be bool, '
            f'got {config["use_demag"]!r}.'
        )
    if not isinstance(config['param_overrides'], dict):
        raise RuntimeError(
            f'trajectory_worker: param_overrides must be a '
            f'dict (use {{}} for none), got '
            f'{type(config["param_overrides"]).__name__}.'
        )


# -----------------------------------------------------------------------------
def trajectory_worker(config):
    """Run one stochastic SAF skyrmion trajectory.

    Parameters
    ----------
    config : dict
        Required keys (no defaults):
            T_sub, R_th, j_current, nx, ny, dt,
            n_relax, n_drive, sample_every, seed,
            tol_norm, use_demag, q_threshold,
            k_consecutive.
        Optional keys:
            traj_idx (for logging).

    Returns
    -------
    payload : dict
        Per-trajectory time series and aggregates:
            t_sample, cx_wrapped, cy_wrapped, Q, diameter,
            norm_drift_max, cx_unwrapped, cy_unwrapped,
            T_effective, alive_at_end, flip_index,
            v_x, v_y, hall_deg, sigma_y.
    """
    _validate_config(config)
    T_sub = float(config['T_sub'])
    R_th = float(config['R_th'])
    j = float(config['j_current'])
    nx = int(config['nx'])
    ny = int(config['ny'])
    dt = float(config['dt'])
    n_relax = int(config['n_relax'])
    n_drive = int(config['n_drive'])
    sample_every = int(config['sample_every'])
    seed = int(config['seed'])
    tol_norm = float(config['tol_norm'])
    use_demag = bool(config['use_demag'])
    q_threshold = float(config['q_threshold'])
    k_consecutive = int(config['k_consecutive'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Joule heating
    T_eff = T_of_j(j, T_sub=T_sub, R_th=R_th)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parameters with current, lattice, and user overrides
    overrides = dict(config['param_overrides'])
    overrides['nx'] = nx
    overrides['ny'] = ny
    overrides['J_current'] = j
    p = make_params(**overrides)
    p.dt = dt
    # make_params re-derives p.H_DL/p.H_FL from J_current but does
    # NOT update p.pulse; the refactored llgs_rhs reads
    # p.pulse(t) directly, so any J_current override must be
    # mirrored into a new pulse object.
    p.pulse = ConstantPulse(j)
    attach_thermal(p, T=T_eff, R_th=R_th, seed=seed)
    kernels = (precompute_demag_kernels(
        p, kind='slab', accuracy=None, tol_conv=None)
        if use_demag else None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial condition: SAF skyrmion
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw,
    )
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relaxation phase (J = 0): swap p.pulse to ConstantPulse(0)
    # so llgs_rhs sees J(t) = 0 at every substep; restore after.
    pulse_save = p.pulse
    p.pulse = ConstantPulse(0.0)
    for _ in range(n_relax):
        h_top = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        m_top, m_bot, _ = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels,
            h_top, h_bot, tol_norm,
        )
    p.pulse = pulse_save
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive phase: record observables every sample_every steps
    n_samples = (n_drive + sample_every - 1) // sample_every
    t_sample = np.empty(n_samples, dtype=float)
    cx_w = np.empty(n_samples, dtype=float)
    cy_w = np.empty(n_samples, dtype=float)
    Q_arr = np.empty(n_samples, dtype=float)
    diam = np.empty(n_samples, dtype=float)
    drift_max = np.empty(n_samples, dtype=float)
    s_idx = 0
    for step in range(1, n_drive + 1):
        h_top = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt) \
            if sigma > 0.0 \
            else np.zeros((ny, nx, 3), dtype=float)
        m_top, m_bot, drift = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels,
            h_top, h_bot, tol_norm,
        )
        if step % sample_every == 0:
            t_sample[s_idx] = step * dt
            # Expected condition: when the skyrmion has
            # collapsed there is no m_z < 0 region, so the
            # circular-mean center is undefined. Record NaN
            # for the center but keep stepping (Q is still
            # measurable and tells us the skyrmion died).
            w_sum = float((1.0 - m_top[..., 2]).sum())
            if w_sum > 0.0:
                cx, cy = skyrmion_center_pbc(
                    m_top, p.a, core_polarity=+1)
            else:
                cx = float('nan')
                cy = float('nan')
            cx_w[s_idx] = cx
            cy_w[s_idx] = cy
            Q_arr[s_idx] = float(topological_charge(m_top, p.a))
            diam[s_idx] = float(
                skyrmion_diameter(m_top, p.a, core_polarity=+1))
            drift_max[s_idx] = float(drift)
            s_idx += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Annihilation detection (works even if center is NaN)
    flip_index = detect_annihilation(
        Q_arr, q_threshold=q_threshold,
        k_consecutive=k_consecutive,
    )
    alive_at_end = bool(flip_index == -1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PBC unwrap and Hall fit. If skyrmion died, only unwrap
    # the pre-death portion.
    if flip_index == -1:
        end = n_samples
    else:
        end = max(int(flip_index), 4)
    cx_unwrap = np.full(n_samples, np.nan)
    cy_unwrap = np.full(n_samples, np.nan)
    v_x = float('nan')
    v_y = float('nan')
    theta_deg = float('nan')
    sigma_y = float('nan')
    if end >= 4 and np.all(np.isfinite(cx_w[:end])):
        L_x = nx * p.a
        L_y = ny * p.a
        cx_u, cy_u = unwrap_trajectory(
            cx_w[:end], cy_w[:end], L_x, L_y,
        )
        cx_unwrap[:end] = cx_u
        cy_unwrap[:end] = cy_u
        v_x, v_y, theta_deg = hall_angle(
            t_sample[:end], cx_u, cy_u, half=0.5,
        )
        sigma_y = float(np.std(cy_u[end // 2:], ddof=1)) \
            if end >= 8 else float('nan')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    payload = {
        't_sample': t_sample,
        'cx_wrapped': cx_w,
        'cy_wrapped': cy_w,
        'cx_unwrapped': cx_unwrap,
        'cy_unwrapped': cy_unwrap,
        'Q': Q_arr,
        'diameter': diam,
        'norm_drift_max': drift_max,
        'T_effective': float(T_eff),
        'sigma_noise': sigma,
        'alive_at_end': alive_at_end,
        'flip_index': int(flip_index),
        'v_x': v_x, 'v_y': v_y,
        'velocity': float(
            np.sqrt(v_x ** 2 + v_y ** 2)
        ) if np.isfinite(v_x) else float('nan'),
        'hall_deg': theta_deg,
        'sigma_y': sigma_y,
    }
    return payload


# -----------------------------------------------------------------------------
def main():
    # =========================== User Configuration =========================
    t_sub           = 300.0          # K
    r_th            = 0.0            # K m^4 / A^2 (0 disables)
    j_current       = 4.0e11         # A/m^2
    nx              = 256
    ny              = 256
    dt              = 5.0e-14        # s
    n_relax         = 10000          # 500 ps
    n_drive         = 20000          # 1 ns
    sample_every    = 100            # 5 ps between samples
    seed            = 17
    tol_norm        = 5.0e-3
    use_demag       = False
    q_threshold     = 0.5
    k_consecutive   = 10
    out_dir         = 'output/stochastic_llgs'
    out_npz         = 'run_single.npz'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    config = {
        'T_sub': t_sub, 'R_th': r_th, 'j_current': j_current,
        'nx': nx, 'ny': ny, 'dt': dt,
        'n_relax': n_relax, 'n_drive': n_drive,
        'sample_every': sample_every,
        'seed': seed, 'tol_norm': tol_norm,
        'use_demag': use_demag,
        'q_threshold': q_threshold,
        'k_consecutive': k_consecutive,
        'param_overrides': {},
    }
    print('run_single: stochastic SAF skyrmion trajectory')
    print('-' * 56)
    print(
        f'T_sub={t_sub} K  j={j_current:.2e} A/m^2  '
        f'lattice={nx}x{ny}  demag={use_demag}'
    )
    t0 = time.time()
    payload = trajectory_worker(config)
    dt_wall = time.time() - t0
    out_path = os.path.join(out_dir, out_npz)
    save_trajectory(out_path, payload, config)
    print('-' * 56)
    print(
        f'T_eff = {payload["T_effective"]:.1f} K, '
        f'alive = {payload["alive_at_end"]}, '
        f'flip_index = {payload["flip_index"]}'
    )
    print(
        f'v = {payload["velocity"]:.1f} m/s '
        f'(vx={payload["v_x"]:.1f}, '
        f'vy={payload["v_y"]:.1f}), '
        f'theta_H = {payload["hall_deg"]:.2f} deg'
    )
    print(f'wall time {dt_wall:.1f} s; saved {out_path}')


# =============================================================================
if __name__ == '__main__':
    main()
