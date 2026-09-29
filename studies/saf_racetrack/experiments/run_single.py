"""Single-trajectory production runner.

Builds a SAF skyrmion, optionally relaxes it under
`J = 0`, then drives it with the existing SOT model in
`skyrmion_simulator.simulator.integrator.llgs_rhs` at the user-specified
substrate temperature `T_sub` and current density `j`. The
effective lattice temperature is set by uniform Joule
heating, `T = T_sub + R_th * j**2`. Sample every
`sample_every` steps: skyrmion center (PBC wrapped),
topological charge, top-layer diameter, and the
end-of-step norm drift.

The worker `trajectory_worker(config)` is module-level
(picklable) so it can be dispatched through
`skyrmion_simulator.stochastic_llgs.ensemble.run_ensemble`. `main()` runs
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
import os
import time
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.params_helper import make_params
from skyrmion_simulator.simulator.analysis import skyrmion_diameter
from skyrmion_simulator.simulator.demag import precompute_demag_kernels
from skyrmion_simulator.simulator.initial_conditions import saf_skyrmion
from skyrmion_simulator.simulator.main import topological_charge
from skyrmion_simulator.simulator.pulses import ConstantPulse
from skyrmion_simulator.stochastic_llgs.diagnostics import (
    detect_annihilation,
    hall_angle,
    skyrmion_center_lcc_pbc,
    skyrmion_center_pbc,
    skyrmion_diameter_lcc,
    skyrmion_ellipse_lcc,
    unwrap_trajectory,
)
from skyrmion_simulator.stochastic_llgs.integrator_sllg import \
    heun_stochastic_step
from skyrmion_simulator.stochastic_llgs.io import save_trajectory
from skyrmion_simulator.stochastic_llgs.joule_heating import T_of_j
from skyrmion_simulator.stochastic_llgs.parameters_thermal import attach_thermal
from skyrmion_simulator.stochastic_llgs.thermal_field import \
    sample_thermal_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
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
    'dump_fields', 'snapshot_every',
    'demag_kind', 'demag_accuracy', 'demag_tol_conv',
    'm_init_top', 'm_init_bot', 'equil',
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
    if config['dump_fields'] not in (True, False):
        raise RuntimeError(
            f'trajectory_worker: dump_fields must be bool, '
            f'got {config["dump_fields"]!r}.'
        )
    # Field dumping needs a positive frame cadence; reject the
    # silently-broken (dump on, cadence zero) combination.
    if config['dump_fields'] and int(config['snapshot_every']) <= 0:
        raise RuntimeError(
            f'trajectory_worker: snapshot_every must be a '
            f'positive int when dump_fields is True, got '
            f'{config["snapshot_every"]!r}.'
        )
    # Demag kind must be one of the recognised formulations and
    # consistent with the use_demag flag.
    allowed_kinds = (
        'none', 'slab', 'newell', 'newell_freebc', 'racetrack')
    if config['demag_kind'] not in allowed_kinds:
        raise RuntimeError(
            f'trajectory_worker: demag_kind must be one of '
            f'{allowed_kinds}, got {config["demag_kind"]!r}.'
        )
    if config['use_demag'] and config['demag_kind'] == 'none':
        raise RuntimeError(
            'trajectory_worker: use_demag is True but demag_kind '
            'is "none"; pick "slab" or "newell".'
        )
    if not config['use_demag'] and config['demag_kind'] != 'none':
        raise RuntimeError(
            f'trajectory_worker: use_demag is False but demag_kind '
            f'is {config["demag_kind"]!r}; use "none".'
        )
    # Newell (periodic or free-BC) needs finite positive accuracy/tol.
    if config['demag_kind'] in ('newell', 'newell_freebc'):
        if not (config['demag_accuracy']
                and float(config['demag_accuracy']) > 0.0):
            raise RuntimeError(
                'trajectory_worker: demag_kind "newell"/"newell_freebc" '
                'requires a positive demag_accuracy.'
            )
        if not (config['demag_tol_conv']
                and float(config['demag_tol_conv']) > 0.0):
            raise RuntimeError(
                'trajectory_worker: demag_kind "newell" requires a '
                'positive demag_tol_conv.'
            )
    # Initial-field override: both layers present or both absent.
    has_top = config['m_init_top'] is not None
    has_bot = config['m_init_bot'] is not None
    if has_top != has_bot:
        raise RuntimeError(
            'trajectory_worker: m_init_top and m_init_bot must '
            'both be provided or both be None.'
        )
    # Equilibration spec: None (fixed n_relax) or a dict with the
    # plateau-detection parameters, all present.
    equil = config['equil']
    if equil is not None:
        if not isinstance(equil, dict):
            raise RuntimeError(
                f'trajectory_worker: equil must be None or a dict, '
                f'got {type(equil).__name__}.'
            )
        equil_keys = (
            'check_every', 'window', 'tol', 'k_consec', 'max_steps')
        missing_equil = [k for k in equil_keys if k not in equil]
        if missing_equil:
            raise RuntimeError(
                f'trajectory_worker: equil missing keys: '
                f'{missing_equil}.'
            )


# -----------------------------------------------------------------------------
def _equilibrate(m_top, m_bot, p, kernels, dt, rng, sigma,
                 tol_norm, equil):
    """Thermally equilibrate at J = 0 until the LCC size plateaus.

    Runs stochastic steps in blocks of `check_every`, sampling the
    top-layer LCC equivalent-disk diameter after each block. Declares
    convergence when the mean diameter over the last `window` samples
    matches the previous `window` within relative `tol` for
    `k_consec` consecutive checks, or stops at `max_steps`. Assumes
    `p.pulse` is already set to zero current.

    Parameters
    ----------
    m_top, m_bot : numpy.ndarray(3d)
        Spin configurations, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameters namespace (with attached thermal fields).
    kernels : dict or None
        Demag kernels.
    dt : float
        Time step in seconds.
    rng : numpy.random.Generator
        Noise generator.
    sigma : float
        Thermal noise amplitude (Tesla*sqrt(s)); 0 disables noise.
    tol_norm : float
        Norm-drift tolerance for the stepper.
    equil : dict
        Plateau-detection parameters check_every, window, tol,
        k_consec, max_steps.

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Equilibrated fields.
    n_used : int
        Number of stochastic steps taken.
    converged : bool
        Whether the plateau criterion was met.
    d1_relaxed, d2_relaxed : float
        Mean major/minor LCC axes over the final `window` checks
        (NaN if the core collapsed).
    """
    check_every = int(equil['check_every'])
    window = int(equil['window'])
    tol = float(equil['tol'])
    k_consec = int(equil['k_consec'])
    max_steps = int(equil['max_steps'])
    ny, nx = m_top.shape[:2]
    diam_hist = []
    d1_hist = []
    d2_hist = []
    stable = 0
    n_used = 0
    converged = False
    while n_used < max_steps:
        for k in range(check_every):
            if sigma > 0.0:
                h_top = sample_thermal_field(rng, (ny, nx), sigma, dt)
                h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt)
            else:
                h_top = np.zeros((ny, nx, 3), dtype=float)
                h_bot = np.zeros((ny, nx, 3), dtype=float)
            m_top, m_bot, _ = heun_stochastic_step(
                m_top, m_bot, dt, p, kernels, h_top, h_bot, tol_norm,
                t=(n_used + k) * dt)
        n_used += check_every
        diam_hist.append(float(skyrmion_diameter_lcc(
            m_top, p.a, core_polarity=+1)))
        try:
            e1, e2, _ = skyrmion_ellipse_lcc(
                m_top, p.a, core_polarity=+1)
        except RuntimeError:
            e1, e2 = float('nan'), float('nan')
        d1_hist.append(e1)
        d2_hist.append(e2)
        # Compare the last two non-overlapping windows of the size.
        if len(diam_hist) >= 2 * window:
            new_mean = float(np.mean(diam_hist[-window:]))
            old_mean = float(np.mean(diam_hist[-2 * window:-window]))
            if old_mean > 0.0 \
                    and abs(new_mean - old_mean) < tol * old_mean:
                stable += 1
                if stable >= k_consec:
                    converged = True
                    break
            else:
                stable = 0
    # Relaxed size: mean over the final window of checks.
    tail = min(window, len(d1_hist))
    if tail > 0:
        d1_relaxed = float(np.nanmean(d1_hist[-tail:]))
        d2_relaxed = float(np.nanmean(d2_hist[-tail:]))
    else:
        d1_relaxed = float('nan')
        d2_relaxed = float('nan')
    return m_top, m_bot, n_used, converged, d1_relaxed, d2_relaxed


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
            k_consecutive, dump_fields, snapshot_every,
            demag_kind, demag_accuracy, demag_tol_conv,
            m_init_top, m_init_bot, equil.
        `demag_kind` is one of 'none'/'slab'/'newell';
        `m_init_top`/`m_init_bot` are a pre-relaxed starting
        field (both or neither), else None to seed a fresh SAF
        skyrmion. Optional keys:
            traj_idx (for logging).

    Returns
    -------
    payload : dict
        Per-trajectory time series and aggregates:
            t_sample, cx_wrapped, cy_wrapped, Q, diameter,
            D1_top, D2_top, theta_top, D1_bot, D2_bot,
            theta_bot, norm_drift_max, cx_unwrapped,
            cy_unwrapped, T_effective, alive_at_end,
            flip_index, v_x, v_y, hall_deg, sigma_y.
        When `dump_fields` is True, also the full-field dumps
            field_m_{initial,relaxed,final}_{top,bot} and the
            animation frames anim_t, anim_mz_{top,bot}.
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
    dump_fields = bool(config['dump_fields'])
    snapshot_every = int(config['snapshot_every'])
    demag_kind = str(config['demag_kind'])
    demag_accuracy = config['demag_accuracy']
    demag_tol_conv = config['demag_tol_conv']
    m_init_top = config['m_init_top']
    m_init_bot = config['m_init_bot']
    equil = config['equil']
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
    # Demag kernels per the selected formulation; None disables.
    if use_demag:
        kernels = precompute_demag_kernels(
            p, kind=demag_kind,
            accuracy=demag_accuracy, tol_conv=demag_tol_conv)
    else:
        kernels = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial condition: a caller-provided (pre-relaxed) field if
    # given, else a fresh SAF skyrmion seed.
    if m_init_top is not None:
        m_top = np.array(m_init_top, dtype=float)
        m_bot = np.array(m_init_bot, dtype=float)
    else:
        m_top, m_bot = saf_skyrmion(
            p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw,
        )
    rng = np.random.default_rng(int(p.seed))
    sigma = float(p.sigma_noise)
    # Named full-field dumps (float32) for the initial / relaxed /
    # final configurations; populated only when dump_fields.
    if dump_fields:
        field_initial_top = m_top.astype(np.float32)
        field_initial_bot = m_bot.astype(np.float32)
    else:
        field_initial_top = None
        field_initial_bot = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Relaxation phase (J = 0): swap p.pulse to ConstantPulse(0)
    # so llgs_rhs sees J(t) = 0 at every substep; restore after.
    # `equil` selects fixed-duration (None) or equilibrate-to-plateau.
    # The H_DL/H_FL diagnostic scalars are zeroed alongside for
    # consistency with the other drivers (llgs_rhs never reads them).
    pulse_save = p.pulse
    H_DL_save = p.H_DL
    H_FL_save = p.H_FL
    p.pulse = ConstantPulse(0.0)
    p.H_DL = 0.0
    p.H_FL = 0.0
    if equil is None:
        for step in range(n_relax):
            if sigma > 0.0:
                h_top = sample_thermal_field(rng, (ny, nx), sigma, dt)
                h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt)
            else:
                h_top = np.zeros((ny, nx, 3), dtype=float)
                h_bot = np.zeros((ny, nx, 3), dtype=float)
            m_top, m_bot, _ = heun_stochastic_step(
                m_top, m_bot, dt, p, kernels,
                h_top, h_bot, tol_norm, t=step * dt,
            )
        n_relax_used = n_relax
        equil_converged = False
        # Single post-relax ellipse measurement for the observable.
        try:
            d1_relaxed, d2_relaxed, _ = skyrmion_ellipse_lcc(
                m_top, p.a, core_polarity=+1)
        except RuntimeError:
            d1_relaxed, d2_relaxed = float('nan'), float('nan')
    else:
        (m_top, m_bot, n_relax_used, equil_converged,
         d1_relaxed, d2_relaxed) = _equilibrate(
            m_top, m_bot, p, kernels, dt, rng, sigma,
            tol_norm, equil)
    p.pulse = pulse_save
    # Relaxed (pre-drive) full-field dump.
    if dump_fields:
        field_relaxed_top = m_top.astype(np.float32)
        field_relaxed_bot = m_bot.astype(np.float32)
    else:
        field_relaxed_top = None
        field_relaxed_bot = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drive phase: record observables every sample_every steps.
    # Ceiling division: number of sampled steps over n_drive.
    n_samples = (n_drive + sample_every - 1) // sample_every
    t_sample = np.empty(n_samples, dtype=float)
    # Top-layer centers (core m_z = -1, weight (1-m_z)/2).
    cx_w = np.empty(n_samples, dtype=float)
    cy_w = np.empty(n_samples, dtype=float)
    # Bot-layer centers (core m_z = +1, opposite polarity).
    cx_w_bot = np.empty(n_samples, dtype=float)
    cy_w_bot = np.empty(n_samples, dtype=float)
    # LCC (largest connected component) versions, robust to
    # thermal-noise blobs in the all-mask center.
    cx_lcc = np.empty(n_samples, dtype=float)
    cy_lcc = np.empty(n_samples, dtype=float)
    cx_lcc_bot = np.empty(n_samples, dtype=float)
    cy_lcc_bot = np.empty(n_samples, dtype=float)
    Q_arr = np.empty(n_samples, dtype=float)
    Q_bot_arr = np.empty(n_samples, dtype=float)
    diam = np.empty(n_samples, dtype=float)
    diam_bot = np.empty(n_samples, dtype=float)
    # LCC diameters (single-skyrmion-area, excludes blobs).
    diam_lcc = np.empty(n_samples, dtype=float)
    diam_lcc_bot = np.empty(n_samples, dtype=float)
    # Elliptical axes via second moments of the largest
    # connected core (thermal-noise-robust).
    d1_top = np.empty(n_samples, dtype=float)
    d2_top = np.empty(n_samples, dtype=float)
    th_top = np.empty(n_samples, dtype=float)
    d1_bot = np.empty(n_samples, dtype=float)
    d2_bot = np.empty(n_samples, dtype=float)
    th_bot = np.empty(n_samples, dtype=float)
    drift_max = np.empty(n_samples, dtype=float)
    s_idx = 0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Animation-frame buffers (m_z only, float32) on a separate,
    # coarser cadence; populated only when dump_fields.
    if dump_fields:
        n_anim = n_drive // snapshot_every
        anim_t = np.empty(n_anim, dtype=float)
        anim_mz_top = np.empty((n_anim, ny, nx), dtype=np.float32)
        anim_mz_bot = np.empty((n_anim, ny, nx), dtype=np.float32)
        a_idx = 0
    for step in range(1, n_drive + 1):
        # Sample independent thermal noise for both layers.
        if sigma > 0.0:
            h_top = sample_thermal_field(rng, (ny, nx), sigma, dt)
            h_bot = sample_thermal_field(rng, (ny, nx), sigma, dt)
        else:
            h_top = np.zeros((ny, nx, 3), dtype=float)
            h_bot = np.zeros((ny, nx, 3), dtype=float)
        m_top, m_bot, drift = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels, h_top, h_bot, tol_norm,
            t=(step - 1) * dt)
        if step % sample_every == 0:
            t_sample[s_idx] = step * dt
            # Top center: undefined if the core has collapsed.
            w_top = float((1.0 - m_top[..., 2]).sum())
            if w_top > 0.0:
                cx, cy = skyrmion_center_pbc(
                    m_top, p.a, core_polarity=+1)
            else:
                cx = float('nan')
                cy = float('nan')
            # Bot center: same logic with opposite polarity.
            w_bot = float((1.0 + m_bot[..., 2]).sum())
            if w_bot > 0.0:
                cxb, cyb = skyrmion_center_pbc(
                    m_bot, p.a, core_polarity=-1)
            else:
                cxb = float('nan')
                cyb = float('nan')
            cx_w[s_idx] = cx
            cy_w[s_idx] = cy
            cx_w_bot[s_idx] = cxb
            cy_w_bot[s_idx] = cyb
            # Topological charge and diameter for each layer.
            Q_arr[s_idx] = float(topological_charge(m_top, p.a))
            Q_bot_arr[s_idx] = float(topological_charge(m_bot, p.a))
            diam[s_idx] = float(skyrmion_diameter(
                m_top, p.a, core_polarity=+1))
            diam_bot[s_idx] = float(skyrmion_diameter(
                m_bot, p.a, core_polarity=-1))
            # LCC diameters and centers (the trustworthy
            # single-skyrmion measurements; the all-mask
            # versions above are kept for back-compat).
            diam_lcc[s_idx] = float(skyrmion_diameter_lcc(
                m_top, p.a, core_polarity=+1))
            diam_lcc_bot[s_idx] = float(skyrmion_diameter_lcc(
                m_bot, p.a, core_polarity=-1))
            # Elliptical axes (major D1, minor D2, orientation
            # theta) of the largest connected core.
            # skyrmion_ellipse_lcc raises if the core has
            # collapsed (<3 sites); record NaN there.
            try:
                e1, e2, et = skyrmion_ellipse_lcc(
                    m_top, p.a, core_polarity=+1)
            except RuntimeError:
                e1, e2, et = (float('nan'),) * 3
            d1_top[s_idx] = e1
            d2_top[s_idx] = e2
            th_top[s_idx] = et
            try:
                e1b, e2b, etb = skyrmion_ellipse_lcc(
                    m_bot, p.a, core_polarity=-1)
            except RuntimeError:
                e1b, e2b, etb = (float('nan'),) * 3
            d1_bot[s_idx] = e1b
            d2_bot[s_idx] = e2b
            th_bot[s_idx] = etb
            if w_top > 0.0:
                cxl, cyl = skyrmion_center_lcc_pbc(
                    m_top, p.a, core_polarity=+1)
            else:
                cxl = float('nan')
                cyl = float('nan')
            if w_bot > 0.0:
                cxlb, cylb = skyrmion_center_lcc_pbc(
                    m_bot, p.a, core_polarity=-1)
            else:
                cxlb = float('nan')
                cylb = float('nan')
            cx_lcc[s_idx] = cxl
            cy_lcc[s_idx] = cyl
            cx_lcc_bot[s_idx] = cxlb
            cy_lcc_bot[s_idx] = cylb
            drift_max[s_idx] = float(drift)
            s_idx += 1
        # Animation frame capture on the coarse snapshot cadence.
        if dump_fields and step % snapshot_every == 0 \
                and a_idx < n_anim:
            anim_t[a_idx] = step * dt
            anim_mz_top[a_idx] = m_top[..., 2].astype(np.float32)
            anim_mz_bot[a_idx] = m_bot[..., 2].astype(np.float32)
            a_idx += 1
    # Final (end-of-drive) full-field dump.
    if dump_fields:
        field_final_top = m_top.astype(np.float32)
        field_final_bot = m_bot.astype(np.float32)
    else:
        field_final_top = None
        field_final_bot = None
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
    # Unwrap top trajectory if alive long enough; drives v, theta_H.
    # Fit the LCC (largest-connected-component core) tracker: unlike
    # the whole-lattice centroid it stays on the skyrmion once other
    # domains nucleate. Free-y racetrack: y is not periodic, never
    # unwrap it.
    L_x = nx * p.a
    L_y = ny * p.a
    periodic_y = (demag_kind != 'racetrack')
    if end >= 4 and np.all(np.isfinite(cx_lcc[:end])):
        cx_u, cy_u = unwrap_trajectory(
            cx_lcc[:end], cy_lcc[:end], L_x, L_y, periodic_y)
        cx_unwrap[:end] = cx_u
        cy_unwrap[:end] = cy_u
        v_x, v_y, theta_deg = hall_angle(
            t_sample[:end], cx_u, cy_u, half=0.5)
        sigma_y = float(np.std(cy_u[end // 2:], ddof=1)) \
            if end >= 8 else float('nan')
    # Same unwrap for the bot layer (inter-layer Q3 diagnostics).
    cx_unwrap_bot = np.full(n_samples, np.nan)
    cy_unwrap_bot = np.full(n_samples, np.nan)
    v_x_bot = float('nan')
    v_y_bot = float('nan')
    theta_deg_bot = float('nan')
    if end >= 4 and np.all(np.isfinite(cx_lcc_bot[:end])):
        cxb_u, cyb_u = unwrap_trajectory(
            cx_lcc_bot[:end], cy_lcc_bot[:end], L_x, L_y, periodic_y)
        cx_unwrap_bot[:end] = cxb_u
        cy_unwrap_bot[:end] = cyb_u
        v_x_bot, v_y_bot, theta_deg_bot = hall_angle(
            t_sample[:end], cxb_u, cyb_u, half=0.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    payload = {
        't_sample': t_sample,
        # Top layer
        'cx_wrapped': cx_w, 'cy_wrapped': cy_w,
        'cx_unwrapped': cx_unwrap, 'cy_unwrapped': cy_unwrap,
        'Q': Q_arr, 'diameter': diam,
        # Elliptical axes (major, minor, orientation), top layer
        'D1_top': d1_top, 'D2_top': d2_top, 'theta_top': th_top,
        # Bot layer (inter-layer Q3 diagnostics)
        'cx_wrapped_bot': cx_w_bot, 'cy_wrapped_bot': cy_w_bot,
        'cx_unwrapped_bot': cx_unwrap_bot,
        'cy_unwrapped_bot': cy_unwrap_bot,
        'Q_bot': Q_bot_arr, 'diameter_bot': diam_bot,
        # Elliptical axes, bot layer
        'D1_bot': d1_bot, 'D2_bot': d2_bot, 'theta_bot': th_bot,
        # LCC variants (Q3a-clean, single-skyrmion)
        'cx_lcc': cx_lcc, 'cy_lcc': cy_lcc,
        'cx_lcc_bot': cx_lcc_bot, 'cy_lcc_bot': cy_lcc_bot,
        'diameter_lcc': diam_lcc,
        'diameter_lcc_bot': diam_lcc_bot,
        # Bookkeeping
        'norm_drift_max': drift_max,
        'T_effective': float(T_eff),
        'sigma_noise': sigma,
        # Physical box extent (m); track width is L_y.
        'L_x': float(L_x), 'L_y': float(L_y),
        # Pre-drive (J=0) finite-T equilibrium size and the
        # thermal-equilibration outcome.
        'D1_relaxed_top': float(d1_relaxed),
        'D2_relaxed_top': float(d2_relaxed),
        'n_relax_used': int(n_relax_used),
        'equil_converged': bool(equil_converged),
        'alive_at_end': alive_at_end,
        'flip_index': int(flip_index),
        # Final top-layer m_z snapshot: the per-realization
        # survival criterion is the field classifier
        # (skyrmion_simulator.stochastic_llgs.stability), applied at
        # aggregation.
        'mz_final_top': m_top[..., 2].astype(np.float32),
        # Drift / Hall fit, top
        'v_x': v_x, 'v_y': v_y,
        'velocity': float(np.sqrt(v_x ** 2 + v_y ** 2))
        if np.isfinite(v_x) else float('nan'),
        'hall_deg': theta_deg, 'sigma_y': sigma_y,
        # Drift / Hall fit, bot
        'v_x_bot': v_x_bot, 'v_y_bot': v_y_bot,
        'velocity_bot': float(
            np.sqrt(v_x_bot ** 2 + v_y_bot ** 2))
        if np.isfinite(v_x_bot) else float('nan'),
        'hall_deg_bot': theta_deg_bot,
    }
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Optional full-field dumps for visualization. The driver
    # separates these `field_`/`anim_` keys into a dedicated file
    # so the per-trajectory NPZ stays light for aggregation.
    if dump_fields:
        payload['field_m_initial_top'] = field_initial_top
        payload['field_m_initial_bot'] = field_initial_bot
        payload['field_m_relaxed_top'] = field_relaxed_top
        payload['field_m_relaxed_bot'] = field_relaxed_bot
        payload['field_m_final_top'] = field_final_top
        payload['field_m_final_bot'] = field_final_bot
        payload['anim_t'] = anim_t[:a_idx]
        payload['anim_mz_top'] = anim_mz_top[:a_idx]
        payload['anim_mz_bot'] = anim_mz_bot[:a_idx]
    return payload


# -----------------------------------------------------------------------------
def main():
    """Standalone single-trajectory driver; no argparse."""
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
    demag_kind      = 'none'         # 'none'/'slab'/'newell'
    demag_accuracy  = None           # newell only
    demag_tol_conv  = None           # newell only
    q_threshold     = 0.5
    k_consecutive   = 10
    dump_fields     = True
    snapshot_every  = 200            # 10 ps between anim frames
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
        'dump_fields': dump_fields,
        'snapshot_every': snapshot_every,
        'demag_kind': demag_kind,
        'demag_accuracy': demag_accuracy,
        'demag_tol_conv': demag_tol_conv,
        'm_init_top': None,
        'm_init_bot': None,
        'equil': None,
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
