"""Trajectory diagnostics for the stochastic LLGS solver.

PBC-aware skyrmion-center tracking, topological-charge
time-series, annihilation detection with a consecutive-step
guard, and a Hall-angle fit. The PBC-aware center is computed
via the phase-of-circular-mean estimator rather than the
naive weighted mean used by
`src.simulator.analysis.skyrmion_center`: a skyrmion straddling
a periodic boundary gives the right center under the
circular-mean estimator and a wrong one under the naive
estimator.

Functions
---------
skyrmion_center_pbc
    Phase-of-circular-mean skyrmion center on a single
    snapshot.
unwrap_trajectory
    Remove `L`-jumps from a piecewise center trajectory to
    yield a continuous position history.
detect_annihilation
    First step at which `|Q|` stays below `q_threshold` for
    `k_consecutive` consecutive samples.
hall_angle
    Linear-fit Hall angle on the second half of an unwrapped
    trajectory.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def skyrmion_center_pbc(m, a, core_polarity):
    """Phase-of-circular-mean skyrmion center under PBC.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3). Periodic in
        both lattice directions.
    a : float
        Lattice constant in meters. Strictly positive.
    core_polarity : {+1, -1}
        Sign convention for the skyrmion core.
        `+1` if the core sits at `m_z = -1` (top-layer
        convention used by `saf_skyrmion`); `-1` if the core
        sits at `m_z = +1` (bottom layer). Required, no
        default, to match `simulator.analysis.skyrmion_center`.

    Returns
    -------
    cx : float
        Center x in meters, modulo `L_x = nx * a`.
    cy : float
        Center y in meters, modulo `L_y = ny * a`.

    Notes
    -----
    Weights `w(r) = (1 - core_polarity * m_z(r)) / 2` so the
    skyrmion core is the bright region for either polarity.
    Cartesian coordinates are mapped onto the unit circle by
    `theta_r = 2 pi * j / nx` (x) and `2 pi * i / ny` (y);
    the weighted circular mean gives the center modulo the
    box length.
    """
    if core_polarity not in (+1, -1):
        raise RuntimeError(
            f'skyrmion_center_pbc: core_polarity must be +1 '
            f'or -1, got {core_polarity!r}.')
    if not (np.isfinite(a) and a > 0.0):
        raise RuntimeError(
            f'skyrmion_center_pbc: a must be finite and '
            f'strictly positive, got {a!r}.'
        )
    if m.ndim != 3 or m.shape[-1] != 3:
        raise RuntimeError(
            f'skyrmion_center_pbc: m must have shape '
            f'(ny, nx, 3), got {m.shape}.'
        )
    ny, nx = m.shape[:2]
    # Polarity-aware weight: peaks at the core for either layer.
    w = (1.0 - core_polarity * m[..., 2]) / 2.0
    ws = float(w.sum())
    if ws <= 0.0:
        raise RuntimeError(
            f'skyrmion_center_pbc: total weight is '
            f'non-positive ({ws!r}); no core region present '
            f'for polarity={core_polarity:+d} '
            f'(skyrmion may have annihilated).'
        )
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
    )
    theta_x = 2.0 * np.pi * jj / nx
    theta_y = 2.0 * np.pi * ii / ny
    Sx = float(np.sum(w * np.sin(theta_x)))
    Cx = float(np.sum(w * np.cos(theta_x)))
    Sy = float(np.sum(w * np.sin(theta_y)))
    Cy = float(np.sum(w * np.cos(theta_y)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # atan2 returns in (-pi, pi]; shift to [0, 2*pi)
    ang_x = np.arctan2(Sx, Cx)
    if ang_x < 0.0:
        ang_x += 2.0 * np.pi
    ang_y = np.arctan2(Sy, Cy)
    if ang_y < 0.0:
        ang_y += 2.0 * np.pi
    cx = ang_x * nx * a / (2.0 * np.pi)
    cy = ang_y * ny * a / (2.0 * np.pi)
    return cx, cy


# -----------------------------------------------------------------------------
def unwrap_trajectory(cx_series, cy_series, L_x, L_y):
    """Remove L-jumps from a wrapped trajectory.

    Parameters
    ----------
    cx_series : numpy.ndarray(1d)
        Wrapped center-x history in meters, shape (n,).
    cy_series : numpy.ndarray(1d)
        Wrapped center-y history in meters, shape (n,).
    L_x : float
        Box length in x, in meters. Strictly positive.
    L_y : float
        Box length in y, in meters. Strictly positive.

    Returns
    -------
    cx_unwrapped : numpy.ndarray(1d)
        Continuous x history (no wraps), shape (n,).
    cy_unwrapped : numpy.ndarray(1d)
        Continuous y history.

    Notes
    -----
    Jumps larger than `L / 2` between consecutive samples are
    interpreted as PBC wraps and removed by adding +/- L.
    The implementation requires inter-sample displacement to
    be smaller than `L / 2`; if not, the unwrap is
    ambiguous and a `RuntimeError` would have to come from
    the caller's sampling cadence rather than this function.
    """
    if not (np.isfinite(L_x) and L_x > 0.0 and
            np.isfinite(L_y) and L_y > 0.0):
        raise RuntimeError(
            f'unwrap_trajectory: L_x and L_y must be finite '
            f'and strictly positive, got '
            f'{L_x!r}, {L_y!r}.'
        )
    cx = np.asarray(cx_series, dtype=float).copy()
    cy = np.asarray(cy_series, dtype=float).copy()
    if cx.shape != cy.shape or cx.ndim != 1:
        raise RuntimeError(
            f'unwrap_trajectory: cx_series and cy_series '
            f'must be 1D arrays of equal length, got '
            f'{cx.shape!r}, {cy.shape!r}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    dx = np.diff(cx)
    dy = np.diff(cy)
    dx_corr = np.where(dx > 0.5 * L_x, dx - L_x, dx)
    dx_corr = np.where(dx_corr < -0.5 * L_x,
                       dx_corr + L_x, dx_corr)
    dy_corr = np.where(dy > 0.5 * L_y, dy - L_y, dy)
    dy_corr = np.where(dy_corr < -0.5 * L_y,
                       dy_corr + L_y, dy_corr)
    cx_unwrapped = np.concatenate(([cx[0]], cx[0] + np.cumsum(dx_corr)))
    cy_unwrapped = np.concatenate(([cy[0]], cy[0] + np.cumsum(dy_corr)))
    return cx_unwrapped, cy_unwrapped


# -----------------------------------------------------------------------------
def detect_annihilation(Q_history, q_threshold, k_consecutive):
    """First step at which `|Q|` stays below `q_threshold`
    for `k_consecutive` consecutive samples.

    Parameters
    ----------
    Q_history : numpy.ndarray(1d)
        Topological charge time-series, shape (n,).
    q_threshold : float
        Absolute-value threshold below which `Q` is
        considered consistent with annihilation. Strictly
        positive (typically `0.5`).
    k_consecutive : int
        Number of consecutive samples that must satisfy
        `|Q| < q_threshold` for an annihilation to be
        declared. Strictly positive.

    Returns
    -------
    flip_index : int
        Index of the last sample of the first run of
        `k_consecutive` below-threshold samples, or `-1` if
        the skyrmion never annihilated within the history.

    Notes
    -----
    The K-consecutive guard suppresses false positives from
    finite-T fluctuations in `Q` near the boundary of the
    skyrmion. With our central-difference `Q` (see
    `src.simulator.main.topological_charge`) the per-step
    noise is order 0.1 for a 256x256 lattice at T = 300 K;
    `q_threshold = 0.5`, `k_consecutive >= 10` is usually
    enough.
    """
    if not (np.isfinite(q_threshold) and q_threshold > 0.0):
        raise RuntimeError(
            f'detect_annihilation: q_threshold must be '
            f'finite and strictly positive, got '
            f'{q_threshold!r}.'
        )
    if not isinstance(k_consecutive, (int, np.integer)) \
            or k_consecutive <= 0:
        raise RuntimeError(
            f'detect_annihilation: k_consecutive must be a '
            f'positive int, got {k_consecutive!r}.'
        )
    Q = np.asarray(Q_history, dtype=float)
    if Q.ndim != 1:
        raise RuntimeError(
            f'detect_annihilation: Q_history must be 1D, '
            f'got shape {Q.shape}.'
        )
    below = np.abs(Q) < q_threshold
    if below.size < k_consecutive:
        return -1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sliding-window "all True" via cumulative product
    run_length = np.zeros_like(below, dtype=np.int64)
    run_length[0] = int(below[0])
    for i in range(1, below.size):
        if below[i]:
            run_length[i] = run_length[i - 1] + 1
        else:
            run_length[i] = 0
    hits = np.where(run_length >= int(k_consecutive))[0]
    if hits.size == 0:
        return -1
    return int(hits[0])


# -----------------------------------------------------------------------------
def hall_angle(t_series, cx_unwrapped, cy_unwrapped, half):
    """Linear-fit Hall angle on the second half of an
    unwrapped trajectory.

    Parameters
    ----------
    t_series : numpy.ndarray(1d)
        Sample times in seconds, shape (n,). Monotonically
        increasing.
    cx_unwrapped : numpy.ndarray(1d)
        Unwrapped center-x history in meters, shape (n,).
    cy_unwrapped : numpy.ndarray(1d)
        Unwrapped center-y history in meters, shape (n,).
    half : float
        Fraction of the trajectory at the end to use for the
        fit (0 < half < 1, e.g. 0.5 for the second half).

    Returns
    -------
    v_x : float
        Linear-fit drift velocity along x, in m/s.
    v_y : float
        Linear-fit drift velocity along y, in m/s.
    theta_deg : float
        Hall angle `arctan2(v_y, v_x)` in degrees.
    """
    t = np.asarray(t_series, dtype=float)
    x = np.asarray(cx_unwrapped, dtype=float)
    y = np.asarray(cy_unwrapped, dtype=float)
    if t.ndim != 1 or x.shape != t.shape or y.shape != t.shape:
        raise RuntimeError(
            f'hall_angle: shapes mismatch: t {t.shape}, x '
            f'{x.shape}, y {y.shape}.'
        )
    if not (0.0 < half < 1.0):
        raise RuntimeError(
            f'hall_angle: half must be in (0, 1), got '
            f'{half!r}.'
        )
    n = t.size
    if n < 4:
        raise RuntimeError(
            f'hall_angle: need at least 4 samples to fit, '
            f'got {n}.'
        )
    i0 = int((1.0 - half) * n)
    t_fit = t[i0:]
    x_fit = x[i0:]
    y_fit = y[i0:]
    vx = float(np.polyfit(t_fit, x_fit, 1)[0])
    vy = float(np.polyfit(t_fit, y_fit, 1)[0])
    theta_deg = float(np.degrees(np.arctan2(vy, vx)))
    return vx, vy, theta_deg
