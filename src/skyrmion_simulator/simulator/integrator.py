"""LLGS time integration for SAF skyrmion dynamics.

Implements the explicit Landau-Lifshitz-Gilbert-Slonczewski
equation and a 4th-order Runge-Kutta stepper for two
coupled magnetic layers.

Functions
---------
normalize
    Normalize spin vectors to unit length.
llgs_rhs
    Compute dm/dt from the explicit LLGS equation.
rhs_local_keff
    Effective-field + dm/dt RHS using local K_eff (no demag).
rhs_demag
    Factory returning a RHS that uses explicit FFT demag.
rk4_step
    Advance both layers by one RK4 step using a supplied RHS.
rk4_step_single
    Advance one layer by one RK4 step (single-FM benchmarks).
zhang_li_torque
    Zhang-Li adiabatic + non-adiabatic spin-transfer torque.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.fields import (
    effective_field,
    effective_field_demag_pair,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def normalize(m):
    """Normalize spin vectors to unit length.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).

    Returns
    -------
    m_norm : numpy.ndarray(3d)
        Normalized spins, shape (ny, nx, 3).
    """
    # Enforces |m| = 1 site-wise, correcting RK4 drift off the sphere.
    norm = np.sqrt(m[..., 0] ** 2 + m[..., 1] ** 2 + m[..., 2] ** 2)
    # Loud rejection of a zero-magnitude spin (undefined unit vector);
    # silent division would propagate NaN through the integrator.
    if np.any(norm == 0.0):
        raise RuntimeError(
            'normalize: zero-magnitude spin at one or more lattice '
            'sites.')
    return m / norm[..., np.newaxis]


# ---------------------------------------------------------------------
def llgs_rhs(m, H_eff, p, t, mask=None):
    """Compute dm/dt from the explicit LLGS equation.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    H_eff : numpy.ndarray(3d)
        Effective field, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulation parameters. Must expose `p.pulse`, a callable
        mapping time (s) -> current density (A/m^2), and the
        SOT coefficients `p.DL_SOT` and `p.FL_SOT`.
    t : float
        Time in seconds, used to evaluate `p.pulse(t)`.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array, True inside the magnetic region.
        Vacuum sites get dm/dt = 0 exactly: the effective field
        is already zero there, but the SOT (and TSH) torques do
        not depend on H and would otherwise rotate vacuum spins.

    Returns
    -------
    dmdt : numpy.ndarray(3d)
        Time derivative of m, shape (ny, nx, 3).

    Notes
    -----
    Explicit form of the LLGS equation:

    dm/dt = 1/(1+a^2) * [
        -gamma (m x H) - gamma*alpha m x (m x H)
        + (tau_DL + alpha*tau_FL)(m x s)
        + (tau_FL - alpha*tau_DL) s
    ]

    where s = p_hat x m (unnormalized, standard Slonczewski
    form). The SOT effective fields
    H_DL(t) = DL_SOT * J(t) and H_FL(t) = FL_SOT * J(t) are
    recomputed every call from `p.pulse(t)`, so the integrator
    follows arbitrary time-varying drives.
    """
    # gp = gamma / (1 + alpha^2) absorbs the explicit LLGS prefactor.
    gp = p.gamma_p
    alpha = p.alpha
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Precession and damping
    # -gp (m x H) is precession;
    m_cross_H = np.cross(m, H_eff)
    # -gp alpha m x (m x H) is Gilbert damping;
    m_cross_m_cross_H = np.cross(m, m_cross_H)
    # Precession + damping contribution to dm/dt.
    dmdt = -gp * (m_cross_H + alpha * m_cross_m_cross_H)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin-orbit torque
    # Time-varying current density evaluated at the substage time t.
    if p.pulse is None:
        raise RuntimeError(
            'llgs_rhs: p.pulse is None. Set the drive explicitly, '
            'e.g. p.pulse = ConstantPulse(0.0) for no current.')
    J_t = p.pulse(t)
    # Skip the SOT branch entirely when J(t) = 0 (relaxation,
    # outside-the-window of a SquarePulse, deep tails of a Gaussian).
    if J_t != 0.0:
        # SOT effective fields at time t (Tesla).
        H_DL_t = p.DL_SOT * J_t
        H_FL_t = p.FL_SOT * J_t
        # Legacy normalized form (nonstandard: torque magnitude lost
        # the sin factor between m and p_hat; kept for reference):
        # # p_hat x m
        # p_cross_m = np.cross(p.p_hat[np.newaxis, np.newaxis, :], m)
        # # |p_hat x m|
        # p_cross_m_norm = np.sqrt(
        #     np.sum(p_cross_m ** 2, axis=-1, keepdims=True,))
        # # Avoid division by zero where p_hat || m
        # safe = np.maximum(p_cross_m_norm, 1e-30)
        # s = p_cross_m / safe
        # s = p_hat x m (unnormalized, standard Slonczewski form):
        # the torque carries the sin factor between m and p_hat and
        # vanishes where m || p_hat.
        s = np.cross(p.p_hat[np.newaxis, np.newaxis, :], m)
        # m x s
        m_cross_s = np.cross(m, s)
        # Effective SOT coefficients (in Tesla)
        # Explicit-form coefficients mix DL and FL via alpha.
        c_dl = H_DL_t + alpha * H_FL_t
        c_fl = H_FL_t - alpha * H_DL_t
        dmdt += gp * (c_dl * m_cross_s + c_fl * s)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Topological spin Hall torque
    # Off by default (p.lambda_sq = 0); the entire block is skipped.
    if p.lambda_sq != 0.0:
        # The TSH gradient stencil wraps periodically and would
        # read vacuum spins; refuse loudly instead of computing
        # silently wrong gradients at the mask boundary.
        if mask is not None:
            raise RuntimeError(
                'llgs_rhs: the TSH torque (lambda_sq != 0) does '
                'not support a free-boundary mask; its gradient '
                'stencil would read vacuum neighbors.')
        # Read the substage current density once.
        J_t_tsh = p.pulse(t)
        # If the pulse is zero there is no TSH contribution either.
        if J_t_tsh != 0.0:
            # b_j = (mu_B/q_e) * J(t) * P / Ms (units: m/s).
            b_j = p.mu_B_over_q_e * J_t_tsh * p.P / p.Ms
            # Lattice constant in metres (for the central differences).
            a = p.a
            # +x neighbour via roll (gives m at site (i, j+1)).
            m_px = np.roll(m, -1, axis=1)
            # -x neighbour via roll (gives m at site (i, j-1)).
            m_mx = np.roll(m, +1, axis=1)
            # +y neighbour via roll (gives m at site (i+1, j)).
            m_py = np.roll(m, -1, axis=0)
            # -y neighbour via roll (gives m at site (i-1, j)).
            m_my = np.roll(m, +1, axis=0)
            # Central-difference gradients (shape (ny, nx, 3)).
            dmdx = (m_px - m_mx) / (2.0 * a)
            # Same along y.
            dmdy = (m_py - m_my) / (2.0 * a)
            # Topological charge density per area: N_xy = m . (dx m x dy m).
            # Shape (ny, nx) scalar field with units 1/m^2.
            N_xy = np.sum(m * np.cross(dmdx, dmdy), axis=-1)
            # TSH contribution: -b_j * lambda_sq * N_xy * (dm/dy).
            # The newaxis broadcasts the scalar N_xy onto the
            # 3-vector dm/dy without an explicit loop.
            dmdt += (-b_j * p.lambda_sq * N_xy[..., np.newaxis] * dmdy)
    # Vacuum sites must not move (SOT/TSH do not depend on H_eff).
    if mask is not None:
        dmdt = dmdt * mask[..., np.newaxis]
    return dmdt


# ---------------------------------------------------------------------
def rhs_local_keff(m_top, m_bot, p, t):
    """Compute RHS for both layers using the local-K_eff field.

    This is the legacy effective-field path: anisotropy uses
    K_eff (with the uniform slab demag folded in) and there is
    no explicit FFT demag convolution. Pair with `rk4_step` to
    reproduce the original simulator behaviour.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulation parameters.
    t : float
        Time in seconds, passed through to `llgs_rhs`.

    Returns
    -------
    dmdt_top : numpy.ndarray(3d)
        Time derivative of top layer.
    dmdt_bot : numpy.ndarray(3d)
        Time derivative of bottom layer.
    """
    # Uses C_anis with K_eff convention (uniform demag folded in).
    # Top
    H_top = effective_field(
        m_top, m_bot,
        p.C_ex, p.C_dmi, p.C_anis_top,
        p.H_ext, p.H_RKKY,)
    # Bottom
    H_bot = effective_field(
        m_bot, m_top,
        p.C_ex, p.C_dmi, p.C_anis_bot,
        p.H_ext, p.H_RKKY,)
    # Compute explicit dm/dt for both layers.
    dmdt_top = llgs_rhs(m_top, H_top, p, t)
    dmdt_bot = llgs_rhs(m_bot, H_bot, p, t)
    # Return
    return dmdt_top, dmdt_bot


# ---------------------------------------------------------------------
def rhs_demag(kernels):
    """Factory: closure that returns dm/dt using the explicit
    FFT demag field. The returned callable matches the
    `_rhs_pair(m_top, m_bot, p, t)` protocol expected by
    `rk4_step`.

    Anisotropy is taken in the bare-K convention (paired with
    explicit demag), matching
    `skyrmion_simulator.simulator.energy.total_energy`.

    Parameters
    ----------
    kernels : dict
        Demag kernels from
        `simulator.demag.precompute_demag_kernels(p)`.

    Returns
    -------
    rhs : callable
        `rhs(m_top, m_bot, p, t) -> (dmdt_top, dmdt_bot)`.
    """
    def rhs(m_top, m_bot, p, t):
        H_top, H_bot = effective_field_demag_pair(
            m_top, m_bot, p, kernels)
        return (llgs_rhs(m_top, H_top, p, t),
                llgs_rhs(m_bot, H_bot, p, t))
    return rhs


# ---------------------------------------------------------------------
def rk4_step(_rhs_pair, m_top, m_bot, t, dt, p):
    """Advance both layers by one RK4 step using `_rhs_pair`.

    Parameters
    ----------
    _rhs_pair : callable
        `_rhs_pair(m_top, m_bot, p, t) -> (dmdt_top, dmdt_bot)`.
        Encapsulates the field model (e.g. `rhs_local_keff`
        for the slab/K_eff path or `rhs_demag(kernels)` for
        explicit FFT demag). Underscore prefix marks the slot
        as factory-supplied, not for direct user calls.
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
    t : float
        Time in seconds at the start of the step. The pulse is
        evaluated at substage times t, t+dt/2, t+dt/2, t+dt.
    dt : float
        Time step in seconds.
    p : SimpleNamespace
        Simulation parameters.

    Returns
    -------
    m_top_new : numpy.ndarray(3d)
        Updated top layer spins.
    m_bot_new : numpy.ndarray(3d)
        Updated bottom layer spins.

    Notes
    -----
    Intermediate states are renormalized to enforce
    the constraint |m| = 1 at each substep. The pulse is evaluated
    at the classical RK4 substage times so the integrator stays
    O(dt^5) locally even when J(t) varies inside one step.
    """
    # k1
    # Slope at the start of the interval (substage time = t).
    k1t, k1b = _rhs_pair(m_top, m_bot, p, t)
    k1t *= dt
    k1b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k2
    # Slope at midpoint using half-step k1; normalize keeps |m|=1.
    mt2 = normalize(m_top + 0.5 * k1t)
    mb2 = normalize(m_bot + 0.5 * k1b)
    k2t, k2b = _rhs_pair(mt2, mb2, p, t + 0.5 * dt)
    k2t *= dt
    k2b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k3
    # Midpoint slope using k2 (improves coupling between substages).
    mt3 = normalize(m_top + 0.5 * k2t)
    mb3 = normalize(m_bot + 0.5 * k2b)
    k3t, k3b = _rhs_pair(mt3, mb3, p, t + 0.5 * dt)
    k3t *= dt
    k3b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k4
    # End-of-interval slope using full-step k3 (substage time = t + dt).
    mt4 = normalize(m_top + k3t)
    mb4 = normalize(m_bot + k3b)
    k4t, k4b = _rhs_pair(mt4, mb4, p, t + dt)
    k4t *= dt
    k4b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combine
    # Simpson-weighted average; final normalize keeps |m|=1 to O(dt^5).
    # Top
    m_top_new = normalize(m_top + (k1t + 2.0 * k2t + 2.0 * k3t + k4t) / 6.0)
    # Bottom
    m_bot_new = normalize(m_bot + (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0)
    # Return
    return m_top_new, m_bot_new


# ---------------------------------------------------------------------
def rk4_step_single(rhs_single, m, t, dt, p):
    """Advance a SINGLE magnetic layer by one RK4 step.

    Same 4th-order Runge-Kutta scheme as `rk4_step`, but for a
    lone ferromagnet (no second layer, no interlayer coupling).
    Single-FM benchmarks (e.g. muMAG SP4 / SP5) use this so
    they exercise the production integrator instead of an
    ad-hoc per-test loop; the two-layer `rk4_step` cannot be
    used directly because its second-layer `normalize` rejects
    the zero/absent bottom layer.

    Parameters
    ----------
    rhs_single : callable
        `rhs_single(m, p, t) -> dmdt`, the single-layer LLGS
        right-hand side (encapsulates the field model).
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    t : float
        Current time (s).
    dt : float
        Time step (s).
    p : SimpleNamespace
        Simulation parameters.

    Returns
    -------
    m_new : numpy.ndarray(3d)
        Updated spins, shape (ny, nx, 3), renormalised to
        |m| = 1.
    """
    # k1: slope at the interval start.
    k1 = dt * rhs_single(m, p, t)
    # k2, k3: midpoint slopes (half-step), normalised to stay
    # on the unit sphere.
    k2 = dt * rhs_single(normalize(m + 0.5 * k1), p, t + 0.5 * dt)
    k3 = dt * rhs_single(normalize(m + 0.5 * k2), p, t + 0.5 * dt)
    # k4: end-of-interval slope (full step).
    k4 = dt * rhs_single(normalize(m + k3), p, t + dt)
    # Simpson-weighted combination; final normalize keeps
    # |m| = 1 to O(dt^5).
    return normalize(m + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0)


# ---------------------------------------------------------------------
def zhang_li_torque(m, a, u_T, beta, alpha):
    """Zhang-Li adiabatic + non-adiabatic spin-transfer torque.

    Contribution to dm/dt from an in-plane spin-polarised
    current with spin-drift velocity u = u_T x_hat, in the
    explicit Landau form (Thiaville et al., EPL 69, 990 (2005)):

        dm/dt += 1/(1+a^2) [ -(1 + a b)(u . grad) m
                             + (b - a) m x ((u . grad) m) ]

    where a = alpha (Gilbert damping), b = beta (non-
    adiabaticity), and (u . grad) m = u_T d m / dx is evaluated
    by central differences (np.gradient) along x. The
    1/(1+a^2) prefactor matches `llgs_rhs`'s `gamma_p`, so the
    return value is added directly to the `llgs_rhs` output.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant (m); the finite-difference spacing.
    u_T : float
        Spin-drift speed along +x (m/s).
    beta : float
        Non-adiabaticity parameter xi.
    alpha : float
        Gilbert damping.

    Returns
    -------
    dmdt_stt : numpy.ndarray(3d)
        STT contribution to dm/dt, shape (ny, nx, 3).
    """
    # (u . grad) m = u_T d m / dx; central differences (one-
    # sided at the array edges via np.gradient).
    dm_dx = np.gradient(m, a, axis=1)
    conv = u_T * dm_dx
    inv = 1.0 / (1.0 + alpha * alpha)
    return inv * (-(1.0 + alpha * beta) * conv
                  + (beta - alpha) * np.cross(m, conv))
