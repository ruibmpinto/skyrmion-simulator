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
rk4_step
    Advance both layers by one RK4 step.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.simulator.fields import effective_field

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
    return m / norm[..., np.newaxis]


# ---------------------------------------------------------------------
def llgs_rhs(m, H_eff, p, t):
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

    where s = (p_hat x m) / |p_hat x m|. The SOT effective fields
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
    J_t = p.pulse(t)
    # Skip the SOT branch entirely when J(t) = 0 (relaxation,
    # outside-the-window of a SquarePulse, deep tails of a Gaussian).
    if J_t != 0.0:
        # SOT effective fields at time t (Tesla).
        H_DL_t = p.DL_SOT * J_t
        H_FL_t = p.FL_SOT * J_t
        # p_hat x m
        p_cross_m = np.cross(p.p_hat[np.newaxis, np.newaxis, :], m)
        # |p_hat x m|
        p_cross_m_norm = np.sqrt(
            np.sum(p_cross_m ** 2, axis=-1, keepdims=True,))
        # Avoid division by zero where p_hat || m
        # Floor prevents NaN when m aligns with the polarization axis.
        safe = np.maximum(p_cross_m_norm, 1e-30)
        # s = (p_hat x m) / |p_hat x m| is the SOT spin polarization direction.
        s = p_cross_m / safe
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
    return dmdt


# ---------------------------------------------------------------------
def _compute_rhs_both(m_top, m_bot, p, t):
    """Compute RHS for both layers simultaneously.

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
def rk4_step(m_top, m_bot, t, dt, p):
    """Advance both layers by one RK4 step.

    Parameters
    ----------
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
    k1t, k1b = _compute_rhs_both(m_top, m_bot, p, t)
    k1t *= dt
    k1b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k2
    # Slope at midpoint using half-step k1; normalize keeps |m|=1.
    mt2 = normalize(m_top + 0.5 * k1t)
    mb2 = normalize(m_bot + 0.5 * k1b)
    k2t, k2b = _compute_rhs_both(mt2, mb2, p, t + 0.5 * dt)
    k2t *= dt
    k2b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k3
    # Midpoint slope using k2 (improves coupling between substages).
    mt3 = normalize(m_top + 0.5 * k2t)
    mb3 = normalize(m_bot + 0.5 * k2b)
    k3t, k3b = _compute_rhs_both(mt3, mb3, p, t + 0.5 * dt)
    k3t *= dt
    k3b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k4
    # End-of-interval slope using full-step k3 (substage time = t + dt).
    mt4 = normalize(m_top + k3t)
    mb4 = normalize(m_bot + k3b)
    k4t, k4b = _compute_rhs_both(mt4, mb4, p, t + dt)
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
