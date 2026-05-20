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
def llgs_rhs(m, H_eff, p):
    """Compute dm/dt from the explicit LLGS equation.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    H_eff : numpy.ndarray(3d)
        Effective field, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulation parameters.

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

    where s = (p_hat x m) / |p_hat x m|.
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
    # Skip the SOT branch entirely if both torques are zero (relaxation).
    if p.H_DL != 0.0 or p.H_FL != 0.0:
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
        c_dl = p.H_DL + alpha * p.H_FL
        c_fl = p.H_FL - alpha * p.H_DL
        dmdt += gp * (c_dl * m_cross_s + c_fl * s)
    return dmdt


# ---------------------------------------------------------------------
def _compute_rhs_both(m_top, m_bot, p):
    """Compute RHS for both layers simultaneously.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulation parameters.

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
    dmdt_top = llgs_rhs(m_top, H_top, p)
    dmdt_bot = llgs_rhs(m_bot, H_bot, p)
    # Return
    return dmdt_top, dmdt_bot


# ---------------------------------------------------------------------
def rk4_step(m_top, m_bot, dt, p):
    """Advance both layers by one RK4 step.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
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
    the constraint |m| = 1 at each substep.
    """
    # k1
    # Slope at the start of the interval.
    k1t, k1b = _compute_rhs_both(m_top, m_bot, p)
    k1t *= dt
    k1b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k2
    # Slope at midpoint using half-step k1; normalize keeps |m|=1.
    mt2 = normalize(m_top + 0.5 * k1t)
    mb2 = normalize(m_bot + 0.5 * k1b)
    k2t, k2b = _compute_rhs_both(mt2, mb2, p)
    k2t *= dt
    k2b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k3
    # Midpoint slope using k2 (improves coupling between substages).
    mt3 = normalize(m_top + 0.5 * k2t)
    mb3 = normalize(m_bot + 0.5 * k2b)
    k3t, k3b = _compute_rhs_both(mt3, mb3, p)
    k3t *= dt
    k3b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k4
    # End-of-interval slope using full-step k3.
    mt4 = normalize(m_top + k3t)
    mb4 = normalize(m_bot + k3b)
    k4t, k4b = _compute_rhs_both(mt4, mb4, p)
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
