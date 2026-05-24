"""Convergence-driven relaxation for SAF micromagnetics.

Implements an RK4 LLGS integrator that evaluates the
demag-aware effective field on each substep and stops when
both the maximum tangential torque and the relative energy
drift fall below user-set tolerances. Spin-orbit torques
are forcibly disabled inside the loop (J=0 relaxation) and
restored on exit, so the routine never silently drives a
finite current.

Functions
---------
relax
    Relax a SAF state to its (meta)stable equilibrium.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.simulator.energy import total_energy
from src.simulator.fields import effective_field_demag_pair
from src.simulator.integrator import llgs_rhs, normalize
from src.simulator.pulses import ConstantPulse

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def _rhs_pair(m_top, m_bot, p, kernels, t):
    """Compute LLGS RHS for both layers using demag-aware H."""
    H_t, H_b = effective_field_demag_pair(m_top, m_bot, p, kernels)
    return (
        llgs_rhs(m_top, H_t, p, t),
        llgs_rhs(m_bot, H_b, p, t),
    )


# ---------------------------------------------------------------------
def _rk4_step(m_top, m_bot, t, dt, p, kernels):
    """Advance both layers by one demag-aware RK4 step.

    The pulse is evaluated at the classical RK4 substage times.
    For the relaxation loop the active pulse is always
    ConstantPulse(0), so t passes through harmlessly; carrying it
    keeps the signature consistent with the deterministic
    simulator and makes future driven-relaxation variants
    straightforward.
    """
    k1t, k1b = _rhs_pair(m_top, m_bot, p, kernels, t)
    k1t *= dt
    k1b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt2 = normalize(m_top + 0.5 * k1t)
    mb2 = normalize(m_bot + 0.5 * k1b)
    k2t, k2b = _rhs_pair(mt2, mb2, p, kernels, t + 0.5 * dt)
    k2t *= dt
    k2b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt3 = normalize(m_top + 0.5 * k2t)
    mb3 = normalize(m_bot + 0.5 * k2b)
    k3t, k3b = _rhs_pair(mt3, mb3, p, kernels, t + 0.5 * dt)
    k3t *= dt
    k3b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt4 = normalize(m_top + k3t)
    mb4 = normalize(m_bot + k3b)
    k4t, k4b = _rhs_pair(mt4, mb4, p, kernels, t + dt)
    k4t *= dt
    k4b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top_new = normalize(
        m_top + (k1t + 2.0 * k2t + 2.0 * k3t + k4t) / 6.0
    )
    m_bot_new = normalize(
        m_bot + (k1b + 2.0 * k2b + 2.0 * k3b + k4b) / 6.0
    )
    return m_top_new, m_bot_new


# ---------------------------------------------------------------------
def _max_tangential_torque(m_top, m_bot, p, kernels):
    """Return max |m x (m x H)| across both layers (Tesla)."""
    H_t, H_b = effective_field_demag_pair(m_top, m_bot, p, kernels)
    tau_t = np.cross(m_top, np.cross(m_top, H_t))
    tau_b = np.cross(m_bot, np.cross(m_bot, H_b))
    n_t = np.sqrt(np.sum(tau_t * tau_t, axis=-1))
    n_b = np.sqrt(np.sum(tau_b * tau_b, axis=-1))
    return float(max(n_t.max(), n_b.max()))


# ---------------------------------------------------------------------
def relax(m_top, m_bot, p, kernels,
          max_steps=200000, alpha_relax=None,
          tol_torque=1e-5, tol_dE=1e-8, check_every=1000,
          print_every=0):
    """Relax a SAF spin configuration to (meta)stable state.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameters namespace. SOT fields and Gilbert
        damping are temporarily overridden inside this
        function and restored on exit.
    kernels : dict
        Demag kernels from `precompute_demag_kernels(p)`.
    max_steps : int, default=200000
        Safety cutoff on the number of RK4 steps.
    alpha_relax : float or None, default=None
        If given, override `p.alpha` (and the dependent
        `p.gamma_p`) for the relaxation. A common choice
        is alpha=1.0 for over-damped quench.
    tol_torque : float, default=1e-5
        Convergence threshold on max|m x (m x H_eff)| in
        Tesla.
    tol_dE : float, default=1e-8
        Convergence threshold on relative energy change
        per `check_every`-step window.
    check_every : int, default=1000
        Stride of convergence checks.
    print_every : int, default=0
        Stride of progress prints. Pass 0 to disable. Each
        print reports the step number, the elapsed simulated
        time (ps), and the most recently computed `tau_max`
        in Tesla.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Final top-layer spins.
    m_bot : numpy.ndarray(3d)
        Final bottom-layer spins.
    converged : bool
        True if both convergence criteria were satisfied
        before the safety cutoff.
    n_steps : int
        Number of RK4 steps actually executed.
    E_final : float
        Final total energy in Joules.
    tau_max : float
        Final maximum tangential torque in Tesla.
    """
    if not hasattr(p, 'H_DL') or not hasattr(p, 'H_FL'):
        raise RuntimeError(
            'Parameters namespace must expose `H_DL` and '
            '`H_FL` (precomputed by `parameters._precompute` '
            'or `make_params`). Got an incomplete `p`.'
        )
    alpha_save = p.alpha
    gamma_p_save = p.gamma_p
    H_DL_save = p.H_DL
    H_FL_save = p.H_FL
    pulse_save = getattr(p, 'pulse', None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Disable SOT and apply optional damping override
    # Both p.H_DL/p.H_FL and p.pulse are zeroed: the pulse is the
    # value llgs_rhs actually reads; the scalars are kept consistent
    # for header prints and any external diagnostic.
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.pulse = ConstantPulse(0.0)
    if alpha_relax is not None:
        p.alpha = float(alpha_relax)
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    converged = False
    n_steps = 0
    tau_max = np.inf
    try:
        E_prev = total_energy(m_top, m_bot, p, kernels)
        # Internal pulse time; the pulse is zero so the exact value
        # does not affect dynamics, only kept consistent with the
        # pulse-aware integrator signature.
        t = 0.0
        for step in range(1, max_steps + 1):
            m_top, m_bot = _rk4_step(
                m_top, m_bot, t, p.dt, p, kernels,
            )
            t += p.dt
            n_steps = step
            if print_every > 0 and step % print_every == 0:
                print(
                    f'    relax step {step}/{max_steps} '
                    f'({step*p.dt*1e12:.1f} ps), '
                    f'tau_max={tau_max:.2e} T',
                    flush=True,
                )
            if step % check_every == 0:
                tau_max = _max_tangential_torque(
                    m_top, m_bot, p, kernels,
                )
                E_now = total_energy(m_top, m_bot, p, kernels)
                if E_now != 0.0:
                    dE_rel = abs((E_now - E_prev) / E_now)
                else:
                    dE_rel = abs(E_now - E_prev)
                E_prev = E_now
                if tau_max < tol_torque and dE_rel < tol_dE:
                    converged = True
                    break
        E_final = total_energy(m_top, m_bot, p, kernels)
    finally:
        p.alpha = alpha_save
        p.gamma_p = gamma_p_save
        p.H_DL = H_DL_save
        p.H_FL = H_FL_save
        if pulse_save is not None:
            p.pulse = pulse_save
    return m_top, m_bot, converged, n_steps, E_final, tau_max
