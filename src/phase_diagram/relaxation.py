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
from src.simulator.fields import (
    effective_field,
    effective_field_demag_pair,
)
from src.simulator.integrator import (
    llgs_rhs,
    normalize,
    rk4_step_single,
)
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


def _pair_fields(m_top, m_bot, p, kernels, mask=None):
    """Effective fields for both layers, demag-aware or K_eff.

    `kernels=None` selects the no-demag local-K_eff assembly
    (uniform thin-film demag folded into `p.C_anis_*`),
    mirroring the C++ `RK4LocalKeffStepper` used by `relax`
    when no demag state is supplied. Otherwise the bare-K +
    explicit-demag pair assembly is used.
    """
    if kernels is None:
        # K_eff convention: no explicit demag, no free-y.
        H_t = effective_field(
            m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
            p.H_ext, p.H_RKKY, mask=mask)
        H_b = effective_field(
            m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
            p.H_ext, p.H_RKKY, mask=mask)
        return H_t, H_b
    return effective_field_demag_pair(
        m_top, m_bot, p, kernels, mask=mask)


# ---------------------------------------------------------------------
def _rhs_pair(m_top, m_bot, p, kernels, t, mask=None):
    """Compute LLGS RHS for both layers using demag-aware H.

    `mask` (if given) is forwarded to the field assembly for
    free-boundary geometries.
    """
    H_t, H_b = _pair_fields(m_top, m_bot, p, kernels, mask=mask)
    return (
        llgs_rhs(m_top, H_t, p, t, mask=mask),
        llgs_rhs(m_bot, H_b, p, t, mask=mask),
    )


# ---------------------------------------------------------------------
def _rhs_single(m, p, t):
    """Single-layer LLGS RHS (no demag), `rk4_step_single` form.

    Matches the 3-argument `rk4_step_single(rhs, m, t, dt, p)`
    callback contract. The free-BC mask is read from
    `p.relax_mask`, which `relax` sets on every call (just like
    the SOT fields it temporarily overrides), so no closure is
    needed; a direct attribute read raises loudly if this
    private helper is ever called outside `relax`. The no-demag
    `effective_field` is used (a lone ferromagnet has no
    interlayer demag); the layer is its own RKKY partner,
    harmless because single-layer relax sets `p.H_RKKY = 0`.
    """
    mask = p.relax_mask
    H = effective_field(
        m, m, p.C_ex, p.C_dmi, p.C_anis_top,
        p.H_ext, p.H_RKKY, mask=mask)
    return llgs_rhs(m, H, p, t, mask=mask)


# ---------------------------------------------------------------------
def _rk4_step(m_top, m_bot, t, dt, p, kernels, mask=None):
    """Advance both layers by one demag-aware RK4 step.

    The pulse is evaluated at the classical RK4 substage times.
    For the relaxation loop the active pulse is always
    ConstantPulse(0), so t passes through harmlessly; carrying it
    keeps the signature consistent with the deterministic
    simulator and makes future driven-relaxation variants
    straightforward.
    """
    k1t, k1b = _rhs_pair(m_top, m_bot, p, kernels, t, mask=mask)
    k1t *= dt
    k1b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt2 = normalize(m_top + 0.5 * k1t)
    mb2 = normalize(m_bot + 0.5 * k1b)
    k2t, k2b = _rhs_pair(
        mt2, mb2, p, kernels, t + 0.5 * dt, mask=mask)
    k2t *= dt
    k2b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt3 = normalize(m_top + 0.5 * k2t)
    mb3 = normalize(m_bot + 0.5 * k2b)
    k3t, k3b = _rhs_pair(
        mt3, mb3, p, kernels, t + 0.5 * dt, mask=mask)
    k3t *= dt
    k3b *= dt
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    mt4 = normalize(m_top + k3t)
    mb4 = normalize(m_bot + k3b)
    k4t, k4b = _rhs_pair(mt4, mb4, p, kernels, t + dt, mask=mask)
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
def _max_tangential_torque(m_top, m_bot, p, kernels, mask=None):
    """Return max |m x (m x H)| across both layers (Tesla)."""
    H_t, H_b = _pair_fields(m_top, m_bot, p, kernels, mask=mask)
    tau_t = np.cross(m_top, np.cross(m_top, H_t))
    tau_b = np.cross(m_bot, np.cross(m_bot, H_b))
    if mask is not None:
        msk = mask[..., np.newaxis]
        tau_t = tau_t * msk
        tau_b = tau_b * msk
    n_t = np.sqrt(np.sum(tau_t * tau_t, axis=-1))
    n_b = np.sqrt(np.sum(tau_b * tau_b, axis=-1))
    return float(max(n_t.max(), n_b.max()))


# ---------------------------------------------------------------------
def _max_tangential_torque_single(m, p, mask=None):
    """Return max |m x (m x H)| for one layer (Tesla, no demag)."""
    H = effective_field(
        m, m, p.C_ex, p.C_dmi, p.C_anis_top,
        p.H_ext, p.H_RKKY, mask=mask)
    tau = np.cross(m, np.cross(m, H))
    if mask is not None:
        tau = tau * mask[..., np.newaxis]
    return float(np.sqrt(np.sum(tau * tau, axis=-1)).max())


# ---------------------------------------------------------------------
def relax(m_top, m_bot, p, kernels,
          max_steps=200000, alpha_relax=None,
          tol_torque=1e-5, tol_dE=1e-8, check_every=1000,
          print_every=0, mask=None):
    """Relax a spin configuration to a (meta)stable state.

    Three modes:

    - SAF pair with demag (default): `m_bot` is an array and
      `kernels` a dict; both layers relax together with
      explicit demag, converging on torque + energy drift.
    - SAF pair, no demag: `m_bot` is an array and
      `kernels=None`; both layers relax on the local-K_eff
      field (mirrors the C++ `RK4LocalKeffStepper` path),
      converging on the tangential torque alone. `E_final`
      is NaN.
    - Single layer: `m_bot=None` relaxes a lone ferromagnet
      via the no-demag `effective_field` + `rk4_step_single`.
      Requires `kernels=None` (a single layer has no interlayer
      demag) and converges on the tangential torque alone (no
      two-layer energy). The returned bottom layer is `None`
      and `E_final` is NaN.

    A free-boundary geometry is selected with `mask` (a Boolean
    array True inside the magnetic region); it is forwarded to
    the field assembly so cells outside the mask receive no
    field and the DMI edge condition is applied.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d) or None
        Bottom-layer spins; `None` selects single-layer mode.
    p : SimpleNamespace
        Parameters namespace. SOT fields and Gilbert
        damping are temporarily overridden inside this
        function and restored on exit.
    kernels : dict or None
        Demag kernels from `precompute_demag_kernels(p)`;
        `None` (required in single-layer mode) disables demag.
    mask : numpy.ndarray(2d, bool) or None, default=None
        Free-boundary magnetic-region mask; `None` is the
        original periodic path.
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
    m_bot : numpy.ndarray(3d) or None
        Final bottom-layer spins; `None` in single-layer mode.
    converged : bool
        True if the convergence criteria were satisfied before
        the safety cutoff (torque + energy in pair mode; torque
        alone in single-layer mode).
    n_steps : int
        Number of RK4 steps actually executed.
    E_final : float
        Final total energy in Joules (NaN in single-layer mode).
    tau_max : float
        Final maximum tangential torque in Tesla.
    """
    if not hasattr(p, 'H_DL') or not hasattr(p, 'H_FL'):
        raise RuntimeError(
            'Parameters namespace must expose `H_DL` and '
            '`H_FL` (precomputed by `parameters._precompute` '
            'or `make_params`). Got an incomplete `p`.'
        )
    single = m_bot is None
    if single and kernels is not None:
        raise RuntimeError(
            'relax: single-layer mode (m_bot is None) requires '
            'kernels=None (a lone ferromagnet has no '
            'interlayer demag).')
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
    # Free-BC mask for the single-layer RHS (read by
    # `_rhs_single` via `p.relax_mask`); set on every call and
    # removed in the `finally` block.
    p.relax_mask = mask
    if alpha_relax is not None:
        # Zero damping cannot relax anything; reject loudly.
        if float(alpha_relax) == 0.0:
            raise RuntimeError(
                'relax: alpha_relax = 0 cannot relax (zero '
                'damping); pass None for no override.')
        p.alpha = float(alpha_relax)
        p.gamma_p = p.gamma / (1.0 + p.alpha * p.alpha)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    converged = False
    n_steps = 0
    tau_max = np.inf
    try:
        # Pair mode with demag tracks the two-layer energy for the
        # dE test; single-layer and no-demag pair modes converge on
        # the torque alone (mirrors the C++ relax semantics).
        use_energy = (not single) and (kernels is not None)
        E_prev = (total_energy(m_top, m_bot, p, kernels, mask=mask)
                  if use_energy else None)
        # Internal pulse time; the pulse is zero so the exact value
        # does not affect dynamics, only kept consistent with the
        # pulse-aware integrator signature.
        t = 0.0
        for step in range(1, max_steps + 1):
            if single:
                m_top = rk4_step_single(
                    _rhs_single, m_top, t, p.dt, p)
            else:
                m_top, m_bot = _rk4_step(
                    m_top, m_bot, t, p.dt, p, kernels, mask=mask)
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
                if single:
                    tau_max = _max_tangential_torque_single(
                        m_top, p, mask=mask)
                    if tau_max < tol_torque:
                        converged = True
                        break
                else:
                    tau_max = _max_tangential_torque(
                        m_top, m_bot, p, kernels, mask=mask)
                    torque_ok = tau_max < tol_torque
                    energy_ok = True
                    if use_energy:
                        E_now = total_energy(
                            m_top, m_bot, p, kernels, mask=mask)
                        dE_rel = (abs((E_now - E_prev) / E_now)
                                  if E_now != 0.0
                                  else abs(E_now - E_prev))
                        E_prev = E_now
                        energy_ok = dE_rel < tol_dE
                    if torque_ok and energy_ok:
                        converged = True
                        break
        if use_energy:
            E_final = total_energy(m_top, m_bot, p, kernels, mask=mask)
        else:
            E_final = float('nan')
    finally:
        p.alpha = alpha_save
        p.gamma_p = gamma_p_save
        p.H_DL = H_DL_save
        p.H_FL = H_FL_save
        if pulse_save is not None:
            p.pulse = pulse_save
        # Remove the transient mask attribute (always set above).
        del p.relax_mask
    return m_top, m_bot, converged, n_steps, E_final, tau_max
