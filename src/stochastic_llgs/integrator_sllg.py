"""Stratonovich-Heun stepper for the stochastic SAF LLGS.

A single-step predictor-corrector that integrates Brown's
stochastic LLGS in the Stratonovich interpretation. The
deterministic right-hand side is reused unchanged from
`src.simulator.integrator.llgs_rhs`; the effective field is
assembled by `src.simulator.fields.effective_field` (no demag)
or `src.simulator.fields.effective_field_demag_pair`
(with demag). The thermal noise is supplied by the caller as
pre-sampled arrays and is added to the assembled field
immediately before each `llgs_rhs` call -- never inside the
field-assembly routines.

The same noise sample is used in both the predictor and
corrector stages; this is what makes the discrete scheme
converge to the Stratonovich SDE rather than the Ito one
(Greenside--Helfand 1981, Garcia-Palacios & Lazaro 1998). No
explicit Stratonovich drift-correction term is added: the Heun
predictor-corrector arithmetic captures it automatically. The
predictor magnetization is NOT renormalized (matches PysLLG
and the original derivation); a single end-of-step
renormalization restores `|m| = 1` after a norm-drift check.

Functions
---------
heun_stochastic_step
    Advance both SAF layers by one stochastic Heun step.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from src.simulator.fields import (
    effective_field,
    effective_field_demag_pair,
)
from src.simulator.integrator import llgs_rhs

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _rhs_with_noise(m_top, m_bot, h_top, h_bot, p, kernels, t):
    """Assemble H_eff (noise-free), add thermal noise, return
    dm/dt for both layers via the existing llgs_rhs.

    Notes
    -----
    The noise is added to the already-assembled H_eff tensor;
    it never enters `effective_field` or
    `effective_field_demag_pair`. This guarantees that the
    exchange Laplacian and DMI difference operators act only
    on the magnetization, preserving the white-in-space
    structure of the Brown noise.

    The substage time `t` is forwarded to `llgs_rhs` so that the
    SOT terms read `p.pulse(t)` consistently with the
    deterministic integrator.
    """
    if kernels is None:
        H_top = effective_field(
            m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
            p.H_ext, p.H_RKKY,
        )
        H_bot = effective_field(
            m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
            p.H_ext, p.H_RKKY,
        )
    else:
        H_top, H_bot = effective_field_demag_pair(
            m_top, m_bot, p, kernels,
        )
    H_top_total = H_top + h_top
    H_bot_total = H_bot + h_bot
    dmdt_top = llgs_rhs(m_top, H_top_total, p, t)
    dmdt_bot = llgs_rhs(m_bot, H_bot_total, p, t)
    return dmdt_top, dmdt_bot


# -----------------------------------------------------------------------------
def heun_stochastic_step(m_top, m_bot, dt, p, kernels,
                         h_top, h_bot, tol_norm, *, t=0.0):
    """One Stratonovich-Heun step of the stochastic SAF LLGS.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3), `|m| = 1` per
        cell.
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    dt : float
        Time step in seconds. Strictly positive.
    p : SimpleNamespace
        Parameters namespace (must expose at least `alpha,
        gamma_p, C_ex, C_dmi, C_anis_top, C_anis_bot, H_ext,
        H_RKKY, H_DL, H_FL, p_hat`).
    kernels : dict or None
        Demag kernels from
        `src.simulator.demag.precompute_demag_kernels(p)`, or
        `None` to bypass demag entirely. The caller must pass
        one or the other explicitly.
    h_top : numpy.ndarray(3d)
        Pre-sampled thermal-noise field on the top layer,
        shape (ny, nx, 3), in Tesla (instantaneous amplitude,
        already scaled by `1/sqrt(dt)` upstream).
    h_bot : numpy.ndarray(3d)
        Pre-sampled thermal-noise field on the bottom layer.
        Independent from `h_top`.
    tol_norm : float
        Maximum permitted `||m|| - 1` deviation before
        end-of-step renormalization. Strictly positive. If
        exceeded, a `RuntimeError` is raised so a too-coarse
        `dt` or too-large noise is surfaced loudly rather
        than masked by the renormalization.
    t : float, default=0.0
        Time in seconds at the start of the step (keyword-only).
        Forwarded to `llgs_rhs` so the SOT term reads
        `p.pulse(t)`. Callers with constant drives can leave
        this at the default; callers with time-varying pulses
        should pass `t = step * dt` explicitly.

    Returns
    -------
    m_top_new : numpy.ndarray(3d)
        Updated top-layer spins, renormalized to `|m| = 1`.
    m_bot_new : numpy.ndarray(3d)
        Updated bottom-layer spins, renormalized.
    norm_drift : float
        `max(||m_new|| - 1)` across both layers before
        renormalization; recorded for diagnostics.

    Notes
    -----
    The scheme:

        Predictor (no renorm):
            f1 = LLGS(m,   H_eff(m)   + h)
            m~ = m + dt * f1
        Corrector (same noise h):
            f2 = LLGS(m~, H_eff(m~) + h)
            m_new = m + 0.5 * dt * (f1 + f2)
        End-of-step:
            check max(||m_new|| - 1) <= tol_norm
            renormalize.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate inputs
    if m_top.shape != m_bot.shape:
        raise RuntimeError(
            f'heun_stochastic_step: m_top.shape '
            f'{m_top.shape} != m_bot.shape {m_bot.shape}.'
        )
    if m_top.ndim != 3 or m_top.shape[-1] != 3:
        raise RuntimeError(
            f'heun_stochastic_step: m_top must have shape '
            f'(ny, nx, 3), got {m_top.shape}.'
        )
    if h_top.shape != m_top.shape:
        raise RuntimeError(
            f'heun_stochastic_step: h_top.shape '
            f'{h_top.shape} != m_top.shape {m_top.shape}.'
        )
    if h_bot.shape != m_bot.shape:
        raise RuntimeError(
            f'heun_stochastic_step: h_bot.shape '
            f'{h_bot.shape} != m_bot.shape {m_bot.shape}.'
        )
    if not (np.isfinite(dt) and dt > 0.0):
        raise RuntimeError(
            f'heun_stochastic_step: dt must be finite and '
            f'strictly positive, got {dt!r}.'
        )
    if not (np.isfinite(tol_norm) and tol_norm > 0.0):
        raise RuntimeError(
            f'heun_stochastic_step: tol_norm must be finite '
            f'and strictly positive, got {tol_norm!r}.'
        )
    if kernels is not None and not isinstance(kernels, dict):
        raise RuntimeError(
            f'heun_stochastic_step: kernels must be None or '
            f'a dict from precompute_demag_kernels, got '
            f'{type(kernels).__name__}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Predictor: noise-free H + thermal h, no renorm
    # Pulse evaluated at the substage time t.
    f1_top, f1_bot = _rhs_with_noise(
        m_top, m_bot, h_top, h_bot, p, kernels, t,
    )
    m_top_pred = m_top + dt * f1_top
    m_bot_pred = m_bot + dt * f1_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Corrector: same noise sample h
    # Pulse evaluated at the end-of-step time t + dt.
    f2_top, f2_bot = _rhs_with_noise(
        m_top_pred, m_bot_pred, h_top, h_bot, p, kernels, t + dt,
    )
    m_top_new = m_top + 0.5 * dt * (f1_top + f2_top)
    m_bot_new = m_bot + 0.5 * dt * (f1_bot + f2_bot)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Norm-drift check (loud) and end-of-step renormalization
    norm_top = np.sqrt(np.sum(m_top_new ** 2, axis=-1))
    norm_bot = np.sqrt(np.sum(m_bot_new ** 2, axis=-1))
    drift = float(
        max(
            np.max(np.abs(norm_top - 1.0)),
            np.max(np.abs(norm_bot - 1.0)),
        )
    )
    if drift > tol_norm:
        raise RuntimeError(
            f'heun_stochastic_step: end-of-step norm drift '
            f'max(||m||-1) = {drift:.3e} exceeds tol_norm = '
            f'{tol_norm:.3e}. Reduce dt or check thermal '
            f'amplitude (sigma_noise / sqrt(dt)).'
        )
    m_top_new = m_top_new / norm_top[..., np.newaxis]
    m_bot_new = m_bot_new / norm_bot[..., np.newaxis]
    return m_top_new, m_bot_new, drift
