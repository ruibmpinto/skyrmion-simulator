"""Integrator-step factories for the sweep driver.

Each factory returns a callable matching the protocol

    step(m_top, m_bot, t, dt, p) -> (m_top_new, m_bot_new),

so `driver.run_one` can advance one substep without knowing
whether the integrator underneath is deterministic RK4,
demag-aware RK4, or stochastic Heun. Sweep scripts pick the
factory they want and pass its return value to `run_one`.

Functions
---------
step_deterministic
    Plain RK4 with local-K_eff anisotropy (no demag, no
    noise). Wraps `simulator.integrator.rk4_step`.
step_demag_deterministic
    RK4 with explicit FFT demag. Closure over the precomputed
    demag kernels.
step_stochastic
    Stratonovich-Heun stepper with thermal noise sampled per
    call. Optional FFT demag via `kernels`; pass `None` for
    no demag.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from src.simulator.integrator import (
    rhs_demag, rhs_local_keff, rk4_step)
from src.stochastic_llgs.integrator_sllg import heun_stochastic_step
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


def step_deterministic():
    """Return a plain deterministic RK4 stepper with local-K_eff
    anisotropy (no explicit FFT demag).

    Returns
    -------
    step : callable
        `step(m_top, m_bot, t, dt, p) -> (m_top, m_bot)`.
    """
    return lambda m_top, m_bot, t, dt, p: rk4_step(
        rhs_local_keff, m_top, m_bot, t, dt, p)


# -----------------------------------------------------------------------------
def step_demag_deterministic(kernels):
    """Return a demag-aware deterministic RK4 stepper.

    Closes over precomputed FFT demag kernels so the per-step
    field evaluation includes the long-range magnetostatic
    contribution. Anisotropy is taken in the bare-K convention
    (paired with explicit demag), matching the convention in
    `src.simulator.energy.total_energy`.

    Parameters
    ----------
    kernels : dict
        Demag kernels from
        `simulator.demag.precompute_demag_kernels(p)`. Required;
        no `None` fallback — callers that do not want demag
        should use `step_deterministic()` instead.

    Returns
    -------
    step : callable
        `step(m_top, m_bot, t, dt, p) -> (m_top, m_bot)`.
    """
    # Loud rejection of None so a missing kernels argument is
    # surfaced at factory time rather than at first call.
    if kernels is None or not isinstance(kernels, dict):
        raise RuntimeError(
            'step_demag_deterministic: `kernels` must be a dict '
            f'from precompute_demag_kernels, got {type(kernels).__name__}.')
    rhs = rhs_demag(kernels)
    return lambda m_top, m_bot, t, dt, p: rk4_step(
        rhs, m_top, m_bot, t, dt, p)


# -----------------------------------------------------------------------------
def step_stochastic(rng, sigma, tol_norm, kernels):
    """Return a Stratonovich-Heun stochastic stepper.

    Samples an independent thermal field per layer per call,
    scaled by `sigma / sqrt(dt)`, and dispatches to
    `stochastic_llgs.integrator_sllg.heun_stochastic_step`.

    Parameters
    ----------
    rng : numpy.random.Generator
        Source of randomness; the closure pulls noise samples
        on every call without re-seeding.
    sigma : float
        Thermal field amplitude (Tesla * sqrt(s)) as produced
        by `stochastic_llgs.parameters_thermal`. Must be
        strictly positive: `sample_thermal_field` rejects a
        zero amplitude, so the T = 0 deterministic limit is not
        available through this stepper -- use
        `step_demag_deterministic` for that.
    tol_norm : float
        Allowed |m| drift per step; passed straight to
        `heun_stochastic_step`.
    kernels : dict or None
        Demag kernels for FFT demag, or `None` to bypass demag.
        Must be supplied explicitly; no default.

    Returns
    -------
    step : callable
        `step(m_top, m_bot, t, dt, p) -> (m_top, m_bot)`.
    """
    # Validate arguments loudly so silent errors do not creep
    # into long ensemble runs.
    if rng is None:
        raise RuntimeError(
            'step_stochastic: `rng` is required '
            '(e.g. numpy.random.default_rng(seed)).')
    if not np.isfinite(sigma) or sigma < 0.0:
        raise RuntimeError(
            f'step_stochastic: `sigma` must be a non-negative '
            f'finite float, got {sigma!r}.')
    if not np.isfinite(tol_norm) or tol_norm <= 0.0:
        raise RuntimeError(
            f'step_stochastic: `tol_norm` must be a positive '
            f'finite float, got {tol_norm!r}.')
    if kernels is not None and not isinstance(kernels, dict):
        raise RuntimeError(
            'step_stochastic: `kernels` must be None or a dict '
            f'from precompute_demag_kernels, got '
            f'{type(kernels).__name__}.')

    def step(m_top, m_bot, t, dt, p):
        # Sample two independent thermal fields, one per layer.
        # sample_thermal_field takes the lattice shape (ny, nx) and
        # returns (ny, nx, 3); passing the full field shape raises.
        h_top = sample_thermal_field(rng, m_top.shape[:2], sigma, dt)
        h_bot = sample_thermal_field(rng, m_bot.shape[:2], sigma, dt)
        # Dispatch to the Heun stepper; discard the norm-drift
        # diagnostic since the driver does not track it here.
        m_top_new, m_bot_new, _drift = heun_stochastic_step(
            m_top, m_bot, dt, p, kernels,
            h_top, h_bot, tol_norm, t=t,)
        return m_top_new, m_bot_new
    return step
