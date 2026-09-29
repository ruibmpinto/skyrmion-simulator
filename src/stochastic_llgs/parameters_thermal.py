"""Thermal-noise parameters for the stochastic LLGS solver.

Attaches temperature `T`, thermal resistance `R_th`, random
seed, per-cell volume `V_cell`, Boltzmann constant `k_B`, and
the noise standard deviation `sigma_noise` (in Tesla * sqrt(s))
to an already-built parameters namespace. The caller supplies a
fresh `p` from `default_params()` or `make_params(...)`; this
module mutates it in place.

Functions
---------
attach_thermal
    Validate and attach thermal-noise fields to `p`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def attach_thermal(p, T, R_th, seed):
    """Attach thermal-noise fields to an existing parameters namespace.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace built upstream by
        `default_params()` or
        `src.phase_diagram.params_helper.make_params`. Must
        already expose `alpha, gamma, Ms, a, t_Co`. Mutated in
        place: `T, R_th, seed, V_cell, k_B, sigma_noise` are
        added.
    T : float
        Substrate (bath) temperature in Kelvin. Non-negative.
        `T = 0` is allowed and yields `sigma_noise = 0`,
        making the stochastic stepper degenerate to a
        deterministic predictor-corrector (this is the
        T = 0 reproducibility gate, Phase 3).
    R_th : float
        Joule thermal resistance in K * m^4 / A^2.
        Non-negative; `R_th = 0` disables self-heating.
    seed : int
        Master integer seed for noise generation. Each
        trajectory derives its own RNG from this value.

    Returns
    -------
    p : SimpleNamespace
        Same object passed in; mutated and returned for convenience.

    Notes
    -----
    The per-cell thermal-field standard deviation is

        sigma_noise = sqrt(2 * alpha * k_B * T / (gamma * Ms * V_cell))

    with units of Tesla * sqrt(s); the instantaneous Gaussian
    field has std `sigma_noise / sqrt(dt)`. The integration of
    the noise increment over a time step `dt` thus has variance
    `sigma_noise**2 * dt` (in Tesla**2 * s).
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate required attributes on p
    for attr in ('alpha', 'gamma', 'Ms', 'a', 't_Co'):
        if not hasattr(p, attr):
            raise RuntimeError(
                f'attach_thermal: p is missing required attribute {attr!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate T
    if not np.isfinite(T):
        raise RuntimeError(f'attach_thermal: T must be finite, got {T!r}.')
    if T < 0.0:
        raise RuntimeError(
            f'attach_thermal: T must be non-negative '
            f'(T = 0 is allowed; gives the deterministic limit), got {T!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate R_th
    if not np.isfinite(R_th):
        raise RuntimeError(
            f'attach_thermal: R_th must be finite, got {R_th!r}.')
    if R_th < 0.0:
        raise RuntimeError(
            f'attach_thermal: R_th must be non-negative, got {R_th!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Validate seed
    if not isinstance(seed, (int, np.integer)):
        raise RuntimeError(
            f'attach_thermal: seed must be an integer, got '
            f'{type(seed).__name__}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Attach values
    # Bath temperature in Kelvin.
    p.T = float(T)
    # Thermal resistance for Joule self-heating (K m^4 / A^2).
    p.R_th = float(R_th)
    # Master integer seed; per-trajectory RNGs derive from this.
    p.seed = int(seed)
    # Boltzmann constant (J/K), SI 2019 redefinition exact value.
    p.k_B = 1.380649e-23
    # Per-cell volume V_cell = a^2 * t_Co, used by the
    # Fluctuation–Dissipation Theorem (FDT) amplitude.
    p.V_cell = float(p.a) * float(p.a) * float(p.t_Co)
    # Guard: a or t_Co set to 0/negative would silently zero sigma.
    if p.V_cell <= 0.0:
        raise RuntimeError(
            f'attach_thermal: V_cell = a**2 * t_Co must be '
            f'positive, got {p.V_cell!r}.')
    # Fluctuation–Dissipation Theorem (FDT) variance:
    # sigma^2 = 2 alpha k_B T / (gamma Ms V_cell).
    sigma2 = (
        2.0 * float(p.alpha) * p.k_B * p.T
        / (float(p.gamma) * float(p.Ms) * p.V_cell))
    # Should never trigger after the positivity checks above;
    # kept as a defensive guard against future parameter changes.
    if sigma2 < 0.0:
        raise RuntimeError(
            f'attach_thermal: computed sigma_noise**2 is '
            f'negative ({sigma2!r}); check alpha, gamma, Ms, a, t_Co.')
    # Noise amplitude in Tesla * sqrt(s); instantaneous field
    # std passed to the sampler is sigma_noise / sqrt(dt).
    p.sigma_noise = math.sqrt(sigma2)
    return p
