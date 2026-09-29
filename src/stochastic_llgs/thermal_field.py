"""Thermal (Brown) noise field sampler.

Generates a per-cell Gaussian white-noise field with the
amplitude prescribed by the fluctuation-dissipation theorem,
scaled so that the Stratonovich-Heun integrator produces an
increment of variance `sigma**2 * dt` per step.

Functions
---------
sample_thermal_field
    Draw a Gaussian noise field of shape (ny, nx, 3) with std
    sigma / sqrt(dt).
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


def sample_thermal_field(rng, shape, sigma, dt):
    """Draw a per-cell Gaussian thermal field.

    Parameters
    ----------
    rng : numpy.random.Generator
        Random generator. The caller must construct this
        explicitly; no fallback is provided.
    shape : tuple[int]
        Spatial shape (ny, nx). The output adds a length-3
        trailing axis for the (h_x, h_y, h_z) components.
    sigma : float
        Noise amplitude in Tesla * sqrt(s) (typically
        `p.sigma_noise` from `parameters_thermal.attach_thermal`). Strictly
        positive.
    dt : float
        Integration time step in seconds. Strictly positive.

    Returns
    -------
    h : numpy.ndarray(3d)
        Thermal field in Tesla, shape `(ny, nx, 3)`, with
        independent Gaussian samples of standard deviation
        `sigma / sqrt(dt)`. After the Heun increment
        `dt * f(m, H + h)` is formed, the noise contribution
        to the magnetization increment has variance
        `sigma**2 * dt`.

    Notes
    -----
    The amplitude `sigma / sqrt(dt)` is the standard
    discrete-time representation of a continuous-time Wiener
    process with intensity `sigma`: the integral
    `int_0^dt h(t) dt` is approximated by `h * dt` with
    `h ~ N(0, sigma**2 / dt)`, giving an increment whose
    variance is `sigma**2 * dt`.
    """
    if not isinstance(rng, np.random.Generator):
        raise RuntimeError(
            'sample_thermal_field: rng must be a '
            'numpy.random.Generator; the caller must '
            'construct it explicitly.'
        )
    if not (isinstance(shape, tuple) and len(shape) == 2):
        raise RuntimeError(
            f'sample_thermal_field: shape must be a 2-tuple '
            f'(ny, nx), got {shape!r}.'
        )
    ny, nx = shape
    if not (isinstance(ny, (int, np.integer))
            and isinstance(nx, (int, np.integer))):
        raise RuntimeError(
            f'sample_thermal_field: shape entries must be '
            f'ints, got {shape!r}.'
        )
    if ny <= 0 or nx <= 0:
        raise RuntimeError(
            f'sample_thermal_field: shape entries must be '
            f'positive, got {shape!r}.'
        )
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise RuntimeError(
            f'sample_thermal_field: sigma must be finite '
            f'and strictly positive, got {sigma!r}.'
        )
    if not np.isfinite(dt) or dt <= 0.0:
        raise RuntimeError(
            f'sample_thermal_field: dt must be finite and '
            f'strictly positive, got {dt!r}.'
        )
    std = sigma / math.sqrt(dt)
    return rng.standard_normal((int(ny), int(nx), 3)) * std
