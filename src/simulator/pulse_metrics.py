"""Delivered charge and action of a current pulse.

Two integrals normalise pulse shapes against each other and turn a
displacement into an efficiency:

    charge  Q = int J(t) dt          (A s / m^2)
    action  S = int J(t)^2 dt        (A^2 s / m^4)

`Q` is the natural cost measure when the budget is current-time
(displacement per charge) and `S` is proportional to the Ohmic
dissipation of the drive line, so displacement per unit `S` is the
displacement per Joule. Both are computed by quadrature on a uniform
grid so any callable pulse is supported, including
`SuperpositionPulse`; `analytic_charge` and `analytic_action` give
closed forms for the primitive shapes and exist to check the
quadrature.

Functions
---------
pulse_charge
    Quadrature of int J dt over a window.
pulse_action
    Quadrature of int J^2 dt over a window.
analytic_charge
    Closed-form charge of a primitive pulse.
analytic_action
    Closed-form action of a primitive pulse.

Notes
-----
The quadrature is Simpson's rule on an odd number of samples, exact
for the piecewise-linear and quadratic parts of the triangular and
square shapes up to the kink, and converging as h^4 for the smooth
ones. Sample the window densely enough that a kink is not straddled
by a single interval: `n_sample` of a few thousand over a
nanosecond-scale pulse is ample. The count must be odd; an even one
is rejected, never silently adjusted.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
# Local
from src.simulator.pulses import (
    ConstantPulse, GaussianPulse, HalfSinePulse, SquarePulse,
    TrianglePulse,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _simpson(values, dt):
    """Simpson's rule over uniformly spaced samples.

    Parameters
    ----------
    values : numpy.ndarray(1d)
        Integrand samples; the length must be odd and at least 3.
    dt : float
        Sample spacing.

    Returns
    -------
    integral : float
        Approximated integral.
    """
    n = values.size
    if n < 3 or n % 2 == 0:
        raise RuntimeError(
            f'_simpson: needs an odd sample count >= 3, got {n}.')
    weights = np.ones(n)
    weights[1:-1:2] = 4.0
    weights[2:-1:2] = 2.0
    return float(dt/3.0*np.dot(weights, values))


# -----------------------------------------------------------------------------
def _sampled(pulse, t_start, t_end, n_sample):
    """Sample a pulse on a uniform grid with an odd sample count.

    Parameters
    ----------
    pulse : callable
        Pulse mapping time (s) to current density (A/m^2).
    t_start, t_end : float
        Integration window in seconds.
    n_sample : int
        Sample count; must be odd and at least 3 so Simpson's rule
        applies. An even count is rejected rather than adjusted, so
        the quadrature never runs on a grid the caller did not ask
        for.

    Returns
    -------
    values : numpy.ndarray(1d)
        Pulse samples.
    dt : float
        Sample spacing in seconds.
    """
    if t_end <= t_start:
        raise RuntimeError(
            f'_sampled: t_end ({t_end}) must exceed t_start '
            f'({t_start}).')
    if n_sample < 3:
        raise RuntimeError(
            f'_sampled: n_sample must be >= 3, got {n_sample}.')
    if int(n_sample) % 2 == 0:
        raise RuntimeError(
            f'_sampled: n_sample must be odd for Simpson\'s rule, '
            f'got {n_sample}.')
    grid = np.linspace(t_start, t_end, int(n_sample))
    values = np.array([float(pulse(t)) for t in grid])
    return values, float(grid[1] - grid[0])


# -----------------------------------------------------------------------------
def pulse_charge(pulse, t_start, t_end, n_sample):
    """Delivered charge int J dt over a window.

    Parameters
    ----------
    pulse : callable
        Pulse mapping time (s) to current density (A/m^2).
    t_start, t_end : float
        Integration window in seconds.
    n_sample : int
        Quadrature sample count; must be odd and >= 3.

    Returns
    -------
    charge : float
        int J dt in A s / m^2.
    """
    values, dt = _sampled(pulse, t_start, t_end, n_sample)
    return _simpson(values, dt)


# -----------------------------------------------------------------------------
def pulse_action(pulse, t_start, t_end, n_sample):
    """Action int J^2 dt over a window.

    Parameters
    ----------
    pulse : callable
        Pulse mapping time (s) to current density (A/m^2).
    t_start, t_end : float
        Integration window in seconds.
    n_sample : int
        Quadrature sample count; must be odd and >= 3.

    Returns
    -------
    action : float
        int J^2 dt in A^2 s / m^4.
    """
    values, dt = _sampled(pulse, t_start, t_end, n_sample)
    return _simpson(values*values, dt)


# -----------------------------------------------------------------------------
def analytic_charge(pulse):
    """Closed-form charge of a primitive pulse.

    Parameters
    ----------
    pulse : {SquarePulse, GaussianPulse, TrianglePulse, HalfSinePulse}
        Pulse whose charge is known in closed form. `ConstantPulse`
        is rejected: its charge diverges without a window.

    Returns
    -------
    charge : float
        int J dt over the pulse's own support, in A s / m^2.
    """
    if isinstance(pulse, SquarePulse):
        return pulse.J0*(pulse.t_end - pulse.t_start)
    if isinstance(pulse, TrianglePulse):
        # Independent of t_peak: the area of a triangle depends only
        # on base and height.
        return 0.5*pulse.J0*(pulse.t_end - pulse.t_start)
    if isinstance(pulse, HalfSinePulse):
        return (2.0/np.pi)*pulse.J0*(pulse.t_end - pulse.t_start)
    if isinstance(pulse, GaussianPulse):
        # Over the whole real line; the pulse has no finite support.
        return pulse.J0*pulse.sigma*np.sqrt(2.0*np.pi)
    if isinstance(pulse, ConstantPulse):
        raise RuntimeError(
            'analytic_charge: ConstantPulse has unbounded support; '
            'use pulse_charge with an explicit window.')
    raise RuntimeError(
        f'analytic_charge: no closed form for '
        f'{type(pulse).__name__}; use pulse_charge.')


# -----------------------------------------------------------------------------
def analytic_action(pulse):
    """Closed-form action of a primitive pulse.

    Parameters
    ----------
    pulse : {SquarePulse, GaussianPulse, TrianglePulse, HalfSinePulse}
        Pulse whose action is known in closed form. `ConstantPulse`
        is rejected: its action diverges without a window.

    Returns
    -------
    action : float
        int J^2 dt over the pulse's own support, in A^2 s / m^4.
    """
    if isinstance(pulse, SquarePulse):
        return pulse.J0*pulse.J0*(pulse.t_end - pulse.t_start)
    if isinstance(pulse, TrianglePulse):
        # Also independent of t_peak: the rise contributes
        # J0^2 (t_peak - t_start)/3 and the fall
        # J0^2 (t_end - t_peak)/3.
        return pulse.J0*pulse.J0*(pulse.t_end - pulse.t_start)/3.0
    if isinstance(pulse, HalfSinePulse):
        return 0.5*pulse.J0*pulse.J0*(pulse.t_end - pulse.t_start)
    if isinstance(pulse, GaussianPulse):
        return pulse.J0*pulse.J0*pulse.sigma*np.sqrt(np.pi)
    if isinstance(pulse, ConstantPulse):
        raise RuntimeError(
            'analytic_action: ConstantPulse has unbounded support; '
            'use pulse_action with an explicit window.')
    raise RuntimeError(
        f'analytic_action: no closed form for '
        f'{type(pulse).__name__}; use pulse_action.')
