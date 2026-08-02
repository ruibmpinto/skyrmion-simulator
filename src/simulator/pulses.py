"""Time profiles for the current density driving the SOT terms.

A pulse is any callable mapping time (in seconds) to current
density (in A/m^2). The integrator evaluates the pulse at every
RK4 substage to obtain time-varying SOT effective fields:

    H_DL(t) = chi_DL * J(t),    H_FL(t) = chi_FL * J(t).

Classes
-------
ConstantPulse
    J(t) = J0 for all t.
SquarePulse
    J(t) = J0 inside [t_start, t_end], else 0.
GaussianPulse
    J(t) = J0 * exp(-(t - t_center)^2 / (2 sigma^2)), with sigma
    derived from the FWHM.
TrianglePulse
    Linear rise to J0 at t_peak then linear fall; t_peak placement
    selects sharp-rise, sharp-fall or symmetric.
HalfSinePulse
    J(t) = J0 * sin(pi (t - t_start) / (t_end - t_start)).
SuperpositionPulse
    Sum of pulses, for multi-pulse trains or composite shapes.

Notes
-----
All pulses implement the same interface: a __call__(t) method that
returns the current density at time t. The integrator does not
inspect the concrete class.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


class ConstantPulse:
    """Time-independent current density.

    Recovers the pre-pulse-refactor behavior of the simulator.

    Attributes
    ----------
    J0 : float
        Constant current density in A/m^2.

    Methods
    -------
    __call__(self, t)
        Return J0 regardless of t.
    """
    def __init__(self, J0):
        """Constructor.

        Parameters
        ----------
        J0 : float
            Current density in A/m^2.
        """
        # Cast to float so isinstance checks and arithmetic are stable.
        self.J0 = float(J0)
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds (ignored).

        Returns
        -------
        J : float
            Current density in A/m^2.
        """
        # Constant in time: ignore t entirely.
        return self.J0
    # -------------------------------------------------------------------------
    def __repr__(self):
        return f'ConstantPulse(J0={self.J0:.3e})'


# =============================================================================
class SquarePulse:
    """Square (top-hat) pulse with finite duration.

    J(t) = J0 for t_start <= t <= t_end, else 0. No rise or fall.
    Use for DC drive of bounded duration (e.g. S41, S48).

    Attributes
    ----------
    J0 : float
        Pulse amplitude in A/m^2.
    t_start : float
        Pulse leading edge in seconds.
    t_end : float
        Pulse trailing edge in seconds.

    Methods
    -------
    __call__(self, t)
        Return J0 inside the window, else 0.
    """
    def __init__(self, J0, t_start, t_end):
        """Constructor.

        Parameters
        ----------
        J0 : float
            Pulse amplitude in A/m^2.
        t_start : float
            Pulse leading edge in seconds.
        t_end : float
            Pulse trailing edge in seconds.
        """
        # Reject inverted windows loudly; silent acceptance corrupts runs.
        if t_end <= t_start:
            raise RuntimeError(
                f't_end ({t_end}) must exceed t_start ({t_start}).')
        self.J0 = float(J0)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        J : float
            J0 if t in [t_start, t_end], otherwise 0.
        """
        # Inclusive bounds: matches the typical pulse-window convention.
        if self.t_start <= t <= self.t_end:
            return self.J0
        return 0.0
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'SquarePulse(J0={self.J0:.3e}, '
                f't_start={self.t_start:.3e}, '
                f't_end={self.t_end:.3e})')


# =============================================================================
class GaussianPulse:
    """Gaussian pulse with given peak amplitude, center and FWHM.

    J(t) = J0 * exp(-0.5 * ((t - t_center) / sigma)^2),
    with sigma = FWHM / (2 * sqrt(2 * ln 2)).

    Attributes
    ----------
    J0 : float
        Peak current density in A/m^2.
    t_center : float
        Time at which J is maximal in seconds.
    FWHM : float
        Full width at half maximum in seconds.
    sigma : float
        Standard deviation in seconds (derived from FWHM).

    Methods
    -------
    __call__(self, t)
        Return J(t).
    """
    # Conversion factor: FWHM = 2 sqrt(2 ln 2) sigma.
    _FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    # -------------------------------------------------------------------------
    def __init__(self, J0, t_center, FWHM):
        """Constructor.

        Parameters
        ----------
        J0 : float
            Peak current density in A/m^2.
        t_center : float
            Time at which J is maximal in seconds.
        FWHM : float
            Full width at half maximum in seconds.
        """
        # Reject non-positive widths loudly; silent acceptance corrupts runs.
        if FWHM <= 0.0:
            raise RuntimeError(f'FWHM must be positive, got {FWHM}.')
        self.J0 = float(J0)
        self.t_center = float(t_center)
        self.FWHM = float(FWHM)
        # Derived standard deviation; cached to avoid recomputation per call.
        self.sigma = self.FWHM * self._FWHM_TO_SIGMA
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        J : float
            J(t) in A/m^2.
        """
        # Direct Gaussian evaluation; sigma is precomputed.
        z = (t - self.t_center) / self.sigma
        return self.J0 * np.exp(-0.5 * z * z)
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'GaussianPulse(J0={self.J0:.3e}, '
                f't_center={self.t_center:.3e}, '
                f'FWHM={self.FWHM:.3e})')


# =============================================================================
class TrianglePulse:
    """Triangular pulse with a placeable peak.

    J rises linearly from 0 at `t_start` to `J0` at `t_peak`, then
    falls linearly back to 0 at `t_end`; outside the window J = 0.
    Moving `t_peak` selects the asymmetry, which is why one class
    covers all three variants of interest:

    - `t_peak == t_start`: instantaneous rise, linear fall.
    - `t_peak == t_end`: linear rise, instantaneous fall.
    - `t_peak` midway: symmetric ramp up and down.

    Both the delivered charge, 0.5 * J0 * (t_end - t_start), and the
    action, J0^2 * (t_end - t_start) / 3, are independent of where
    the peak sits, so the three variants are matched in charge and in
    dissipated energy and differ only in asymmetry.

    Attributes
    ----------
    J0 : float
        Peak current density in A/m^2.
    t_start : float
        Pulse leading edge in seconds.
    t_peak : float
        Time of the peak in seconds, in [t_start, t_end].
    t_end : float
        Pulse trailing edge in seconds.

    Methods
    -------
    __call__(self, t)
        Return J(t).
    """
    def __init__(self, J0, t_start, t_peak, t_end):
        """Constructor.

        Parameters
        ----------
        J0 : float
            Peak current density in A/m^2.
        t_start : float
            Pulse leading edge in seconds.
        t_peak : float
            Time of the peak in seconds; must lie in
            [t_start, t_end]. Equal to an edge for a vertical
            rise or fall.
        t_end : float
            Pulse trailing edge in seconds.
        """
        # Reject inverted windows and out-of-window peaks loudly;
        # silent acceptance corrupts runs.
        if t_end <= t_start:
            raise RuntimeError(
                f't_end ({t_end}) must exceed t_start ({t_start}).')
        if not (t_start <= t_peak <= t_end):
            raise RuntimeError(
                f't_peak ({t_peak}) must lie within '
                f'[{t_start}, {t_end}].')
        self.J0 = float(J0)
        self.t_start = float(t_start)
        self.t_peak = float(t_peak)
        self.t_end = float(t_end)
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        J : float
            J(t) in A/m^2; 0 outside [t_start, t_end].
        """
        # Inclusive window, matching SquarePulse's convention.
        if t < self.t_start or t > self.t_end:
            return 0.0
        if t < self.t_peak:
            # Rising edge; the degenerate case t_peak == t_start is
            # unreachable here because t < t_peak would imply
            # t < t_start, already returned above.
            return self.J0 * ((t - self.t_start)
                              / (self.t_peak - self.t_start))
        # Falling edge, and t == t_peak. A peak pinned to t_end has
        # no falling edge, so the amplitude is held.
        if self.t_end == self.t_peak:
            return self.J0
        return self.J0 * ((self.t_end - t)
                          / (self.t_end - self.t_peak))
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'TrianglePulse(J0={self.J0:.3e}, '
                f't_start={self.t_start:.3e}, '
                f't_peak={self.t_peak:.3e}, '
                f't_end={self.t_end:.3e})')


# =============================================================================
class HalfSinePulse:
    """Half-period sine lobe over a finite window.

    J(t) = J0 * sin(pi * (t - t_start) / (t_end - t_start)) inside
    [t_start, t_end] and 0 outside, so J vanishes at both edges and
    peaks at the midpoint. Delivered charge is
    2 * J0 * (t_end - t_start) / pi and the action is
    J0^2 * (t_end - t_start) / 2.

    Attributes
    ----------
    J0 : float
        Peak current density in A/m^2.
    t_start : float
        Pulse leading edge in seconds.
    t_end : float
        Pulse trailing edge in seconds.

    Methods
    -------
    __call__(self, t)
        Return J(t).
    """
    def __init__(self, J0, t_start, t_end):
        """Constructor.

        Parameters
        ----------
        J0 : float
            Peak current density in A/m^2.
        t_start : float
            Pulse leading edge in seconds.
        t_end : float
            Pulse trailing edge in seconds.
        """
        # Reject inverted windows loudly; silent acceptance corrupts
        # runs.
        if t_end <= t_start:
            raise RuntimeError(
                f't_end ({t_end}) must exceed t_start ({t_start}).')
        self.J0 = float(J0)
        self.t_start = float(t_start)
        self.t_end = float(t_end)
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        J : float
            J(t) in A/m^2; 0 outside [t_start, t_end].
        """
        # Inclusive window, matching SquarePulse's convention.
        if t < self.t_start or t > self.t_end:
            return 0.0
        phase = (np.pi * (t - self.t_start)
                 / (self.t_end - self.t_start))
        return self.J0 * np.sin(phase)
    # -------------------------------------------------------------------------
    def __repr__(self):
        return (f'HalfSinePulse(J0={self.J0:.3e}, '
                f't_start={self.t_start:.3e}, '
                f't_end={self.t_end:.3e})')


# =============================================================================
class SuperpositionPulse:
    """Linear combination of pulses.

    J(t) = sum_i pulses[i](t). Use for pulse trains, multi-shape
    composites, or any user-defined superposition.

    Attributes
    ----------
    pulses : list
        Underlying pulse callables.

    Methods
    -------
    __call__(self, t)
        Return the sum of pulse contributions at time t.
    """
    def __init__(self, pulses):
        """Constructor.

        Parameters
        ----------
        pulses : list
            Sequence of pulse callables (each implementing __call__).
        """
        # Reject empty superpositions loudly; the implied J=0 trivial case
        # should be expressed as ConstantPulse(0.0).
        if len(pulses) == 0:
            raise RuntimeError(
                'SuperpositionPulse requires at least one pulse; '
                'use ConstantPulse(0.0) for the zero current case.')
        self.pulses = list(pulses)
    # -------------------------------------------------------------------------
    def __call__(self, t):
        """Return the current density at time t.

        Parameters
        ----------
        t : float
            Time in seconds.

        Returns
        -------
        J : float
            Sum of constituent pulse values at t.
        """
        # Sum each pulse contribution; small N keeps the Python loop cheap.
        total = 0.0
        for p in self.pulses:
            total += p(t)
        return total
    # -------------------------------------------------------------------------
    def __repr__(self):
        return f'SuperpositionPulse({self.pulses!r})'
# =============================================================================
