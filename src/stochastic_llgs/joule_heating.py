"""Uniform global Joule self-heating model.

For a current density `j` flowing through the SAF stack on a
substrate held at `T_sub`, the simplest self-heating model
yields a uniform global lattice temperature

    T(j) = T_sub + R_th * j**2.

`R_th` is a stack-dependent thermal resistance (units
`K * m^4 / A^2`); it is supplied explicitly by the caller and
must be obtained from independent measurement or device-level
simulation. No literature-anchored value is assumed here.

Functions
---------
T_of_j
    Compute `T(j) = T_sub + R_th * j**2`.
"""
#
#                                                                       Modules
# =============================================================================
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


def T_of_j(j, T_sub, R_th):
    """Compute the effective lattice temperature under
    uniform Joule heating.

    Parameters
    ----------
    j : float
        Current density in A/m^2. May be zero or negative
        (the heating term scales as `j**2`).
    T_sub : float
        Substrate (bath) temperature in Kelvin. Strictly
        positive.
    R_th : float
        Thermal resistance in `K * m^4 / A^2`. Non-negative
        (`R_th = 0` disables self-heating).

    Returns
    -------
    T : float
        Effective lattice temperature in Kelvin.
    """
    if not np.isfinite(j):
        raise RuntimeError(
            f'T_of_j: j must be finite, got {j!r}.'
        )
    if not (np.isfinite(T_sub) and T_sub > 0.0):
        raise RuntimeError(
            f'T_of_j: T_sub must be finite and strictly '
            f'positive, got {T_sub!r}.'
        )
    if not (np.isfinite(R_th) and R_th >= 0.0):
        raise RuntimeError(
            f'T_of_j: R_th must be finite and non-negative, '
            f'got {R_th!r}.'
        )
    return float(T_sub) + float(R_th) * float(j) ** 2
