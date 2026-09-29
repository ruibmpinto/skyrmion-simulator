"""Digitised Banerjee 2014 Fig. 1(b) phase boundaries.

Reference: S. Banerjee, J. Rowland, O. Erten, M. Randeria,
Phys. Rev. X 4, 031045 (2014), Fig. 1(b) -- the T = 0
anisotropy-field phase diagram (FM / spiral / SkX) for a 2D
Rashba chiral magnet from a circular-cell variational ansatz
(D/J = 0.01, A_c J/D^2 = 1/2, A = A_c + A_s).

Banerjee's axes are A_s J/D^2 (x) and H J/D^2 (y) -- the SAME
dimensionless plane as Güngördü 2016 Fig. 3, so the two
references overlay directly on the simulator phase map with
no convention shift. Güngördü notes Banerjee's variational
boundaries are the dotted lines in their Fig. 3, and that the
ansatz underestimates SkX stability and misses the SC phase.

The FM-tilt line H = 2A maps to H J/D^2 = 1 + 2 a_s here
(A_c J/D^2 = 1/2).

Module constants
----------------
BAN_FM_SKX
    Digitised FM <-> SkX boundary, list of (a_s, h_g).
BAN_SKX_SP
    Digitised SkX <-> spiral boundary, list of (a_s, h_g).

Functions
---------
ban_fm_skx_hc(a_s)
    Banerjee FM<->SkX critical field h_g at anisotropy a_s
    (linear interpolation of BAN_FM_SKX).
ban_fm_tilt(a_s)
    The H = 2A FM out-of-plane / tilted separator,
    h_g = 1 + 2 a_s.
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
# Digitised Banerjee 2014 Fig. 1(b), coordinates (a_s, h_g).
# FM <-> SkX (bold black upper-left edge of the SkX region).
BAN_FM_SKX = [
    (-1.25, 0.00), (-1.00, 0.30), (-0.50, 0.65),
    (0.00, 1.00), (0.50, 1.50), (1.00, 2.10),
    (1.25, 2.30),
]
# SkX <-> spiral (lower edge of the SkX region).
BAN_SKX_SP = [
    (-0.50, 0.10), (0.00, 0.40), (0.50, 0.55),
    (1.00, 0.72), (1.30, 0.90),
]


def ban_fm_skx_hc(a_s):
    """Banerjee FM<->SkX critical field h_g at anisotropy a_s.

    Parameters
    ----------
    a_s : float
        Dimensionless anisotropy A_s J / D^2.

    Returns
    -------
    h_c : float
        Critical field h_g = H J / D^2 on the FM<->SkX
        boundary (linear interpolation of the digitised
        BAN_FM_SKX polyline). Returns NaN outside the
        digitised a_s range so callers do not extrapolate.
    """
    xs = np.array([p[0] for p in BAN_FM_SKX])
    ys = np.array([p[1] for p in BAN_FM_SKX])
    if a_s < xs.min() or a_s > xs.max():
        return float('nan')
    return float(np.interp(a_s, xs, ys))


def ban_fm_tilt(a_s):
    """FM out-of-plane / tilted-FM separator: h_g = 1 + 2 a_s
    (Banerjee H = 2A with A_c J/D^2 = 1/2).

    Parameters
    ----------
    a_s : float
        Dimensionless anisotropy A_s J / D^2.

    Returns
    -------
    h_g : float
        Field on the FM-tilt line.
    """
    return 1.0 + 2.0 * float(a_s)
