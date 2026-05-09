"""Lattice geometry and neighbor indexing with PBC.

Provides nearest-neighbor arrays on a 2D square lattice
using periodic boundary conditions via np.roll.

Functions
---------
neighbors
    Return four nearest-neighbor shifted arrays.
lattice_positions
    Return physical coordinates for all lattice sites.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def neighbors(m):
    """Return four nearest-neighbor arrays with PBC.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).

    Returns
    -------
    m_px : numpy.ndarray(3d)
        Neighbor in +x direction (j+1).
    m_mx : numpy.ndarray(3d)
        Neighbor in -x direction (j-1).
    m_py : numpy.ndarray(3d)
        Neighbor in +y direction (i+1).
    m_my : numpy.ndarray(3d)
        Neighbor in -y direction (i-1).
    """
    m_px = np.roll(m, -1, axis=1)
    m_mx = np.roll(m, +1, axis=1)
    m_py = np.roll(m, -1, axis=0)
    m_my = np.roll(m, +1, axis=0)
    return m_px, m_mx, m_py, m_my


def lattice_positions(nx, ny, a):
    """Return physical coordinates for all lattice sites.

    Parameters
    ----------
    nx : int
        Number of sites along x.
    ny : int
        Number of sites along y.
    a : float
        Lattice constant in meters.

    Returns
    -------
    pos : numpy.ndarray(3d)
        Position array, shape (ny, nx, 3). The z-coordinate
        is zero for all sites.
    """
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
    )
    pos = np.zeros((ny, nx, 3))
    pos[..., 0] = jj * a
    pos[..., 1] = ii * a
    return pos
