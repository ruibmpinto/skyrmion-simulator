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

        - m[i, j, 0] = m_x at site (i, j)
        - m[i, j, 1] = m_y
        - m[i, j, 2] = m_z

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

    Notes
    -----
    The four shifted arrays let any nearest-neighbor stencil be
    written as a single vectorized expression. Compare the explicit
    loop form with the roll form for the discrete Laplacian:

    # Loop form (slow Python, but transparent)
    laplacian = np.zeros_like(m)
    for i in range(ny):
        for j in range(nx):
            ip = (i + 1) % ny
            im = (i - 1) % ny
            jp = (j + 1) % nx
            jm = (j - 1) % nx
            laplacian[i, j] = (
                m[i, jp] + m[i, jm] + m[ip, j] + m[im, j]
                - 4 * m[i, j])

    vs.

    # Roll form (vectorized, fast)
    m_px, m_mx, m_py, m_my = neighbors(m)
    laplacian = m_px + m_mx + m_py + m_my - 4 * m
    """
    # np.roll wraps indices, giving periodic BCs for free.
    # axis=1 is x (column index); axis=0 is y (row index).
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shift left along x: site at (i, j+1) now appears at (i, j),
    # so m_px[i, j] is the +x neighbor of m[i, j].
    # After rolling, the value at index $j$ equals the original value 
    # at the neighbor position.
    m_px = np.roll(m, -1, axis=1)
    # Shift right along x: site at (i, j-1) now appears at (i, j),
    # so m_mx[i, j] is the -x neighbor of m[i, j].
    m_mx = np.roll(m, +1, axis=1)
    # Shift up along y: site at (i+1, j) now appears at (i, j),
    # so m_py[i, j] is the +y neighbor of m[i, j].
    m_py = np.roll(m, -1, axis=0)
    # Shift down along y: site at (i-1, j) now appears at (i, j),
    # so m_my[i, j] is the -y neighbor of m[i, j].
    m_my = np.roll(m, +1, axis=0)
    # Return the four neighbor arrays in the canonical (+x,-x,+y,-y)
    # order expected by fields.exchange_field and fields.dmi_field.
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
    # meshgrid with default 'xy' indexing: jj indexes columns (x),
    # ii indexes rows (y); matches the (ny, nx, 3) spin convention.
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build 2D index grids: jj[i, j] = j (column), ii[i, j] = i (row).
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
    )
    # Allocate position array (z = 0 for every site by default).
    pos = np.zeros((ny, nx, 3))
    # x-coordinate at site (i, j) = column index * lattice constant.
    pos[..., 0] = jj * a
    # y-coordinate at site (i, j) = row index * lattice constant.
    pos[..., 1] = ii * a
    # z-coordinate is left at 0; main.py offsets the bottom layer by
    # -t_Co for Ovito visualization only.
    return pos
