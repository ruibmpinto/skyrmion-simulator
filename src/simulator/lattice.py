"""Lattice geometry, neighbor indexing, and region masks.

Provides nearest-neighbor arrays on a 2D square lattice
using periodic boundary conditions via np.roll, plus mask
builders that mark cells as inside or outside a magnetic
region for use with the free-BC path in `fields.py`.

Functions
---------
neighbors
    Return four nearest-neighbor shifted arrays.
lattice_positions
    Return physical coordinates for all lattice sites.
disk_mask
    Boolean mask for a circular magnetic region.
rect_mask
    Boolean mask for a rectangular magnetic region.
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


def disk_mask(nx, ny, a, R, center=None):
    """Build a Boolean (ny, nx) mask for a circular region.

    Parameters
    ----------
    nx, ny : int
        Lattice size in cells.
    a : float
        Lattice constant in metres.
    R : float
        Disk radius in metres.
    center : {tuple(float, float), None}, default=None
        Disk centre in (x, y) metres. None puts the centre
        at the geometric centre of the lattice.

    Returns
    -------
    mask : numpy.ndarray(2d, bool)
        True inside the disk, False outside.

    Notes
    -----
    Loud rejection of non-positive nx, ny, a, R.
    """
    if int(nx) <= 0 or int(ny) <= 0:
        raise RuntimeError(
            f'disk_mask: nx, ny must be positive integers, '
            f'got nx={nx!r}, ny={ny!r}.')
    if not (float(a) > 0.0):
        raise RuntimeError(
            f'disk_mask: a must be > 0, got a={a!r}.')
    if not (float(R) > 0.0):
        raise RuntimeError(
            f'disk_mask: R must be > 0, got R={R!r}.')
    # Default centre: geometric centre of the lattice in metres.
    if center is None:
        cx = 0.5 * (int(nx) - 1) * float(a)
        cy = 0.5 * (int(ny) - 1) * float(a)
    else:
        cx, cy = float(center[0]), float(center[1])
    # Index grids: jj is the column (x) index, ii the row (y) index.
    jj, ii = np.meshgrid(
        np.arange(int(nx), dtype=float),
        np.arange(int(ny), dtype=float),
    )
    # Physical offsets from the disk centre (metres).
    x = jj * float(a) - cx
    y = ii * float(a) - cy
    # Inside-the-disk test r^2 <= R^2 (squared to avoid a sqrt).
    return (x * x + y * y) <= (float(R) * float(R))


def rect_mask(nx, ny, a, Lx, Ly, center=None):
    """Build a Boolean (ny, nx) mask for a rectangular region.

    Parameters
    ----------
    nx, ny : int
        Lattice size in cells.
    a : float
        Lattice constant in metres.
    Lx, Ly : float
        Rectangle dimensions in metres.
    center : {tuple(float, float), None}, default=None
        Rectangle centre in (x, y) metres. None puts the
        centre at the geometric centre of the lattice.

    Returns
    -------
    mask : numpy.ndarray(2d, bool)
        True inside the rectangle, False outside.
    """
    if int(nx) <= 0 or int(ny) <= 0:
        raise RuntimeError(
            f'rect_mask: nx, ny must be positive integers, '
            f'got nx={nx!r}, ny={ny!r}.')
    if not (float(a) > 0.0 and float(Lx) > 0.0
            and float(Ly) > 0.0):
        raise RuntimeError(
            f'rect_mask: a, Lx, Ly must be > 0, got a={a!r}, '
            f'Lx={Lx!r}, Ly={Ly!r}.')
    # Default centre: geometric centre of the lattice in metres.
    if center is None:
        cx = 0.5 * (int(nx) - 1) * float(a)
        cy = 0.5 * (int(ny) - 1) * float(a)
    else:
        cx, cy = float(center[0]), float(center[1])
    # Index grids: jj is the column (x) index, ii the row (y) index.
    jj, ii = np.meshgrid(
        np.arange(int(nx), dtype=float),
        np.arange(int(ny), dtype=float),
    )
    # Physical offsets from the rectangle centre (metres).
    x = jj * float(a) - cx
    y = ii * float(a) - cy
    half_Lx = 0.5 * float(Lx)
    half_Ly = 0.5 * float(Ly)
    # Inside-the-rectangle test: within half-extents on both axes.
    return (np.abs(x) <= half_Lx) & (np.abs(y) <= half_Ly)
