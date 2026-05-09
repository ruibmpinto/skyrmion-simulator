"""Initial spin configurations for skyrmion simulations.

Generates Neel skyrmion profiles and uniform magnetic states
on a 2D square lattice.

Functions
---------
skyrmion_profile
    Generate a single Neel skyrmion.
uniform_state
    Generate a uniform magnetization state.
saf_skyrmion
    Generate SAF skyrmion pair (antiparallel cores).
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


def skyrmion_profile(nx, ny, a, R, dw=27e-9, polarity=1):
    """Generate a single Neel skyrmion at the lattice center.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    a : float
        Lattice constant in meters.
    R : float
        Skyrmion radius in meters (mz=0 contour).
    dw : float, default=27e-9
        Domain wall width in meters. From the 1D
        Euler-Lagrange solution: dw = sqrt(A / K_eff).
        Paper value: 27 nm.
    polarity : {1, -1}, default=1
        Core polarity. +1 means core points down (m_z=-1),
        background points up (m_z=+1).

    Returns
    -------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).

    Notes
    -----
    Domain-wall profile (1D Euler-Lagrange solution):
        theta(r) = 2 * arctan(exp(-(r - R) / dw))

    This gives:
        r << R: theta -> pi  (m_z = -1, core)
        r  = R: theta  = pi/2 (m_z = 0)
        r >> R: theta -> 0   (m_z = +1, background)

    The wall is concentrated in a ring of width ~dw
    around r = R.

    Neel helicity:
        phi = atan2(dy, dx)
    """
    # Lattice center
    x0 = (nx - 1) * a / 2.0
    y0 = (ny - 1) * a / 2.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Coordinate grids
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
    )
    x = jj * a - x0
    y = ii * a - y0
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Domain-wall polar angle profile
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    if polarity == -1:
        theta = np.pi - theta
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin components (Neel skyrmion)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    m = np.zeros((ny, nx, 3))
    m[..., 0] = sin_theta * np.cos(phi)
    m[..., 1] = sin_theta * np.sin(phi)
    m[..., 2] = cos_theta
    return m


# ---------------------------------------------------------------------
def uniform_state(nx, ny, direction=None):
    """Generate a uniform magnetization state.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    direction : numpy.ndarray(1d), default=None
        Unit vector for magnetization direction. Defaults
        to +z if None.

    Returns
    -------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    """
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    direction = direction / np.linalg.norm(direction)
    m = np.zeros((ny, nx, 3))
    m[..., :] = direction[np.newaxis, np.newaxis, :]
    return m


# ---------------------------------------------------------------------
def saf_skyrmion(nx, ny, a, R, dw=27e-9):
    """Generate a SAF skyrmion pair.

    Top layer has a skyrmion with core down (m_z=-1),
    bottom layer has a skyrmion with core up (m_z=+1).
    This is the ground state for antiferromagnetic RKKY
    coupling.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    a : float
        Lattice constant in meters.
    R : float
        Skyrmion radius in meters.
    dw : float, default=27e-9
        Domain wall width in meters.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
    """
    m_top = skyrmion_profile(
        nx, ny, a, R, dw=dw, polarity=1,
    )
    m_bot = skyrmion_profile(
        nx, ny, a, R, dw=dw, polarity=-1,
    )
    return m_top, m_bot
