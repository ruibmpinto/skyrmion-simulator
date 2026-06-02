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
    # Loud rejection of an unrecognised polarity sign.
    if polarity not in (1, -1):
        raise RuntimeError(
            f'skyrmion_profile: polarity must be +1 or -1, '
            f'got {polarity!r}.')
    # Lattice center
    # Half-integer offset so center sits between sites for even nx, ny.
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
    # Neel helicity: in-plane spins point radially (m_xy parallel to r).
    phi = np.arctan2(y, x)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Domain-wall polar angle profile
    # 1D Euler-Lagrange solution: theta(r) interpolates pi (core) -> 0.
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    # Flip the profile so core points up (m_z = +1) instead of down.
    if polarity == -1:
        theta = np.pi - theta
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin components (Neel skyrmion)
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    m = np.zeros((ny, nx, 3))
    # Radial in-plane components (Neel form, not Bloch).
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
    # Default to the PMA easy axis (+z) when no direction is supplied.
    if direction is None:
        direction = np.array([0.0, 0.0, 1.0])
    # Loud rejection of a zero-magnitude direction (undefined unit
    # vector); silent division would broadcast NaN to every site.
    norm = np.linalg.norm(direction)
    if norm == 0.0:
        raise RuntimeError(
            'uniform_state: zero-magnitude direction vector.')
    # Enforce |direction| = 1 in case the caller passed a non-unit vec.
    direction = direction / norm
    # Allocate the spin array; the (ny, nx, 3) layout matches the rest
    # of the simulator (axis 0 = y, axis 1 = x, axis 2 = component).
    m = np.zeros((ny, nx, 3))
    # Broadcast the same 3-vector to every site via newaxis on (y, x).
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
    # Opposite polarities make m_top antiparallel to m_bot, which is
    # the ground state of the antiferromagnetic RKKY coupling.
    m_top = skyrmion_profile(nx, ny, a, R, dw=dw, polarity=1,)
    
    m_bot = skyrmion_profile(nx, ny, a, R, dw=dw, polarity=-1,)
    return m_top, m_bot
