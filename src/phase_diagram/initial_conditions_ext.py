"""Extra initial conditions for the phase-diagram sweep.

Adds randomized and helical stripe initial states for the
SAF stack in addition to the existing skyrmion, FM, and
SAF-skyrmion ICs in `src.simulator.initial_conditions`.

Both ICs return an antiparallel SAF pair: the bottom layer
is the negative of the top so the AFM RKKY ground state is
respected at the start of relaxation.

Functions
---------
random_state
    Uniform-on-the-sphere random spins with antiparallel
    SAF pairing.
stripe_state
    Helical stripe pattern with period lambda = 4*pi*A/D.
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


def random_state(nx, ny, seed):
    """Return a uniform random SAF spin configuration.

    Spins are drawn isotropically on the unit sphere using
    phi ~ U[0, 2*pi] and cos(theta) ~ U[-1, 1]. The bottom
    layer is the negation of the top so the antiferromagnetic
    interlayer state is the natural starting point.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    seed : int
        RNG seed; identical seeds give identical ICs.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    """
    rng = np.random.default_rng(seed)
    phi = rng.uniform(0.0, 2.0 * np.pi, size=(ny, nx))
    cos_t = rng.uniform(-1.0, 1.0, size=(ny, nx))
    sin_t = np.sqrt(np.clip(1.0 - cos_t * cos_t, 0.0, 1.0))
    m_top = np.empty((ny, nx, 3))
    m_top[..., 0] = sin_t * np.cos(phi)
    m_top[..., 1] = sin_t * np.sin(phi)
    m_top[..., 2] = cos_t
    m_bot = -m_top
    return m_top, m_bot


# ---------------------------------------------------------------------
def stripe_state(nx, ny, period, a, axis='x'):
    """Return a helical-stripe SAF spin configuration.

    The top layer rotates in a plane containing the z-axis
    and the propagation direction:

        theta(r) = 2*pi * r / lambda
        m_top = (sin(theta) * d_hat) * (1 - delta_axis_z)
                + cos(theta) * z_hat,

    where r is the spatial coordinate along the propagation
    axis and d_hat is the in-plane unit vector along that
    axis. The bottom layer is the negation of the top.

    Parameters
    ----------
    nx : int
        Number of lattice sites along x.
    ny : int
        Number of lattice sites along y.
    period : float
        Stripe wavelength in meters. Use lambda = 4*pi*A/D
        for the natural DMI helix pitch.
    a : float
        Lattice constant in meters.
    axis : {'x', 'y'}, default='x'
        Direction of stripe propagation.

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    """
    if axis not in ('x', 'y'):
        raise RuntimeError(
            f"axis must be 'x' or 'y', got {axis!r}."
        )
    if period <= 0.0:
        raise RuntimeError(
            f'period must be positive, got {period}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        indexing='xy',
    )
    if axis == 'x':
        r = jj * a
        ip_dir = np.array([1.0, 0.0])
    else:
        r = ii * a
        ip_dir = np.array([0.0, 1.0])
    theta = 2.0 * np.pi * r / period
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top = np.zeros((ny, nx, 3))
    m_top[..., 0] = sin_t * ip_dir[0]
    m_top[..., 1] = sin_t * ip_dir[1]
    m_top[..., 2] = cos_t
    m_bot = -m_top
    return m_top, m_bot
