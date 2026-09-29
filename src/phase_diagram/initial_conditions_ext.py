"""Extra initial conditions for the phase-diagram sweep.

Adds five seeds beyond the simulator's built-in
`skyrmion_profile`, `uniform_state`, and `saf_skyrmion`:

* random spheres (statistical exploration),
* helical stripes (1D periodic SS seed),
* hex skyrmion lattices (true SkX seed, Q != 0 per cell),
* hex bubble lattices (Q = 0 per cell, BX seed).

Every IC returns an antiparallel SAF pair: the bottom
layer is the negation of the top so the antiferromagnetic
RKKY ground state is respected at the start of
relaxation.

Functions
---------
random_state
    Uniform-on-the-sphere random spins.
stripe_state
    Helical stripe pattern of given period.
hex_lattice_skyrmions
    Hex array of Neel skyrmions (each carries Q = +-1).
hex_lattice_bubbles
    Hex array of axially-symmetric Q = 0 bubbles with the
    same m_z profile as `hex_lattice_skyrmions` but no in-
    plane winding.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
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


# ---------------------------------------------------------------------
def _hex_centers(nx, ny, a, period):
    """Return all hex-lattice site coordinates inside the box.

    Parameters
    ----------
    nx, ny : int
        Lattice cell counts.
    a : float
        Lattice constant (m).
    period : float
        Hex lattice constant (nearest-neighbor distance) in
        metres.

    Returns
    -------
    centers : numpy.ndarray(2d)
        Shape (n_centers, 2). The (x, y) coordinates of all
        hex sites whose Voronoi cell overlaps the box. Two
        layers of margin around the box ensure that grid
        points near the boundary see the correct nearest
        center under PBC.
    """
    if period <= 0.0:
        raise RuntimeError(
            f'period must be positive, got {period}.'
        )
    a1 = np.array([period, 0.0])
    a2 = np.array([0.5 * period, np.sqrt(3.0) / 2.0 * period])
    L_x = nx * a
    L_y = ny * a
    n1_max = int(np.ceil(L_x / period)) + 2
    n2_max = int(np.ceil(L_y / (np.sqrt(3.0) / 2.0 * period))) + 2
    centers = []
    for n1 in range(-2, n1_max + 1):
        for n2 in range(-2, n2_max + 1):
            r = n1 * a1 + n2 * a2
            if (-period <= r[0] <= L_x + period
                    and -period <= r[1] <= L_y + period):
                centers.append(r)
    return np.array(centers)


# ---------------------------------------------------------------------
def _nearest_center_field(nx, ny, a, centers):
    """Map each grid point to (r, phi) relative to its nearest hex center."""
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),
        indexing='xy',
    )
    x_grid = jj * a
    y_grid = ii * a
    # Broadcast distance to all centers; pick the argmin.
    dx = x_grid[:, :, None] - centers[None, None, :, 0]
    dy = y_grid[:, :, None] - centers[None, None, :, 1]
    dist2 = dx * dx + dy * dy
    nearest = np.argmin(dist2, axis=-1)
    rx = x_grid - centers[nearest, 0]
    ry = y_grid - centers[nearest, 1]
    r = np.sqrt(rx * rx + ry * ry)
    phi = np.arctan2(ry, rx)
    return r, phi


# ---------------------------------------------------------------------
def hex_lattice_skyrmions(nx, ny, a, R, period, dw):
    """Hex lattice of Neel skyrmions (each carries Q = -1).

    Each grid point picks up the profile of the nearest
    hex-site skyrmion. Top layer has core m_z = -1 with
    radial in-plane winding (Neel chirality); the bottom
    layer is the negation, giving a SAF skyrmion lattice.

    Parameters
    ----------
    nx, ny : int
        Lattice cell counts.
    a : float
        Lattice constant (m).
    R : float
        Skyrmion radius (m_z = 0 contour) in metres.
    period : float
        Hex lattice constant in metres. Typical choice:
        the natural DMI helix wavelength lambda = 4 pi A / D.
    dw : float
        Domain-wall width (m) controlling the profile
        slope. Use ~ sqrt(A_ex / K_eff).

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Shape (ny, nx, 3). Antiparallel SAF pair.
    """
    if R <= 0.0 or dw <= 0.0:
        raise RuntimeError(
            f'R and dw must be positive; got R={R}, dw={dw}.'
        )
    centers = _hex_centers(nx, ny, a, period)
    r, phi = _nearest_center_field(nx, ny, a, centers)
    # Standard 360-degree-wall skyrmion polar profile: m_z runs
    # from -1 at the core (r=0) to +1 far out, crossing 0 at r=R.
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    m_top = np.zeros((ny, nx, 3))
    m_top[..., 0] = sin_t * np.cos(phi)   # Neel winding
    m_top[..., 1] = sin_t * np.sin(phi)
    m_top[..., 2] = cos_t
    m_bot = -m_top
    return m_top, m_bot


# ---------------------------------------------------------------------
def hex_lattice_bubbles(nx, ny, a, R, period, dw):
    """Hex lattice of Q = 0 bubbles.

    Same hex sites and same m_z profile as
    `hex_lattice_skyrmions`, but the in-plane direction is
    held constant (along +x) rather than winding radially.
    Each bubble therefore carries zero topological charge,
    and the full state carries Q = 0 -- it sits in the BX
    branch of the classifier.

    Parameters
    ----------
    nx, ny : int
        Lattice cell counts.
    a : float
        Lattice constant (m).
    R : float
        Bubble radius (m_z = 0 contour) in metres.
    period : float
        Hex lattice constant in metres.
    dw : float
        Domain-wall width (m).

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Antiparallel SAF pair.
    """
    if R <= 0.0 or dw <= 0.0:
        raise RuntimeError(
            f'R and dw must be positive; got R={R}, dw={dw}.'
        )
    centers = _hex_centers(nx, ny, a, period)
    r, _phi = _nearest_center_field(nx, ny, a, centers)
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    m_top = np.zeros((ny, nx, 3))
    m_top[..., 0] = sin_t        # constant in-plane direction (+x)
    m_top[..., 1] = 0.0
    m_top[..., 2] = cos_t
    m_bot = -m_top
    return m_top, m_bot


# ---------------------------------------------------------------------
def _square_centers(nx, ny, a, period):
    """Return all square-lattice site coordinates inside the box.

    Square Bravais lattice (a1 = (period, 0), a2 = (0, period))
    with two layers of margin so boundary grid points see the
    correct nearest center under PBC. Used to seed the
    square-cell (SC) phase basin of Güngördü 2016.

    Parameters
    ----------
    nx, ny : int
        Lattice cell counts.
    a : float
        Lattice constant (m).
    period : float
        Square lattice constant (nearest-neighbour distance, m).

    Returns
    -------
    centers : numpy.ndarray(2d)
        Shape (n_centers, 2), site (x, y) coordinates (m).
    """
    if period <= 0.0:
        raise RuntimeError(
            f'period must be positive, got {period}.')
    L_x = nx * a
    L_y = ny * a
    n1_max = int(np.ceil(L_x / period)) + 2
    n2_max = int(np.ceil(L_y / period)) + 2
    centers = []
    for n1 in range(-2, n1_max + 1):
        for n2 in range(-2, n2_max + 1):
            r = np.array([n1 * period, n2 * period])
            if (-period <= r[0] <= L_x + period
                    and -period <= r[1] <= L_y + period):
                centers.append(r)
    return np.array(centers)


# ---------------------------------------------------------------------
def square_lattice_skyrmions(nx, ny, a, R, period, dw):
    """Square lattice of Neel skyrmions (each carries Q = -1).

    Square-lattice analogue of `hex_lattice_skyrmions`: each
    grid point picks up the profile of the nearest square-site
    skyrmion, with radial Neel in-plane winding. Seeds the
    four-fold-ordered square-cell (SC) basin of the Güngördü
    2016 phase diagram (the hex `sk_lattice` IC seeds the
    six-fold SkX basin instead).

    Parameters
    ----------
    nx, ny : int
        Lattice cell counts.
    a : float
        Lattice constant (m).
    R : float
        Skyrmion radius (m_z = 0 contour) in metres.
    period : float
        Square lattice constant in metres (typical: the helix
        wavelength lambda = 4 pi A / D).
    dw : float
        Domain-wall width (m), ~ sqrt(A_ex / K_eff).

    Returns
    -------
    m_top, m_bot : numpy.ndarray(3d)
        Shape (ny, nx, 3). Antiparallel SAF pair.
    """
    if R <= 0.0 or dw <= 0.0:
        raise RuntimeError(
            f'R and dw must be positive; got R={R}, dw={dw}.')
    centers = _square_centers(nx, ny, a, period)
    r, phi = _nearest_center_field(nx, ny, a, centers)
    theta = 2.0 * np.arctan(np.exp(-(r - R) / dw))
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    m_top = np.zeros((ny, nx, 3))
    m_top[..., 0] = sin_t * np.cos(phi)   # Neel winding
    m_top[..., 1] = sin_t * np.sin(phi)
    m_top[..., 2] = cos_t
    m_bot = -m_top
    return m_top, m_bot
