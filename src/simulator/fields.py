"""Effective magnetic field contributions for the LLGS equation.

Computes exchange, DMI, Zeeman, anisotropy, and RKKY fields
on a 2D square lattice with periodic boundary conditions.
All fields are returned in Tesla.

Functions
---------
exchange_field
    Discrete Laplacian exchange field.
dmi_field
    Interfacial (Neel-type) DMI field.
zeeman_field
    Uniform external field.
anisotropy_field
    Perpendicular uniaxial anisotropy field.
rkky_field
    Antiferromagnetic interlayer coupling field.
effective_field
    Total effective field for one layer (local-K_eff path; no
    explicit demag).
effective_field_demag_pair
    Total effective field for both layers including the
    long-range demag contribution (bare-K path).
bare_anis_prefactors
    Return the bare 2*K/Ms anisotropy prefactors used by the
    bare-K demag path.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.simulator.demag import demag_field
from src.simulator.lattice import neighbors

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def _neighbors_with_bc(m, mask=None, xi_inv_a=0.0):
    """Return the four nearest-neighbour arrays with optional
    free-BC boundary handling, matching mumax3's exchange+DMI
    edge convention.

    `mask=None` -> straight PBC neighbours (np.roll), same as
    `src.simulator.lattice.neighbors`. With a Boolean (ny, nx)
    mask, ghost cells in each missing direction follow the
    exact formulas in `mumax3/cuda/dmi.cu` (which cites
    Bogdanov-Roesler PRL 87, 037203 (2001)):

        +x missing:
            m_g = m + (a/xi) (-m_z, 0, +m_x)
        -x missing:
            m_g = m + (a/xi) (+m_z, 0, -m_x)
        +y missing:
            m_g = m + (a/xi) (0, -m_z, +m_y)
        -y missing:
            m_g = m + (a/xi) (0, +m_z, -m_y)

    where xi = 2A/D and xi_inv_a = a/xi = C_dmi/C_ex. The
    Bogdanov-Roesler boundary-tilt direction is opposite in
    sign to the RT 2013 Eq. (6) form
    `dm/dn = (1/xi)(zhat x n_out) x m`; the two conventions
    differ by a sign in the BC formula but produce the same
    skyrmion chirality at the same |D| in the bulk (the bulk
    DMI energy is invariant under simultaneous BC- and
    chirality-flip). mumax3's choice is what makes the
    CO 2018 Sci. Data 5:170243 reference simulations match
    its r_sk(D) values exactly.

    Setting `xi_inv_a = 0` reduces to plain Neumann
    `m_ghost = m_center` (forward-extrapolation), the
    natural limit when no DMI is present.
    """
    m_px, m_mx, m_py, m_my = neighbors(m)
    if mask is None:
        return m_px, m_mx, m_py, m_my
    # Indicator of "neighbour is inside the mask" for each
    # direction (Boolean, shape (ny, nx, 1) for broadcasting).
    mask_px = np.roll(mask, -1, axis=1)[..., np.newaxis]
    mask_mx = np.roll(mask, +1, axis=1)[..., np.newaxis]
    mask_py = np.roll(mask, -1, axis=0)[..., np.newaxis]
    mask_my = np.roll(mask, +1, axis=0)[..., np.newaxis]
    xia = float(xi_inv_a)
    g_px = np.empty_like(m)
    g_px[..., 0] = m[..., 0] - xia * m[..., 2]
    g_px[..., 1] = m[..., 1]
    g_px[..., 2] = m[..., 2] + xia * m[..., 0]
    g_mx = np.empty_like(m)
    g_mx[..., 0] = m[..., 0] + xia * m[..., 2]
    g_mx[..., 1] = m[..., 1]
    g_mx[..., 2] = m[..., 2] - xia * m[..., 0]
    g_py = np.empty_like(m)
    g_py[..., 0] = m[..., 0]
    g_py[..., 1] = m[..., 1] - xia * m[..., 2]
    g_py[..., 2] = m[..., 2] + xia * m[..., 1]
    g_my = np.empty_like(m)
    g_my[..., 0] = m[..., 0]
    g_my[..., 1] = m[..., 1] + xia * m[..., 2]
    g_my[..., 2] = m[..., 2] - xia * m[..., 1]
    m_px_eff = np.where(mask_px, m_px, g_px)
    m_mx_eff = np.where(mask_mx, m_mx, g_mx)
    m_py_eff = np.where(mask_py, m_py, g_py)
    m_my_eff = np.where(mask_my, m_my, g_my)
    return m_px_eff, m_mx_eff, m_py_eff, m_my_eff


def exchange_field(m, C_ex, mask=None, neighbors_eff=None):
    """Compute the exchange effective field.

    Uses discrete Laplacian on a square lattice. With
    `mask=None` (default) the boundary condition is periodic
    (PBC, via `np.roll`); with `mask` a Boolean (ny, nx) array
    the boundary condition is Neumann at the mask boundary
    (mumax3 convention: missing neighbours are replaced by the
    cell's own value, so the contribution from a missing side
    is zero), and the field is zero outside the masked region.
    Pass `neighbors_eff` (a 4-tuple `(m_+x, m_-x, m_+y, m_-y)`
    pre-computed via `_neighbors_with_bc`) to share the RT-
    extrapolated ghost cells across exchange + DMI builders.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_ex : float
        Exchange prefactor 2*A_ex / (Ms * a^2) in Tesla.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array. True inside the magnetic
        region, False outside (vacuum). None preserves the
        original PBC behaviour exactly.
    neighbors_eff : {tuple of 4 ndarrays, None}, default=None
        Optional pre-computed neighbour arrays. When given,
        these are used in place of the internal `np.roll`
        call; the `mask` is still applied to zero the field
        outside the magnetic region.

    Returns
    -------
    H_ex : numpy.ndarray(3d)
        Exchange field in Tesla, shape (ny, nx, 3).

    Notes
    -----
    PBC: H_ex = C_ex * (m_+x + m_-x + m_+y + m_-y - 4*m).
    Free-BC: H_ex = C_ex * sum_{neighbours inside mask}
                            (m_neighbour - m_cell),
    zeroed for cells outside the mask. With non-zero DMI the
    `neighbors_eff` tuple should carry the Rohart-Thiaville
    (2013) Eq. (6) extrapolation; with zero DMI plain
    Neumann (ghost = own cell) suffices.

    Loop vs roll form for the PBC path:

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
                - 4 * m[i, j]
            )

    vs.

    # Roll form (vectorized, fast)
    m_px, m_mx, m_py, m_my = neighbors(m)
    laplacian = m_px + m_mx + m_py + m_my - 4 * m
    """
    # 5-point stencil discrete Laplacian on the square lattice.
    # {1, 1, 1, 1, -4}: finite-difference Laplacian stencil weights.
    if neighbors_eff is None:
        # Build the effective neighbours: PBC (mask=None) or
        # plain Neumann (DMI BC off, xi_inv_a=0).
        m_px, m_mx, m_py, m_my = _neighbors_with_bc(
            m, mask=mask, xi_inv_a=0.0)
    else:
        m_px, m_mx, m_py, m_my = neighbors_eff
    if mask is None:
        # PBC retained byte-exactly.
        return C_ex * (m_px + m_mx + m_py + m_my - 4.0 * m)
    lapl = m_px + m_mx + m_py + m_my - 4.0 * m
    # Cells outside the mask are vacuum: no field.
    return (C_ex * lapl) * mask[..., np.newaxis]


# ---------------------------------------------------------------------
def _dmi_edge_correction_mumax3(m, mask, C_dmi):
    """mumax3 / Cnv-extension RT 2013 boundary correction.

    At every boundary cell (a cell whose neighbour in some
    direction is outside the magnetic mask), adds the term

        H_BC = (2D / (mu_0 M_s a)) sum_n_missing
                   (zhat x n_out) x m_center

    discretised here in Tesla as `2*C_dmi*(zhat x n_out) x m`.
    Returned values are zero in the interior (no missing
    neighbour) and inside-the-mask cells with all neighbours
    present.

    Per-direction (ẑ × n̂_out) × m components:
        +x edge (n_out = +xhat):  (m_z, 0, -m_x)
        -x edge (n_out = -xhat): (-m_z, 0,  m_x)
        +y edge (n_out = +yhat): (0, m_z, -m_y) since
                                    (zhat x yhat) x m
                                    = -xhat x m
                                    = (0,  m_z, -m_y)
        -y edge (n_out = -yhat): (0, -m_z,  m_y)
    """
    # Each mask_* indicates whether the *neighbour* in that
    # direction is inside the magnetic region (Boolean (ny, nx)).
    mask_px = np.roll(mask, -1, axis=1)
    mask_mx = np.roll(mask, +1, axis=1)
    mask_py = np.roll(mask, -1, axis=0)
    mask_my = np.roll(mask, +1, axis=0)
    # Boundary cells: cell is inside the mask AND the
    # respective neighbour is outside.
    bx_px = mask & (~mask_px)
    bx_mx = mask & (~mask_mx)
    bx_py = mask & (~mask_py)
    bx_my = mask & (~mask_my)
    two_C = 2.0 * float(C_dmi)
    H = np.zeros_like(m)
    # +x edge contribution: (m_z, 0, -m_x).
    H[..., 0] += two_C * (m[..., 2] * bx_px)
    H[..., 2] += two_C * (-m[..., 0] * bx_px)
    # -x edge contribution: (-m_z, 0, m_x).
    H[..., 0] += two_C * (-m[..., 2] * bx_mx)
    H[..., 2] += two_C * (m[..., 0] * bx_mx)
    # +y edge contribution: (0, m_z, -m_y).
    H[..., 1] += two_C * (m[..., 2] * bx_py)
    H[..., 2] += two_C * (-m[..., 1] * bx_py)
    # -y edge contribution: (0, -m_z, m_y).
    H[..., 1] += two_C * (-m[..., 2] * bx_my)
    H[..., 2] += two_C * (m[..., 1] * bx_my)
    return H


def dmi_field(m, C_dmi, mask=None, neighbors_eff=None):
    """Compute the interfacial DMI effective field.

    Neel-type DMI appropriate for Co/Pt interfaces. With
    `mask=None` (default) the central-difference stencil uses
    PBC neighbours; with `mask` a Boolean (ny, nx) array,
    out-of-mask neighbours are replaced by Rohart-Thiaville
    (2013) Eq. (6) extrapolated ghost cells (or own-cell
    Neumann ghosts when `neighbors_eff` is built with
    `xi_inv_a = 0`) and the field is zero outside the mask.
    Sharing the `neighbors_eff` tuple with `exchange_field`
    ensures the exchange and DMI builders see the same
    ghost-cell extrapolation, which is required by the RT
    boundary condition (derived from the joint
    exchange + DMI variation).

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_dmi : float
        DMI prefactor D / (Ms * a) in Tesla.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array. True inside the magnetic
        region. None preserves the original PBC behaviour.
    neighbors_eff : {tuple of 4 ndarrays, None}, default=None
        Optional pre-computed neighbour arrays
        `(m_+x, m_-x, m_+y, m_-y)` from
        `_neighbors_with_bc(m, mask, xi_inv_a)`. When given,
        these replace the internal `np.roll` call.

    Returns
    -------
    H_dmi : numpy.ndarray(3d)
        DMI field in Tesla, shape (ny, nx, 3).

    Notes
    -----
    Derived from interfacial DMI energy density:
        e_DMI = D [m_z div(m) - (m . grad) m_z]

    Discrete effective field:
        H_x =  C * (m_+x_z - m_-x_z)
        H_y =  C * (m_+y_z - m_-y_z)
        H_z = -C * (m_+x_x - m_-x_x + m_+y_y - m_-y_y)

    The implementation uses the same roll-vs-loop pattern as
    exchange_field, applied to a central-difference stencil
    instead of a Laplacian:

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
                - 4 * m[i, j]
            )

    vs.

    # Roll form (vectorized, fast)
    m_px, m_mx, m_py, m_my = neighbors(m)
    laplacian = m_px + m_mx + m_py + m_my - 4 * m
    """
    # Central-difference discretization of the interfacial DMI field.
    # {+1, -1, +1, -1, 0}: (centered) central-difference first derivative.
    if neighbors_eff is None:
        m_px, m_mx, m_py, m_my = _neighbors_with_bc(
            m, mask=mask, xi_inv_a=0.0)
    else:
        m_px, m_mx, m_py, m_my = neighbors_eff
    if mask is None:
        H = np.zeros_like(m)
        H[..., 0] = C_dmi * (m_px[..., 2] - m_mx[..., 2])
        H[..., 1] = C_dmi * (m_py[..., 2] - m_my[..., 2])
        H[..., 2] = -C_dmi * (
            m_px[..., 0] - m_mx[..., 0]
            + m_py[..., 1] - m_my[..., 1])
        return H
    H = np.zeros_like(m)
    H[..., 0] = C_dmi * (m_px[..., 2] - m_mx[..., 2])
    H[..., 1] = C_dmi * (m_py[..., 2] - m_my[..., 2])
    H[..., 2] = -C_dmi * (
        m_px[..., 0] - m_mx[..., 0]
        + m_py[..., 1] - m_my[..., 1])
    return H * mask[..., np.newaxis]


# ---------------------------------------------------------------------
def zeeman_field(H_ext, shape):
    """Return the external field broadcast to lattice shape.

    Parameters
    ----------
    H_ext : numpy.ndarray(1d)
        External field in Tesla, shape (3,).
    shape : tuple
        Lattice shape (ny, nx).

    Returns
    -------
    H_z : numpy.ndarray(3d)
        Zeeman field in Tesla, shape (ny, nx, 3).
    """
    # Spatially uniform field broadcast to lattice shape.
    H = np.empty((*shape, 3))
    H[..., :] = H_ext[np.newaxis, np.newaxis, :]
    return H


# ---------------------------------------------------------------------
def anisotropy_field(m, C_anis, mask=None):
    """Compute perpendicular uniaxial anisotropy field.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_anis : float
        Anisotropy prefactor 2*K / Ms in Tesla.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array. True inside the magnetic
        region. None preserves the original behaviour.

    Returns
    -------
    H_anis : numpy.ndarray(3d)
        Anisotropy field in Tesla, shape (ny, nx, 3).

    Notes
    -----
    H_anis = C_anis * m_z * z_hat
    """
    # Perpendicular uniaxial K: only the z-component couples to m_z.
    H = np.zeros_like(m)
    H[..., 2] = C_anis * m[..., 2]
    if mask is not None:
        H *= mask[..., np.newaxis]
    return H


# ---------------------------------------------------------------------
def rkky_field(m_other, H_RKKY):
    """Compute RKKY interlayer coupling field.

    Antiferromagnetic coupling: the field on one layer is
    proportional to the negative magnetization of the other.

    Parameters
    ----------
    m_other : numpy.ndarray(3d)
        Spin configuration of the other layer, shape (ny, nx, 3).
    H_RKKY : float
        RKKY field magnitude in Tesla.

    Returns
    -------
    H_rkky : numpy.ndarray(3d)
        RKKY field in Tesla, shape (ny, nx, 3).

    Notes
    -----
    H_RKKY_on_this = -H_RKKY * m_other
    Promotes antiparallel alignment between layers.
    """
    # Negative sign drives antiparallel alignment between layers.
    return -H_RKKY * m_other


# ---------------------------------------------------------------------
def effective_field(m, m_other, C_ex, C_dmi, C_anis,
                    H_ext, H_RKKY, mask=None):
    """Compute total effective field for one layer.

    All contributions are in Tesla. `mask=None` (default)
    keeps the original PBC behaviour; a Boolean (ny, nx)
    `mask` activates the free-BC Neumann path for exchange,
    DMI, and anisotropy (mumax3 convention), zeroes the
    Zeeman and RKKY contributions outside the mask, and is
    forwarded to the per-term builders.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration of this layer,
        shape (ny, nx, 3).
    m_other : numpy.ndarray(3d)
        Spin configuration of the other layer.
    C_ex : float
        Exchange prefactor in Tesla.
    C_dmi : float
        DMI prefactor in Tesla.
    C_anis : float
        Anisotropy prefactor in Tesla.
    H_ext : numpy.ndarray(1d)
        External field in Tesla, shape (3,).
    H_RKKY : float
        RKKY field in Tesla.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array. True inside the magnetic
        region.

    Returns
    -------
    H_eff : numpy.ndarray(3d)
        Total effective field in Tesla,
        shape (ny, nx, 3).
    """
    shape = m.shape[:2]
    # Sum of micromagnetic contributions; demag must be added by caller
    # (energy.py adds it explicitly via demag_field).
    if mask is not None and float(C_ex) > 0.0:
        # mumax3 / Bogdanov-Roesler BC formula: forward-
        # extrapolation from the boundary cell with
        # xi_inv_a = a/xi = 0.5 * a * D/A = C_dmi/C_ex.
        # See `_neighbors_with_bc` docstring.
        xi_inv_a = float(C_dmi) / float(C_ex)
        neighbors_eff = _neighbors_with_bc(
            m, mask=mask, xi_inv_a=xi_inv_a)
    else:
        neighbors_eff = None
    H = exchange_field(
        m, C_ex, mask=mask, neighbors_eff=neighbors_eff)
    H += dmi_field(
        m, C_dmi, mask=mask, neighbors_eff=neighbors_eff)
    H += anisotropy_field(m, C_anis, mask=mask)
    H_ze = zeeman_field(H_ext, shape)
    H_rk = rkky_field(m_other, H_RKKY)
    if mask is not None:
        # Zeeman and RKKY have no spatial derivative, so the
        # only mask effect is to zero them outside the
        # magnetic region.
        m_mask = mask[..., np.newaxis]
        H_ze = H_ze * m_mask
        H_rk = H_rk * m_mask
    H += H_ze + H_rk
    return H


# ---------------------------------------------------------------------
def bare_anis_prefactors(p):
    """Return the bare 2*K/Ms anisotropy prefactors.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace exposing `K_top`, `K_bot`,
        and `Ms`.

    Returns
    -------
    C_top : float
        Bare anisotropy prefactor for the top layer in Tesla.
    C_bot : float
        Bare anisotropy prefactor for the bottom layer in Tesla.

    Notes
    -----
    `parameters._precompute()` stores `C_anis = 2*K/Ms -
    mu0*Ms`, where the second term is the thin-film demag
    correction (K_eff convention). When demag is computed
    explicitly we must use this function to get the bare K only.
    """
    # No K_eff correction: caller adds the demag field explicitly.
    inv_Ms = 1.0 / p.Ms
    C_top = 2.0 * p.K_top * inv_Ms
    C_bot = 2.0 * p.K_bot * inv_Ms
    return C_top, C_bot


# ---------------------------------------------------------------------
def effective_field_demag_pair(m_top, m_bot, p, kernels, mask=None):
    """Total effective field for both layers including the
    long-range demag contribution.

    Composes the per-term builders above with the Fourier-space
    demag field from `src.simulator.demag`. The anisotropy term
    uses the **bare** prefactor 2*K/Ms (in Tesla) so that the
    thin-film K_eff correction baked into `p.C_anis_top/_bot` is
    removed exactly once and replaced by the explicit demag
    contribution.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    p : SimpleNamespace
        Parameters namespace.
    kernels : dict
        Demag kernels from `precompute_demag_kernels(p)`.
    mask : {numpy.ndarray(2d), None}, default=None
        Boolean (ny, nx) array. True inside the magnetic
        region. None preserves the original PBC behaviour.
        When set, the local per-term builders use the Neumann
        free-BC variant and the demag is zeroed outside the
        mask. Pair with a free-BC demag kernel
        (`kind='newell_freebc'`) for the strict free-BC
        result.

    Returns
    -------
    H_top : numpy.ndarray(3d)
        Effective field on the top layer in Tesla.
    H_bot : numpy.ndarray(3d)
        Effective field on the bottom layer in Tesla.

    Notes
    -----
    H = H_exchange + H_DMI + H_anisotropy(bare K)
        + H_Zeeman + H_RKKY + H_demag.
    """
    C_top_bare, C_bot_bare = bare_anis_prefactors(p)
    shape = m_top.shape[:2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Local-in-space terms. With a free-BC mask and non-zero
    # DMI, compute shared RT 2013 Eq. (6) ghost cells once
    # per layer and forward to exchange + DMI.
    if mask is not None and float(p.C_ex) > 0.0:
        # mumax3 / Bogdanov-Roesler BC (see effective_field
        # above): xi_inv_a = a/xi = C_dmi/C_ex.
        xi_inv_a = float(p.C_dmi) / float(p.C_ex)
        nbrs_top = _neighbors_with_bc(
            m_top, mask=mask, xi_inv_a=xi_inv_a)
        nbrs_bot = _neighbors_with_bc(
            m_bot, mask=mask, xi_inv_a=xi_inv_a)
    else:
        nbrs_top = None
        nbrs_bot = None
    H_top = exchange_field(
        m_top, p.C_ex, mask=mask, neighbors_eff=nbrs_top)
    H_top += dmi_field(
        m_top, p.C_dmi, mask=mask, neighbors_eff=nbrs_top)
    H_top += anisotropy_field(m_top, C_top_bare, mask=mask)
    H_ze_t = zeeman_field(p.H_ext, shape)
    H_rk_t = rkky_field(m_bot, p.H_RKKY)
    H_bot = exchange_field(
        m_bot, p.C_ex, mask=mask, neighbors_eff=nbrs_bot)
    H_bot += dmi_field(
        m_bot, p.C_dmi, mask=mask, neighbors_eff=nbrs_bot)
    H_bot += anisotropy_field(m_bot, C_bot_bare, mask=mask)
    H_ze_b = zeeman_field(p.H_ext, shape)
    H_rk_b = rkky_field(m_top, p.H_RKKY)
    if mask is not None:
        m_mask = mask[..., np.newaxis]
        H_ze_t = H_ze_t * m_mask
        H_ze_b = H_ze_b * m_mask
        H_rk_t = H_rk_t * m_mask
        H_rk_b = H_rk_b * m_mask
    H_top += H_ze_t + H_rk_t
    H_bot += H_ze_b + H_rk_b
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Long-range demag (one FFT pair per component per layer).
    # With a mask, the cells outside the magnetic region are
    # vacuum, not "dummy" spins; they must contribute zero
    # source to the convolution. We mask m before the demag
    # call (the field on the physical cells is then computed
    # correctly), and zero the returned field outside the
    # mask for completeness.
    if mask is not None:
        m_mask = mask[..., np.newaxis]
        m_top_dem = m_top * m_mask
        m_bot_dem = m_bot * m_mask
    else:
        m_top_dem = m_top
        m_bot_dem = m_bot
    H_dem_top, H_dem_bot = demag_field(
        m_top_dem, m_bot_dem, kernels)
    if mask is not None:
        H_dem_top = H_dem_top * mask[..., np.newaxis]
        H_dem_bot = H_dem_bot * mask[..., np.newaxis]
    H_top += H_dem_top
    H_bot += H_dem_bot
    return H_top, H_bot
