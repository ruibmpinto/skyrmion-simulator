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


def exchange_field(m, C_ex):
    """Compute the exchange effective field.

    Uses discrete Laplacian on a square lattice with PBC.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_ex : float
        Exchange prefactor 2*A_ex / (Ms * a^2) in Tesla.

    Returns
    -------
    H_ex : numpy.ndarray(3d)
        Exchange field in Tesla, shape (ny, nx, 3).

    Notes
    -----
    H_ex = C_ex * (m_+x + m_-x + m_+y + m_-y - 4*m)

    Loop vs roll form:

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
    m_px, m_mx, m_py, m_my = neighbors(m)

    return C_ex * (m_px + m_mx + m_py + m_my - 4.0 * m)


# ---------------------------------------------------------------------
def dmi_field(m, C_dmi):
    """Compute the interfacial DMI effective field.

    Neel-type DMI appropriate for Co/Pt interfaces.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_dmi : float
        DMI prefactor D / (Ms * a) in Tesla.

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
    # {+1, -1, +1, -1, 0}$: (centered) central-difference first derivative.
    m_px, m_mx, m_py, m_my = neighbors(m)
    H = np.zeros_like(m)
    # In-plane components couple to gradients of m_z.
    H[..., 0] = C_dmi * (m_px[..., 2] - m_mx[..., 2])
    H[..., 1] = C_dmi * (m_py[..., 2] - m_my[..., 2])
    # Out-of-plane component couples to the in-plane divergence.
    H[..., 2] = -C_dmi * (
        m_px[..., 0] - m_mx[..., 0]
        + m_py[..., 1] - m_my[..., 1])
    return H


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
def anisotropy_field(m, C_anis):
    """Compute perpendicular uniaxial anisotropy field.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    C_anis : float
        Anisotropy prefactor 2*K / Ms in Tesla.

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
                    H_ext, H_RKKY):
    """Compute total effective field for one layer.

    All contributions are in Tesla.

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

    Returns
    -------
    H_eff : numpy.ndarray(3d)
        Total effective field in Tesla,
        shape (ny, nx, 3).
    """
    shape = m.shape[:2]
    # Sum of micromagnetic contributions; demag must be added by caller
    # (energy.py adds it explicitly via demag_field).
    H = exchange_field(m, C_ex)
    H += dmi_field(m, C_dmi)
    H += anisotropy_field(m, C_anis)
    H += zeeman_field(H_ext, shape)
    H += rkky_field(m_other, H_RKKY)
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
def effective_field_demag_pair(m_top, m_bot, p, kernels):
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
    # Local-in-space terms
    H_top = exchange_field(m_top, p.C_ex)
    H_top += dmi_field(m_top, p.C_dmi)
    H_top += anisotropy_field(m_top, C_top_bare)
    H_top += zeeman_field(p.H_ext, shape)
    H_top += rkky_field(m_bot, p.H_RKKY)
    H_bot = exchange_field(m_bot, p.C_ex)
    H_bot += dmi_field(m_bot, p.C_dmi)
    H_bot += anisotropy_field(m_bot, C_bot_bare)
    H_bot += zeeman_field(p.H_ext, shape)
    H_bot += rkky_field(m_top, p.H_RKKY)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Long-range demag (one FFT pair per component per layer)
    H_dem_top, H_dem_bot = demag_field(m_top, m_bot, kernels)
    H_top += H_dem_top
    H_bot += H_dem_bot
    return H_top, H_bot
