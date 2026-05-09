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
    Total effective field for one layer.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.lattice import neighbors

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
    """
    m_px, m_mx, m_py, m_my = neighbors(m)
    return C_ex * (
        m_px + m_mx + m_py + m_my - 4.0 * m
    )


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
    """
    m_px, m_mx, m_py, m_my = neighbors(m)
    H = np.zeros_like(m)
    H[..., 0] = C_dmi * (
        m_px[..., 2] - m_mx[..., 2]
    )
    H[..., 1] = C_dmi * (
        m_py[..., 2] - m_my[..., 2]
    )
    H[..., 2] = -C_dmi * (
        m_px[..., 0] - m_mx[..., 0]
        + m_py[..., 1] - m_my[..., 1]
    )
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
        Spin configuration of the other layer,
        shape (ny, nx, 3).
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
    H = exchange_field(m, C_ex)
    H += dmi_field(m, C_dmi)
    H += anisotropy_field(m, C_anis)
    H += zeeman_field(H_ext, shape)
    H += rkky_field(m_other, H_RKKY)
    return H
