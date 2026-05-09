"""Total micromagnetic energy for the SAF stack.

Computes the T=0 energy of the two-layer system summed
over both Co layers:

    E = -0.5 * V * Ms * sum_layers sum_sites m . H_internal
        -       V * Ms * sum_layers sum_sites m . H_zeeman

where V = t_Co * a^2 is the per-site volume (one layer)
and H_internal = H_exchange + H_DMI + H_anisotropy(bare K)
+ H_RKKY + H_demag. Fields are in Tesla, so the magnetic
constant mu0 is already absorbed into H and does not
appear explicitly in the formula. The 0.5 prefactor
prevents double counting of bilinear self-interactions.

Functions
---------
bare_anis_prefactors
    Return the bare 2*K/Ms anisotropy prefactors for both
    layers (without the thin-film K_eff correction).
total_energy
    Total energy of the SAF state in Joules.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.simulator.demag import demag_field
from src.simulator.fields import (
    anisotropy_field,
    dmi_field,
    exchange_field,
    rkky_field,
    zeeman_field,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


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
        Bare anisotropy prefactor for the top layer in
        Tesla.
    C_bot : float
        Bare anisotropy prefactor for the bottom layer in
        Tesla.

    Notes
    -----
    `parameters._precompute()` stores `C_anis = 2*K/Ms -
    mu0*Ms`, where the second term is the thin-film demag
    correction (K_eff convention). When demag is computed
    explicitly we must use bare K only.
    """
    inv_Ms = 1.0 / p.Ms
    C_top = 2.0 * p.K_top * inv_Ms
    C_bot = 2.0 * p.K_bot * inv_Ms
    return C_top, C_bot


# ---------------------------------------------------------------------
def total_energy(m_top, m_bot, p, kernels):
    """Compute the total magnetic energy of the SAF state.

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
    E : float
        Total magnetic energy in Joules.

    Notes
    -----
    The Zeeman contribution is summed without the 1/2
    prefactor because the external field is independent of
    the system's magnetization.
    """
    V_cell = p.t_Co * p.a * p.a
    Ms = p.Ms
    C_top, C_bot = bare_anis_prefactors(p)
    shape = m_top.shape[:2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Internal (bilinear) fields, both layers
    H_int_top = exchange_field(m_top, p.C_ex)
    H_int_top += dmi_field(m_top, p.C_dmi)
    H_int_top += anisotropy_field(m_top, C_top)
    H_int_top += rkky_field(m_bot, p.H_RKKY)
    H_int_bot = exchange_field(m_bot, p.C_ex)
    H_int_bot += dmi_field(m_bot, p.C_dmi)
    H_int_bot += anisotropy_field(m_bot, C_bot)
    H_int_bot += rkky_field(m_top, p.H_RKKY)
    H_dem_top, H_dem_bot = demag_field(m_top, m_bot, kernels)
    H_int_top += H_dem_top
    H_int_bot += H_dem_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bilinear (self) energy with 0.5 prefactor
    dot_int = (
        np.sum(m_top * H_int_top) + np.sum(m_bot * H_int_bot)
    )
    E_internal = -0.5 * V_cell * Ms * dot_int
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Zeeman energy (no 0.5; field is external)
    H_zee = zeeman_field(p.H_ext, shape)
    dot_zee = (
        np.sum(m_top * H_zee) + np.sum(m_bot * H_zee)
    )
    E_zeeman = -V_cell * Ms * dot_zee
    return float(E_internal + E_zeeman)
