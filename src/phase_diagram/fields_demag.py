"""Demag-aware effective field for both SAF layers.

Composes the existing per-term field callables from
`src.simulator.fields` with the new Fourier-space demag
field from `src.simulator.demag`. The anisotropy term is
recomputed locally with the **bare** prefactor 2*K/Ms (in
Tesla) so that the thin-film K_eff correction baked into
`p.C_anis_top/_bot` is removed exactly once and replaced by
the explicit demag contribution.

Functions
---------
effective_field_demag_pair
    Compute (H_top, H_bot) including all five micromagnetic
    terms plus the magnetostatic field.
"""
#
#                                                                Modules
# =====================================================================
# Local
from src.simulator.demag import demag_field
from src.simulator.energy import bare_anis_prefactors
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


def effective_field_demag_pair(m_top, m_bot, p, kernels):
    """Compute the total effective field for both layers.

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
