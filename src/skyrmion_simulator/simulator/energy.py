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
effective_anisotropy
    Return the bare and effective anisotropies per layer
    (K, K_eff = K - mu0*Ms^2/2) plus their layer average.
critical_dmi
    Bogdanov-Hubert critical DMI D_c = 4*sqrt(A*K_eff)/pi.
pma_anisotropy_field
    Anisotropy field H_K = 2*K_eff/(mu0*Ms) in Tesla, used
    as the natural Zeeman scale.
total_energy
    Total energy of the SAF state in Joules.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from skyrmion_simulator.simulator.demag import demag_field
from skyrmion_simulator.simulator.fields import (
    _neighbors_with_bc,
    anisotropy_field,
    bare_anis_prefactors,
    dmi_field,
    exchange_field,
    rkky_field,
    zeeman_field,
)

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def effective_anisotropy(p):
    """Return bare and demag-corrected anisotropies (J/m^3).

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace exposing `K_top`, `K_bot`,
        `Ms`, and `mu0`.

    Returns
    -------
    out : dict
        Keys: `K_top`, `K_bot` (bare), `K_eff_top`,
        `K_eff_bot`, `K_eff_avg`, `mu0_Ms2_over_2`.
        `K_eff_layer = K_layer - 0.5 * mu0 * Ms^2`.
    """
    # Shape-anisotropy correction for an infinite thin film with PMA.
    half_mu0_Ms2 = 0.5 * p.mu0 * p.Ms * p.Ms
    K_eff_top = p.K_top - half_mu0_Ms2
    K_eff_bot = p.K_bot - half_mu0_Ms2
    return {
        'K_top': float(p.K_top),
        'K_bot': float(p.K_bot),
        'K_eff_top': float(K_eff_top),
        'K_eff_bot': float(K_eff_bot),
        'K_eff_avg': float(0.5 * (K_eff_top + K_eff_bot)),
        'mu0_Ms2_over_2': float(half_mu0_Ms2),
    }


# ---------------------------------------------------------------------
def critical_dmi(p):
    """Bogdanov-Hubert critical DMI strength (J/m^2).

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace.

    Returns
    -------
    D_c : float
        D_c = 4 * sqrt(A_ex * K_eff_avg) / pi, where
        K_eff_avg is the layer-averaged effective
        anisotropy. Raises `RuntimeError` if K_eff_avg <= 0
        (no PMA FM to destabilize; D_c is undefined).

    Notes
    -----
    Marks the threshold at which the uniform PMA FM state
    becomes unstable to spontaneous helical winding.
    Below D_c the FM is the stable ground state in the
    absence of demag spatial structure; above D_c
    spirals / SkX / iSk become competitive.
    """
    K_eff_avg = effective_anisotropy(p)['K_eff_avg']
    if K_eff_avg <= 0.0:
        raise RuntimeError(
            f'K_eff_avg = {K_eff_avg:.3e} J/m^3 <= 0; '
            f'D_c is undefined (easy-plane regime, no PMA '
            f'FM to destabilize).')
    # Bogdanov-Hubert critical DMI for FM to spiral instability.
    return 4.0 * np.sqrt(p.A_ex * K_eff_avg) / np.pi


# ---------------------------------------------------------------------
def pma_anisotropy_field(p):
    """Perpendicular Magnetic Anisotropy (PMA):
    Anisotropy field H_K = 2 K_eff / Ms in Tesla.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace.

    Returns
    -------
    H_K : float
        Anisotropy field in Tesla, using the layer-averaged
        effective anisotropy. Raises `RuntimeError` if
        `K_eff_avg <= 0`.

    Notes
    -----
    The textbook anisotropy field is H_K = 2 K_eff /
    (mu0 Ms) in A/m. The simulator carries every H in
    Tesla, so H_K_T = mu0 * H_K_(A/m) = 2 K_eff / Ms.
    """
    K_eff_avg = effective_anisotropy(p)['K_eff_avg']
    if K_eff_avg <= 0.0:
        raise RuntimeError(
            f'K_eff_avg = {K_eff_avg:.3e} J/m^3 <= 0; '
            f'H_K is undefined.')
    # H_K in Tesla: simulator carries every H in T, so absorb mu0.
    return 2.0 * K_eff_avg / p.Ms


# ---------------------------------------------------------------------
def total_energy(m_top, m_bot, p, kernels, mask):
    """Compute the total magnetic energy of the SAF state.

    Used by the phase diagram sweep and as the relax dE
    convergence gate. Every term is assembled exactly like the
    RHS (`fields.effective_field_demag_pair`): ghost-cell
    exchange/DMI at free boundaries, free-y for the racetrack
    kernel kind, masked local terms, and demag sourced by the
    mask-zeroed magnetization -- so E is the Lyapunov function
    of the damped dynamics.

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
    mask : {numpy.ndarray(2d), None}
        Boolean (ny, nx) array, True inside the magnetic
        region; None is the fully periodic path.

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
    # Per-site magnetic volume (one layer); top + bot doubles it in sum.
    V_cell = p.t_Co * p.a * p.a
    Ms = p.Ms
    # Use bare K so anisotropy and demag are not double-counted.
    C_top, C_bot = bare_anis_prefactors(p)
    shape = m_top.shape[:2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary conditions, identical to the RHS assembly.
    # Racetrack kernel kind opens the y edges (free-y ghosts).
    free_y = (kernels.get('kind') == 'racetrack')
    # Shared RT 2013 Eq. (6) ghost cells for exchange + DMI.
    if (mask is not None or free_y) and float(p.C_ex) > 0.0:
        # xi_inv_a = a/xi = C_dmi/C_ex (Bogdanov-Roesler tilt).
        xi_inv_a = float(p.C_dmi) / float(p.C_ex)
        nbrs_top = _neighbors_with_bc(
            m_top, mask=mask, xi_inv_a=xi_inv_a, free_y=free_y)
        nbrs_bot = _neighbors_with_bc(
            m_bot, mask=mask, xi_inv_a=xi_inv_a, free_y=free_y)
    else:
        nbrs_top = None
        nbrs_bot = None
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Internal (bilinear) fields, both layers
    H_int_top = exchange_field(
        m_top, p.C_ex, mask=mask, neighbors_eff=nbrs_top)
    H_int_top += dmi_field(
        m_top, p.C_dmi, mask=mask, neighbors_eff=nbrs_top)
    H_int_top += anisotropy_field(m_top, C_top, mask=mask)
    # RKKY on top is driven by m_bot.
    H_rk_top = rkky_field(m_bot, p.H_RKKY)
    H_int_bot = exchange_field(
        m_bot, p.C_ex, mask=mask, neighbors_eff=nbrs_bot)
    H_int_bot += dmi_field(
        m_bot, p.C_dmi, mask=mask, neighbors_eff=nbrs_bot)
    H_int_bot += anisotropy_field(m_bot, C_bot, mask=mask)
    # RKKY on bottom is driven by m_top.
    H_rk_bot = rkky_field(m_top, p.H_RKKY)
    if mask is not None:
        # Vacuum cells carry no RKKY coupling.
        H_rk_top = H_rk_top * mask[..., np.newaxis]
        H_rk_bot = H_rk_bot * mask[..., np.newaxis]
    H_int_top += H_rk_top
    H_int_bot += H_rk_bot
    # Demag: self + inter-layer via FFT kernels, sourced by the
    # mask-zeroed magnetization (vacuum contributes no charge).
    if mask is not None:
        m_top_dem = m_top * mask[..., np.newaxis]
        m_bot_dem = m_bot * mask[..., np.newaxis]
    else:
        m_top_dem = m_top
        m_bot_dem = m_bot
    H_dem_top, H_dem_bot = demag_field(m_top_dem, m_bot_dem, kernels)
    if mask is not None:
        H_dem_top = H_dem_top * mask[..., np.newaxis]
        H_dem_bot = H_dem_bot * mask[..., np.newaxis]
    H_int_top += H_dem_top
    H_int_bot += H_dem_bot
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bilinear (self) energy with 0.5 prefactor
    dot_int = (np.sum(m_top * H_int_top) + np.sum(m_bot * H_int_bot))
    # 0.5 prevents double counting bilinear self-interactions.
    E_internal = -0.5 * V_cell * Ms * dot_int
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Zeeman energy (no 0.5; field is external)
    H_zee = zeeman_field(p.H_ext, shape)
    if mask is not None:
        # Vacuum cells see no Zeeman energy.
        H_zee = H_zee * mask[..., np.newaxis]
    dot_zee = (np.sum(m_top * H_zee) + np.sum(m_bot * H_zee))
    # No 0.5: external field is independent of m, so no double counting.
    E_zeeman = -V_cell * Ms * dot_zee
    # Return
    return float(E_internal + E_zeeman)
