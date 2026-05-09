"""Phase classification of relaxed SAF spin textures.

Combines four order parameters to label a configuration as
one of five magnetic phases:

    FM+/-      ferromagnetic, single-domain
    iSk        isolated skyrmion (single localized core)
    SkX        skyrmion lattice (multi-skyrmion + 6-fold FFT)
    SS         spin spiral (Q~0, 2-fold FFT peaks)
    Lab        labyrinthine domains (Q~0, isotropic FFT ring)

The order parameters are:

    <m_z>   layer-averaged out-of-plane magnetization
    Q       topological charge of the top layer
    P_n(k*) angular harmonic content at the dominant FFT
            wavevector k* (n=2 and n=6 harmonics give SS vs
            SkX; uniform azimuthal power gives Lab)

Functions
---------
order_parameters
    Compute (<m_z>, Q, k_star, P_2, P_6, P_iso, peak_power).
classify
    Return the phase label given a relaxed (m_top, m_bot)
    pair and the parameters namespace.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
# Local
from src.simulator.main import topological_charge

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
# Decision-tree thresholds. Tuned against canonical
# textures (uniform FM, single skyrmion, SkX lattice, helix,
# random labyrinth) at 256x256.
_TH_FM_MZ = 0.95         # |<m_z>| above this means FM
_TH_ISK_Q_LO = 0.5       # iSk lower bound on |Q|
_TH_ISK_Q_HI = 1.5       # iSk upper bound on |Q|
_TH_HARMONIC_RATIO = 2.0 # min ratio P_n / P_iso for SkX/SS
_TH_PEAK_OVER_BG = 3.0   # min ratio peak/background to call
                         # ordering present in the FFT


def _radial_power(power, K_radius):
    """Bin the 2D FFT power spectrum into radial shells."""
    n_bins = max(power.shape) // 2
    k_max = K_radius.max()
    edges = np.linspace(0.0, k_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    radial = np.zeros(n_bins)
    counts = np.zeros(n_bins)
    bin_idx = np.clip(
        np.digitize(K_radius.ravel(), edges) - 1,
        0, n_bins - 1,
    )
    np.add.at(radial, bin_idx, power.ravel())
    np.add.at(counts, bin_idx, 1.0)
    counts = np.where(counts > 0, counts, 1.0)
    return centers, radial / counts


# ---------------------------------------------------------------------
def _angular_harmonics(power, K_radius, K_angle, k_star,
                       dk_grid, n_az=180, sigma_factor=1.5):
    """Sample angular power at |k|=k* and return its harmonics.

    Each FFT bin is weighted by a Gaussian
    `exp(-0.5 * ((|k| - k*) / sigma)**2)` with
    `sigma = sigma_factor * dk_grid`. The Gaussian width
    is tied to the FFT grid spacing so the ring sampling
    works on any lattice size.

    Returns
    -------
    P : numpy.ndarray(1d)
        Length-`n_az` weighted azimuthal power profile.
    P_iso : float
        Mean power on the ring.
    P_n : numpy.ndarray(1d)
        |FFT|^2 of the symmetrized azimuthal profile so
        that index n is the n-fold angular harmonic.
    """
    n_half = n_az // 2 + 1
    if k_star <= 0.0:
        return np.zeros(n_az), 0.0, np.zeros(n_half)
    # Gaussian ring weight (always nonzero)
    sigma = max(sigma_factor * dk_grid, 1e-30)
    w = np.exp(-0.5 * ((K_radius - k_star) / sigma) ** 2)
    weighted = power * w
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    angles = K_angle.ravel()
    weights = weighted.ravel()
    norm = w.ravel()
    bin_edges = np.linspace(-np.pi, np.pi, n_az + 1)
    P, _ = np.histogram(
        angles, bins=bin_edges, weights=weights,
    )
    W, _ = np.histogram(
        angles, bins=bin_edges, weights=norm,
    )
    safe_W = np.where(W > 0.0, W, 1.0)
    P = P / safe_W
    P_iso = float(np.mean(P))
    if P_iso <= 0.0:
        return P, 0.0, np.zeros(n_half)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Symmetrize antipodal bins: FFT power is centrosymmetric
    # (P(k)=P(-k)) so a 2-fold real-space stripe collapses
    # to 1-fold here, a 6-fold lattice to 3-fold. Convert
    # back to physical fold count by multiplying the
    # harmonic index by 2 in the decision tree.
    half = n_az // 2
    P_sym = 0.5 * (P[:half] + P[half:])
    F = np.fft.rfft(P_sym - P_sym.mean())
    P_n = np.abs(F) ** 2
    return P, P_iso, P_n


# ---------------------------------------------------------------------
def order_parameters(m_top, m_bot, p):
    """Compute the order parameters of a relaxed state.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins.
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins.
    p : SimpleNamespace
        Parameters namespace.

    Returns
    -------
    obs : dict
        Dictionary with keys:
            'mz_top', 'mz_bot' : layer-averaged m_z
            'Q'                : topological charge (top)
            'k_star'           : dominant in-plane wavevector
            'P_iso'            : mean azimuthal power at k*
            'P_2', 'P_6'       : 2-fold and 6-fold harmonics
            'peak_over_bg'     : peak-to-background ratio
    """
    mz_top = float(np.mean(m_top[..., 2]))
    mz_bot = float(np.mean(m_bot[..., 2]))
    Q = float(topological_charge(m_top, p.a))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # FFT power spectrum of m_z (top), zero-mean
    nx, ny = p.nx, p.ny
    mz = m_top[..., 2] - mz_top
    F = np.fft.fft2(mz)
    F = np.fft.fftshift(F)
    power = np.abs(F) ** 2
    kx = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(nx, d=p.a)
    )
    ky = 2.0 * np.pi * np.fft.fftshift(
        np.fft.fftfreq(ny, d=p.a)
    )
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K_r = np.sqrt(KX * KX + KY * KY)
    K_a = np.arctan2(KY, KX)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Radial peak (skip the DC bin)
    centers, radial = _radial_power(power, K_r)
    if len(centers) > 1:
        idx = int(np.argmax(radial[1:])) + 1
        k_star = float(centers[idx])
        peak_over_bg = float(
            radial[idx] / max(radial[1:].mean(), 1e-30)
        )
    else:
        k_star = 0.0
        peak_over_bg = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Azimuthal harmonics on the dominant ring. The ring
    # width is tied to the FFT grid spacing so coarse
    # lattices still pick up a meaningful sample.
    dk_grid = 2.0 * np.pi / (min(nx, ny) * p.a)
    _, P_iso, P_n = _angular_harmonics(
        power, K_r, K_a, k_star, dk_grid,
    )
    # After antipodal symmetrization of the azimuthal
    # profile, harmonic index n corresponds to physical
    # 2*n-fold angular order: n=1 -> stripes (2-fold),
    # n=3 -> hex skyrmion lattice (6-fold).
    P_2 = float(P_n[1]) if len(P_n) > 1 else 0.0
    P_6 = float(P_n[3]) if len(P_n) > 3 else 0.0
    return {
        'mz_top': mz_top, 'mz_bot': mz_bot, 'Q': Q,
        'k_star': k_star, 'P_iso': P_iso,
        'P_2': P_2, 'P_6': P_6,
        'peak_over_bg': peak_over_bg,
    }


# ---------------------------------------------------------------------
def classify(m_top, m_bot, p, obs=None):
    """Return the phase label for a relaxed SAF state.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins.
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins.
    p : SimpleNamespace
        Parameters namespace.
    obs : dict or None, default=None
        Optional pre-computed order parameters. If None,
        `order_parameters` is called internally.

    Returns
    -------
    label : {'FM+', 'FM-', 'iSk', 'SkX', 'SS', 'Lab',
              'undetermined'}
        Phase label.
    obs : dict
        Order parameters used for the decision.
    """
    if obs is None:
        obs = order_parameters(m_top, m_bot, p)
    mz = obs['mz_top']
    Q = obs['Q']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Ferromagnetic
    if abs(mz) > _TH_FM_MZ:
        return ('FM+' if mz > 0 else 'FM-'), obs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Single skyrmion
    if _TH_ISK_Q_LO <= abs(Q) <= _TH_ISK_Q_HI:
        return 'iSk', obs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 3. Periodic ordering: dominant in-plane wavevector
    has_order = obs['peak_over_bg'] > _TH_PEAK_OVER_BG
    P_iso = obs['P_iso']
    if has_order and P_iso > 0.0:
        ratio_2 = obs['P_2'] / P_iso
        ratio_6 = obs['P_6'] / P_iso
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # 6-fold dominant => skyrmion lattice
        if (ratio_6 > _TH_HARMONIC_RATIO
                and ratio_6 > ratio_2):
            return 'SkX', obs
        # 2-fold dominant => spin spiral
        if (ratio_2 > _TH_HARMONIC_RATIO
                and ratio_2 > ratio_6):
            return 'SS', obs
        # Periodic but no clean angular order => labyrinth
        return 'Lab', obs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. No ordering and not FM -> labyrinth or undetermined
    if abs(Q) > _TH_ISK_Q_HI:
        return 'SkX', obs
    if abs(Q) < _TH_ISK_Q_LO:
        return 'Lab', obs
    return 'undetermined', obs
