"""Phase classification of relaxed SAF spin textures.

Combines five order parameters to label a configuration as
one of seven magnetic phases:

    FM_anti    antiparallel SAF ferromagnet
               (m_top * m_bot ~ -1; invisible to H_z)
    FM_par+    parallel ferromagnet aligned with +z
    FM_par-    parallel ferromagnet aligned with -z
    iSk        isolated skyrmion (one or a few cores)
    SkX        skyrmion lattice (6-fold FFT *and* nonzero Q
               per principal period)
    BX         bubble lattice (6-fold FFT but Q~0 per
               period; topologically trivial)
    SS         spin spiral (2-fold FFT peaks)
    Lab        labyrinthine domains (isotropic FFT ring or
               Q~0 without ordering)

The order parameters are:

    <m_z>   layer-averaged out-of-plane magnetization
    m_dot   layer-averaged <m_top . m_bot>; +1 parallel,
            -1 antiparallel
    Q       topological charge of the top layer
    P_n(k*) angular harmonic content at the dominant FFT
            wavevector k* (n=2 and n=6 harmonics give SS vs
            SkX/BX; uniform azimuthal power gives Lab)
    Q_per_period
            |Q| / N_periods, where N_periods is estimated
            from k* and the lattice area; distinguishes
            true skyrmion lattices (Q_per_period ~ 1) from
            bubble lattices (Q_per_period ~ 0).

Functions
---------
order_parameters
    Compute the order parameters as a dict.
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
from skyrmion_simulator.simulator.main import topological_charge

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
# Canonical phase labels. Single source of truth used by
# the sweep aggregator and the plot module.
PHASE_LABELS = (
    'FM_anti', 'FM_par+', 'FM_par-',
    'iSk', 'SkX', 'BX', 'SS', 'Lab',
    'undetermined',
)
# Decision-tree thresholds. Tuned against canonical
# textures (uniform FM, single skyrmion, SkX lattice, helix,
# random labyrinth) at 256x256.
_TH_FM_MZ = 0.95         # |<m_z>| above this means FM
_TH_FM_DOT = 0.9         # |<m_top . m_bot>| above this means
# the FM is cleanly parallel or
# antiparallel (not canted)
_TH_ISK_Q_LO = 0.5       # iSk lower bound on |Q|
_TH_ISK_Q_HI = 1.5       # iSk upper bound on |Q|
_TH_HARMONIC_RATIO = 2.0  # min ratio P_n / P_iso for SkX/SS
_TH_PEAK_OVER_BG = 3.0   # min ratio peak/background to call
# ordering present in the FFT
_TH_Q_PER_PERIOD = 0.5   # min |Q| / N_periods for SkX
# (else 6-fold + Q~0 -> BX)


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
    # Floor empty bins to 1 to avoid divide-by-zero; bins with
    # zero accumulated power then return a harmless 0/1 = 0.
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
            'm_dot'            : layer-averaged
                                 <m_top . m_bot>
            'Q'                : topological charge (top)
            'k_star'           : dominant in-plane wavevector
            'P_iso'            : mean azimuthal power at k*
            'P_2', 'P_6'       : 2-fold and 6-fold harmonics
            'peak_over_bg'     : peak-to-background ratio
            'n_periods'        : (k* L / 2pi)^2 estimate
            'q_per_period'     : |Q| / n_periods
    """
    mz_top = float(np.mean(m_top[..., 2]))
    mz_bot = float(np.mean(m_bot[..., 2]))
    m_dot = float(np.mean(np.sum(m_top * m_bot, axis=-1)))
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
        # argmax over radial[1:] skips the DC bin; +1 restores
        # the index into the full radial array.
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Estimated number of principal periods in the field
    # of view. For an in-plane wavevector k*, a 2D periodic
    # texture has ~(k* L / 2*pi)^2 unit cells in an L x L
    # box. Falls back to 0 when no peak is detected.
    L_x = nx * p.a
    L_y = ny * p.a
    if k_star > 0.0:
        n_periods = (
            (k_star * L_x / (2.0 * np.pi))
            * (k_star * L_y / (2.0 * np.pi))
        )
    else:
        n_periods = 0.0
    if n_periods > 0.0:
        q_per_period = abs(Q) / n_periods
    else:
        q_per_period = 0.0
    return {
        'mz_top': mz_top, 'mz_bot': mz_bot,
        'm_dot': m_dot,
        'Q': Q,
        'k_star': k_star, 'P_iso': P_iso,
        'P_2': P_2, 'P_6': P_6,
        'peak_over_bg': peak_over_bg,
        'n_periods': float(n_periods),
        'q_per_period': float(q_per_period),
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
    label : {'FM_anti', 'FM_par+', 'FM_par-', 'iSk', 'SkX',
              'BX', 'SS', 'Lab', 'undetermined'}
        Phase label.
    obs : dict
        Order parameters used for the decision.
    """
    if obs is None:
        obs = order_parameters(m_top, m_bot, p)
    mz = obs['mz_top']
    m_dot = obs['m_dot']
    Q = obs['Q']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1. Ferromagnetic (uniform). Require uniform out-of-
    #    plane magnetization (|<m_z>| > _TH_FM_MZ) so a
    #    textured antiparallel state (stripe / SkX / iSk
    #    with m_dot ~ -1 cell-by-cell but |<m_z>| < 1) is
    #    *not* swallowed as FM_anti and falls through to
    #    the texture branches. Among uniform states, the
    #    sign of m_dot picks parallel vs antiparallel.
    if abs(mz) > _TH_FM_MZ:
        if m_dot < -_TH_FM_DOT:
            return 'FM_anti', obs
        if m_dot > +_TH_FM_DOT:
            return ('FM_par+' if mz > 0.0 else 'FM_par-'), obs
        return 'undetermined', obs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2. Single skyrmion (or a few isolated skyrmions when
    #    |Q| falls in the iSk window). Periodicity is not
    #    required: a single skyrmion has a broad FFT, not a
    #    ring.
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
        # 6-fold dominant: skyrmion lattice if topology
        # carries one full charge per principal period;
        # otherwise topologically trivial bubble lattice.
        if (ratio_6 > _TH_HARMONIC_RATIO
                and ratio_6 > ratio_2):
            if obs['q_per_period'] >= _TH_Q_PER_PERIOD:
                return 'SkX', obs
            return 'BX', obs
        # 2-fold dominant => spin spiral
        if (ratio_2 > _TH_HARMONIC_RATIO
                and ratio_2 > ratio_6):
            return 'SS', obs
        # Periodic but no clean angular order => labyrinth
        return 'Lab', obs
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 4. No clean periodic ordering and not FM.
    #    High |Q| without ordering: a packed gas of
    #    skyrmions; label as iSk (multi-skyrmion family)
    #    rather than SkX which now demands periodicity.
    if abs(Q) > _TH_ISK_Q_HI:
        return 'iSk', obs
    if abs(Q) < _TH_ISK_Q_LO:
        return 'Lab', obs
    return 'undetermined', obs
