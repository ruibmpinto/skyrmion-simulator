"""Post-simulation analysis for SAF skyrmion dynamics.

Relaxes a skyrmion to equilibrium, then drives it with
current and measures velocity, Hall angle, and diameter.
Compares results against the reference paper.

Functions
---------
skyrmion_center
    Compute skyrmion center of mass.
skyrmion_diameter
    Compute skyrmion diameter from mz < 0 area.
run_analysis
    Full relaxation + current-driven analysis.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import sys
# Third-party
import numpy as np
# Local
from src.simulator.parameters import default_params
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rhs_local_keff, rk4_step
from src.simulator.main import topological_charge
from src.simulator.pulses import ConstantPulse

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def skyrmion_center(m, a, core_polarity):
    """Compute skyrmion center of mass.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.
    core_polarity : {+1, -1}
        Sign convention for the skyrmion core. `+1` if the
        core sits at `m_z = -1` (top-layer convention used by
        `saf_skyrmion`); `-1` if the core sits at `m_z = +1`
        (bottom layer). Required, no default.

    Returns
    -------
    cx : float
        Center x-coordinate in meters.
    cy : float
        Center y-coordinate in meters.
    """
    # Loud rejection of an unrecognised polarity sign.
    if core_polarity not in (+1, -1):
        raise RuntimeError(
            f'skyrmion_center: core_polarity must be +1 or -1, '
            f'got {core_polarity!r}.')
    # Extract number of grid points per direction
    ny, nx = m.shape[:2]
    # Create grid of lattice sites in units of a. jj, ii have shape (ny, nx).
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),)
    # Weight peaks at the core for either polarity:
    #   polarity=+1, core at m_z=-1: w = (1 - m_z)/2
    #   polarity=-1, core at m_z=+1: w = (1 + m_z)/2
    # m_z stored in m[..., 2].
    # w has shape (ny, nx)
    w = (1.0 - core_polarity * m[..., 2]) / 2.0
    ws = w.sum()
    # Empty core mask (annihilated skyrmion) gives an undefined
    # centroid; raise rather than emit NaN from divide-by-zero.
    if ws == 0.0:
        raise RuntimeError(
            'skyrmion_center: empty core mask (ws == 0); the '
            'skyrmion may have annihilated.')
    # Weighted centroid in physical (meters) coordinates.
    cx = np.sum(w * jj * a) / ws
    cy = np.sum(w * ii * a) / ws
    # Return
    return cx, cy


# ---------------------------------------------------------------------
def skyrmion_diameter(m, a, core_polarity):
    """Compute skyrmion diameter from the core-region area.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.
    core_polarity : {+1, -1}
        `+1` if the core is at `m_z = -1` (mask is `m_z < 0`);
        `-1` if the core is at `m_z = +1` (mask is `m_z > 0`).
        Required, no default.

    Returns
    -------
    d : float
        Skyrmion diameter in meters.
    """
    # Loud rejection of an unrecognised polarity sign.
    if core_polarity not in (+1, -1):
        raise RuntimeError(
            f'skyrmion_diameter: core_polarity must be +1 or -1, '
            f'got {core_polarity!r}.')
    # Core mask: sites with sign(m_z) opposite to the background.
    # `polarity * m_z < 0` selects:
    #   polarity=+1: m_z < 0 (top-layer core)
    #   polarity=-1: m_z > 0 (bottom-layer core)
    n_inside = int(np.sum(core_polarity * m[..., 2] < 0))
    area = n_inside * a * a
    # Equivalent disk diameter d = 2 sqrt(A / pi).
    return 2.0 * np.sqrt(area / np.pi)


# ---------------------------------------------------------------------
def skyrmion_ellipse(m, a, core_polarity):
    """Major and minor diameters of the skyrmion via second
    moments of the core-region mask.

    Treats the set of core sites as a 2D point cloud and
    diagonalises its covariance matrix. The major and minor
    diameters are then D_k = 4 sqrt(lambda_k) where lambda_k are
    the eigenvalues (variance along each principal axis). For a
    uniform disk of radius R the covariance has both eigenvalues
    equal to R^2/4, giving D1 = D2 = 2R = diameter, consistent
    with `skyrmion_diameter`.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.
    core_polarity : {+1, -1}
        `+1` if the core is at `m_z = -1`; `-1` if the core is
        at `m_z = +1`. Required, no default.

    Returns
    -------
    D1 : float
        Major-axis diameter in meters.
    D2 : float
        Minor-axis diameter in meters.
    theta : float
        Angle of the major axis with respect to +x, in radians,
        in [-pi/2, pi/2].

    Notes
    -----
    Raises `RuntimeError` if the mask has fewer than 3 sites
    (the covariance matrix is not well-defined). For a circular
    skyrmion the two eigenvalues coincide and `theta` is set to
    zero (the eigen-decomposition is degenerate).
    """
    # Loud rejection of an unrecognised polarity sign.
    if core_polarity not in (+1, -1):
        raise RuntimeError(
            f'skyrmion_ellipse: core_polarity must be +1 or -1, '
            f'got {core_polarity!r}.')
    # Build the core mask using the polarity-aware rule.
    mask = core_polarity * m[..., 2] < 0
    # Number of sites inside the mask.
    n_inside = int(np.sum(mask))
    # Reject masks with too few sites to fit a covariance matrix.
    if n_inside < 3:
        raise RuntimeError(
            f'skyrmion_ellipse: mask has only {n_inside} sites; '
            f'need at least 3 to estimate a covariance.')
    # Lattice shape (ny rows, nx cols).
    ny, nx = m.shape[:2]
    # 2D index grids: jj has column index (x), ii has row index (y).
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),)
    # Physical coordinates (meters) of every masked site.
    x = jj[mask] * a
    y = ii[mask] * a
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Centre the point cloud at its centroid.
    x = x - np.mean(x)
    # Centre along y as well.
    y = y - np.mean(y)
    # Variance along x (covariance matrix entry sxx).
    sxx = float(np.mean(x * x))
    # Variance along y (covariance matrix entry syy).
    syy = float(np.mean(y * y))
    # Cross-covariance (off-diagonal entry sxy = syx).
    sxy = float(np.mean(x * y))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Trace of the 2x2 covariance matrix (sum of eigenvalues).
    tr = sxx + syy
    # Determinant of the 2x2 covariance matrix (product of eigenvalues).
    det = sxx * syy - sxy * sxy
    # Discriminant of the 2x2 characteristic polynomial; clamp to
    # zero to absorb O(epsilon) floating-point negativity.
    disc = max(0.25 * (sxx - syy) ** 2 + sxy * sxy, 0.0)
    # Square root of the discriminant.
    sqrt_disc = np.sqrt(disc)
    # Larger eigenvalue (variance along major principal axis).
    lam1 = 0.5 * tr + sqrt_disc
    # Smaller eigenvalue (variance along minor principal axis).
    lam2 = 0.5 * tr - sqrt_disc
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Major-axis orientation; degenerate when sxx == syy and sxy == 0
    # (perfectly circular skyrmion). Default to theta = 0 there.
    if abs(sxx - syy) < 1e-30 and abs(sxy) < 1e-30:
        theta = 0.0
    else:
        # Half-angle of the eigenvector relative to +x; range
        # [-pi/2, pi/2] courtesy of arctan2.
        theta = 0.5 * np.arctan2(2.0 * sxy, sxx - syy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Major-axis diameter from the larger eigenvalue
    # (D = 4 sqrt(variance) for a uniformly filled ellipse).
    D1 = 4.0 * np.sqrt(max(lam1, 0.0))
    # Minor-axis diameter from the smaller eigenvalue.
    D2 = 4.0 * np.sqrt(max(lam2, 0.0))
    # Return diameters and orientation as plain Python floats.
    return float(D1), float(D2), float(theta)


# ---------------------------------------------------------------------
def dw_angle(m, a, core_polarity, mz_thresh=0.5):
    """In-plane magnetization angle at the right domain wall.

    Restricts to DW sites (|m_z| < mz_thresh) on the +x side of
    the skyrmion (x > centroid_x) and returns the population-
    weighted angle of the in-plane component, measured with
    respect to the -x axis as in Pham et al. 2024 (Fig.~S44C
    inset).

    For a pure Neel skyrmion with outward-pointing in-plane
    magnetization, the right DW has m_xy parallel to +x, so
    `psi = pi` (180 degrees) relative to -x. Under current drive
    the DW magnetization rotates away from Neel; this rotation
    is what the paper reports.

    Averaging over the full mz=0 ring would cancel by Neel-
    radial symmetry, hence the +x half-plane restriction.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters (used to locate the centroid
        via `skyrmion_center`).
    core_polarity : {+1, -1}
        Forwarded to `skyrmion_center` to locate the skyrmion
        centre. Required, no default.
    mz_thresh : float, default=0.5
        Sites with |m_z| < mz_thresh are treated as DW sites.

    Returns
    -------
    psi : float
        Signed DW magnetization angle in radians, measured as the
        deviation from the layer's natural Neel orientation. The
        natural orientation is identified by the sign of the
        averaged in-plane x-component at the +x DW: layers with
        m_xy pointing along +x have reference +x_hat, layers with
        m_xy pointing along -x have reference -x_hat. Both layers
        therefore return psi ~ 0 at rest, and psi tracks the
        rotation of the DW magnetization under drive without ever
        wrapping by +-pi at the branch cut.
    """
    # Skyrmion centroid (in physical metres) used to split DW
    # sites into right (+x) vs left (-x) halves.
    cx, _cy = skyrmion_center(m, a, core_polarity)
    # Ring mask: |m_z| < threshold isolates the wall region around
    # the mz=0 contour.
    dw_mask = np.abs(m[..., 2]) < mz_thresh
    # Build a 2D x-coordinate grid (m) matching the spin array.
    ny, nx = m.shape[:2]
    jj = np.broadcast_to(
        np.arange(nx, dtype=float), (ny, nx))
    # Physical x at each site, in metres.
    x = jj * a
    # Right-half-plane mask: x > centroid.
    right_mask = x > cx
    # Intersect the DW ring with the right half-plane.
    mask = dw_mask & right_mask
    # Count of sites used in the average.
    n_dw = int(np.sum(mask))
    # Reject empty masks loudly; usually means a parameter error.
    if n_dw == 0:
        raise RuntimeError(
            f'dw_angle: no DW sites in the +x half-plane with '
            f'|m_z| < {mz_thresh}; check that the texture has a DW.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Population-averaged in-plane x-component across the right DW.
    mx_avg = float(np.mean(m[..., 0][mask]))
    # Population-averaged in-plane y-component across the right DW.
    my_avg = float(np.mean(m[..., 1][mask]))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The two SAF layers sit on opposite Neel branches at rest:
    # one has mx_avg > 0 (m_xy along +x_hat), the other has
    # mx_avg < 0. A bare atan2 then jumps by +-2*pi at the branch
    # cut under small SOT tilts. Flipping (mx, my) by sign(mx_avg)
    # maps both branches onto +x_hat, so atan2 returns the signed
    # deviation in (-pi/2, +pi/2] -- zero at rest, equal to the
    # actual SOT-induced rotation under drive, no wrap.
    s = 1.0 if mx_avg >= 0.0 else -1.0
    return float(np.arctan2(s * my_avg, s * mx_avg))


# ---------------------------------------------------------------------
def run_analysis():
    """Full relaxation + current-driven analysis.

    Phase 1: Relax skyrmion with J=0 for 500 ps.
    Phase 2: Drive with J=4e11 A/m^2 for 1 ns.
    Compare velocity, Hall angle, diameter to paper.
    """
    # =========================================================
    # Phase 1: Relaxation (no current)
    # =========================================================
    print('=== Phase 1: Relaxation (J=0, 500 ps) ===')
    p = default_params()
    # Zero current and SOT fields to find the true zero-drive equilibrium.
    # The pulse is also swapped to ConstantPulse(0) because the
    # integrator reads p.pulse(t) — not p.H_DL/p.H_FL — at every substep.
    p.J_current = 0.0
    p.H_DL = 0.0
    p.H_FL = 0.0
    p.pulse = ConstantPulse(0.0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
    # Top layer: core at m_z = -1 → core_polarity = +1.
    d0 = skyrmion_diameter(m_top, p.a, core_polarity=+1)
    print(f'Initial diameter: {d0 * 1e9:.1f} nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 10000 * 50 fs = 500 ps
    n_relax = 10000  
    t = 0.0
    for step in range(1, n_relax + 1):
        m_top, m_bot = rk4_step(rhs_local_keff, m_top, m_bot, t, p.dt, p)
        t += p.dt
        if step % 2000 == 0:
            d = skyrmion_diameter(m_top, p.a, core_polarity=+1)
            t_ps = step * p.dt * 1e12
            print(f'  t={t_ps:.0f} ps: d={d * 1e9:.1f} nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    d_eq = skyrmion_diameter(m_top, p.a, core_polarity=+1)
    Q_eq = topological_charge(m_top, p.a)
    print(f'Equilibrium: d={d_eq * 1e9:.1f} nm, Q={Q_eq:.4f}')
    print()
    # =========================================================
    # Phase 2: Current-driven dynamics
    # =========================================================
    print('=== Phase 2: Current drive (1 ns) ===')
    p2 = default_params()
    print(f'J = {p2.J_current:.2e} A/m^2')
    print(f'H_DL = {p2.H_DL:.4e} T')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Copy so the relaxed state is preserved as the initial drive frame.
    m_top_d = m_top.copy()
    m_bot_d = m_bot.copy()
    c0 = skyrmion_center(m_top_d, p2.a, core_polarity=+1)
    print(f'Start: ({c0[0] * 1e9:.1f}, {c0[1] * 1e9:.1f}) nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_drive = 20000  # 20000 * 50 fs = 1 ns
    track = []
    # Phase 2 starts at t = 0 so the default ConstantPulse(p.J_current)
    # is on from the first substep onwards.
    t = 0.0
    for step in range(1, n_drive + 1):
        m_top_d, m_bot_d = rk4_step(rhs_local_keff, m_top_d, m_bot_d, t, p2.dt, p2)
        t += p2.dt
        if step % 4000 == 0:
            c = skyrmion_center(m_top_d, p2.a, core_polarity=+1)
            d = skyrmion_diameter(m_top_d, p2.a, core_polarity=+1)
            Q = topological_charge(m_top_d, p2.a)
            t_ps = step * p2.dt * 1e12
            track.append((t_ps, c[0], c[1]))
            print(
                f'  t={t_ps:.0f} ps: '
                f'({c[0] * 1e9:.1f}, '
                f'{c[1] * 1e9:.1f}) nm, '
                f'd={d * 1e9:.1f} nm, '
                f'Q={Q:.4f}')
    # =========================================================
    # Velocity from steady-state (last 3 points)
    # =========================================================
    if len(track) >= 3:
        # Finite-difference velocity from last two stored centers.
        t1, x1, y1 = track[-3]
        t2, x2, y2 = track[-1]
        dt_s = (t2 - t1) * 1e-12
        vx = (x2 - x1) / dt_s
        vy = (y2 - y1) / dt_s
        v = np.sqrt(vx ** 2 + vy ** 2)
        # Hall angle: deviation of motion from the drive direction (~0 SAF).
        hall = np.degrees(np.arctan2(abs(vy), abs(vx)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print()
        print('=== Comparison with paper ===')
        print(f'Equilibrium diameter:')
        print(f'  Sim:   {d_eq * 1e9:.0f} nm')
        print(f'  Paper: 197 nm')
        print(f'Velocity @ J={p2.J_current:.0e}:')
        print(f'  Sim:   {v:.0f} m/s (vx={vx:.0f}, vy={vy:.0f})')
        print(f'  Paper: ~400 m/s (mumag)')
        print(f'Hall angle:')
        print(f'  Sim:   {hall:.1f} deg')
        print(f'  Paper: ~0 deg (SAF)')
        print(f'Topological charge:')
        print(f'  Sim:   {Q:.4f}')
        print(f'  Paper: +/-1')

# =====================================================================
if __name__ == '__main__':
    run_analysis()
