"""Analytic Thiele integration of the topological spin Hall
torque on a Néel skyrmion profile.

Used to reproduce the v_SOT vs v_TSH comparison of Pham et al.
(2024) Figure S49 without running an LLGS time integration.

The paper's closed-form steady-state speed under SOT alone is

    v_SOT_x =  pi * H_DL * R * gamma /
               (2 alpha * (R/Delta + Delta/R))

(supplementary section 1.7). Working backwards from this and the
explicit Thiele force F_SOT = (pi/2) M_s t H_DL R, the implicit
damping prefactor is

    alpha * D_eff = alpha * M_s t (R/Delta + Delta/R) / gamma,

which, when applied to the TSH Thiele force
F_TSH = -(M_s t / gamma) * b_j * lambda_sq * I, gives

    v_TSH_x = -b_j * lambda_sq * I / (alpha * (R/Delta + Delta/R))

with
    b_j = (mu_B / q_e) * J * P / Ms          [m/s]
    I   = integral N_xy(r)^2 dx dy           [1/m^2]
    N_xy = m . (dx m x dy m).                [1/m^2]

Note: the paper quotes a different literal expression for the
dissipation tensor `D = 2 pi M_s t' R / gamma (R/Delta + Delta/R)`
that does not reconcile with its own v_SOT closed form by a
factor of 2 pi R. The formulas implemented here use the
self-consistent `D_eff` derived from v_SOT, which reproduces the
paper's quoted v_SOT magnitudes to within a few percent.

Functions
---------
neel_skyrmion_profile
    Build (m, x, y) for a Neel skyrmion on a 2D grid.
topological_density
    Per-site N_xy(r) = m . (dx m x dy m) for an arbitrary `m`.
sot_thiele_speed
    Analytic SOT-driven steady-state skyrmion speed.
tsh_thiele_speed
    Analytic TSH-driven steady-state skyrmion speed.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def neel_skyrmion_profile(R, Delta, box=None, a=2.0e-9,
                          polarity=1):
    """Build a left-handed Neel skyrmion magnetization on a 2D
    grid.

    Uses the 1D Euler-Lagrange profile
        theta(r) = 2 * arctan(exp(-(r - R) / Delta))
    wrapped into a Neel-helicity texture with the in-plane
    component pointing radially outward.

    Parameters
    ----------
    R : float
        Skyrmion radius (location of the m_z = 0 contour), in
        metres.
    Delta : float
        Domain wall width, in metres.
    box : float or None, default=None
        Half-width of the integration box in metres. If None,
        defaults to `max(6 * Delta, 3 * R)` which captures more
        than 99.5% of the topological-density integrand.
    a : float, default=2e-9
        Lattice constant in metres. Default matches the
        simulator's `p.a = 2 nm`, which also acts as the natural
        UV cutoff for the topological-density integral at small
        R/Delta (see docstring of `tsh_thiele_speed`).
    polarity : {1, -1}, default=1
        Core sign. +1 sends the core to m_z = -1 (paper
        convention for the top SAF layer).

    Returns
    -------
    m : numpy.ndarray(3d)
        Spin configuration on the n x n grid, shape (n, n, 3).
    x : numpy.ndarray(1d)
        x-coordinates (m).
    y : numpy.ndarray(1d)
        y-coordinates (m).
    a : float
        Realised lattice constant (m).
    """
    # Default to a box large enough that boundary effects on the
    # topological-density integral are negligible.
    if box is None:
        box = max(6.0 * Delta, 3.0 * R)
    # Number of grid points: ceil(2*box/a) + 1 keeps `a` as the
    # primary control while respecting the box extent.
    n = int(np.ceil(2.0 * box / a)) + 1
    # Uniform 1D coordinate arrays centred on the skyrmion core.
    x = np.linspace(-box, box, n)
    y = np.linspace(-box, box, n)
    # Recompute the realised lattice constant from np.linspace.
    a = float(x[1] - x[0])
    # 2D coordinate grids (xy indexing: XX[i,j] = x[j], YY[i,j] = y[i]).
    XX, YY = np.meshgrid(x, y, indexing='xy')
    # Polar coordinates of the grid points.
    r = np.sqrt(XX * XX + YY * YY)
    # Azimuthal angle for the Neel helicity (m_xy parallel to r).
    phi = np.arctan2(YY, XX)
    # Domain-wall profile theta(r); pi at the core, 0 at infinity.
    theta = 2.0 * np.arctan(np.exp(-(r - R) / Delta))
    # Flip the profile if polarity == -1 (core pointing up).
    if polarity == -1:
        theta = np.pi - theta
    # Precompute sin and cos of theta(r).
    sin_th = np.sin(theta)
    cos_th = np.cos(theta)
    # Allocate the magnetization tensor.
    m = np.zeros((n, n, 3))
    # Radial in-plane components (Neel form, outward-pointing).
    m[..., 0] = sin_th * np.cos(phi)
    m[..., 1] = sin_th * np.sin(phi)
    # Out-of-plane component.
    m[..., 2] = cos_th
    # Return the magnetization plus the grid for downstream
    # integration.
    return m, x, y, a


# -----------------------------------------------------------------------------
def topological_density(m, a):
    """Per-site topological charge density N_xy(r).

    N_xy(r) = m . (dx m x dy m).

    Uses central differences with open (non-periodic) boundary
    handling via `np.gradient`, appropriate for the analytic-
    profile integration here where the box is large enough that
    the skyrmion vanishes at the edges.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant / grid spacing in metres.

    Returns
    -------
    N_xy : numpy.ndarray(2d)
        Topological charge density per area (1/m^2),
        shape (ny, nx).
    """
    # Numpy's gradient gives (df/dy, df/dx) for a 2D array; we
    # request both along the spatial axes (0 = y, 1 = x).
    dm_dy = np.gradient(m, a, axis=0)
    dm_dx = np.gradient(m, a, axis=1)
    # Cross product (along the last axis, component dimension).
    cross = np.cross(dm_dx, dm_dy)
    # m . (dx m x dy m) reduces the component axis to a scalar.
    return np.sum(m * cross, axis=-1)


# -----------------------------------------------------------------------------
def sot_thiele_speed(R, Delta, p):
    """Analytic steady-state speed under DL-SOT only.

    Uses the paper's closed-form Thiele expression for an AF
    (SAF) skyrmion:
        v_SOT_x = pi * H_DL * R * gamma /
                  (2 alpha * (R/Delta + Delta/R))
    in the simulator's H-in-Tesla convention. The `H_DL` value
    is recomputed from the supplied pulse to remain consistent
    if the caller has updated `p.pulse`.

    Parameters
    ----------
    R : float
        Skyrmion radius (metres).
    Delta : float
        Domain wall width (metres).
    p : SimpleNamespace
        Parameters namespace exposing `DL_SOT`, `gamma`,
        `alpha`, and a callable `pulse` (read at t = 0).

    Returns
    -------
    v : float
        Skyrmion speed in m/s.
    """
    # Read the steady-state current density from the pulse so
    # the result reflects the caller's choice of drive.
    J0 = float(p.pulse(0.0))
    # SOT effective field at that current density (Tesla).
    H_DL = p.DL_SOT * J0
    # Inverse damping prefactor in the denominator.
    denom = 2.0 * p.alpha * (R / Delta + Delta / R)
    # Paper's closed-form steady-state Thiele speed.
    return float(np.pi * H_DL * R * p.gamma / denom)


# -----------------------------------------------------------------------------
def tsh_thiele_speed(R, Delta, lambda_sq, p,
                     box=None, polarity=1):
    """Analytic steady-state speed under TSH torque only.

    Builds a Neel-skyrmion profile on a 2D grid with spacing
    `p.a` (the simulator lattice constant), integrates N_xy^2
    over it, then evaluates
        v_TSH_x = -b_j * lambda_sq * I /
                  (alpha * (R/Delta + Delta/R))
    where I = integral N_xy^2 dxdy. See the module docstring
    for the derivation; D_eff is taken consistently with the
    paper's v_SOT closed form.

    Grid dependence at small R/Delta
    --------------------------------
    For R/Delta close to 1 the 1D-DW profile has sin(theta(0))
    non-zero, so the topological density N_xy(r) =
    -sin^2(theta)/(r Delta) is integrable but logarithmically
    sensitive to the small-r cutoff. The integral therefore
    grows with grid refinement near R/Delta = 1. Using the
    simulator lattice constant `p.a` as the grid spacing gives
    the same UV cutoff that an LLGS simulation on the same
    mesh would see (MuMax3 with a 2 nm mesh, in the paper).

    Parameters
    ----------
    R : float
        Skyrmion radius (m).
    Delta : float
        Domain wall width (m).
    lambda_sq : float
        TSH coupling length squared (m^2).
    p : SimpleNamespace
        Parameters namespace exposing `Ms`, `alpha`, `a`, `P`,
        `mu_B_over_q_e`, and a callable `pulse` (read at t=0).
    box : float or None, default=None
        Half-width of the integration box (m); see
        `neel_skyrmion_profile`.
    polarity : {1, -1}, default=1
        Skyrmion core sign.

    Returns
    -------
    v : float
        Skyrmion speed in m/s (signed).
    """
    # Build the analytic Neel skyrmion on a 2D grid with the
    # simulator lattice spacing as the UV cutoff.
    m, _x, _y, a = neel_skyrmion_profile(
        R=R, Delta=Delta, box=box, a=p.a, polarity=polarity)
    # Per-site topological charge density (1/m^2).
    N_xy = topological_density(m, a)
    # Integral of N_xy^2 dx dy (units 1/m^2).
    integral = float(np.sum(N_xy * N_xy) * a * a)
    # Steady-state current density read off the pulse callable.
    J0 = float(p.pulse(0.0))
    # TSH coupling constant b_j (m/s); see parameters.py.
    b_j = p.mu_B_over_q_e * J0 * p.P / p.Ms
    # Inverse damping prefactor in the denominator, consistent
    # with the v_SOT closed form (no 2 pi R prefactor).
    denom = p.alpha * (R / Delta + Delta / R)
    # Signed steady-state speed; F < 0 -> v < 0 from the
    # leading minus sign of the torque definition.
    return float(-b_j * lambda_sq * integral / denom)
