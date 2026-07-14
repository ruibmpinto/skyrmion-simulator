"""Newell-style demag tensor via numerical surface-charge integration.

Treats each lattice cell as a uniformly magnetized rectangular
prism of size (a, a, t_Co). The cell-cell tensor is the dest-
volume-averaged field per unit source magnetization, computed
by Gauss-Legendre quadrature of the surface-charge formulation
(Maxwell). Integration density is mumax3-adaptive: maxSize =
edge_to_edge_distance / accuracy, with one quadrature point per
maxSize length along each axis. The source surface integration
uses a stagger factor of 2 (mumax3 default) to improve accuracy
at touching cells.

The self-cell at relative displacement (0, 0, 0) is overridden
with Aharoni's (1998) closed-form demag factors, because the
quadrature converges slowly near the source-coincides-dest
singularity. Off-diagonals at the self-cell vanish by symmetry.

The mumax3 source-charge convention stores N such that H_dest = +N * M_src; 
our slab kernel stores N such that H_dest = -mu0 * Ms * N * m_src. 
The negation is applied at the end so the returned kernel 
matches the slab dict schema.

A convergence assertion compares the kernel at the user-supplied
accuracy and at double that accuracy; if the relative difference
of any component exceeds `tol_conv`, a RuntimeError is raised.

Functions
---------
precompute_demag_kernels_newell
    Build self- and inter-layer demag kernels in k-space using
    the Gauss-Legendre numerical Newell formulation with the
    mumax3 adaptive integration density.
precompute_demag_kernels_newell_freebc
    Zero-padded (2*ny, 2*nx) variant for free (open) boundary
    conditions. Used by single-disc / single-rect benchmarks
    (CO SP1, muMAG SP4) where the magnet is finite and the
    PBC images would corrupt the demag field.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
from types import SimpleNamespace
# Third-party
import numpy as np
import scipy.fft as sfft
from scipy.special import roots_legendre

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Surface integration stagger factor (mumax3 default). nv and nw
# along the source surface get doubled relative to the dest
# volume density; the offset between source-surface and
# dest-volume grids avoids resonant alignment that degrades
# accuracy at touching cells.
SURFACE_STAGGER = 2

# Demag tensor construction method. Two options:
#  'closed'     -- OOMMF Newell-Williams-Dunlop 1993 closed
#                  form (transcribed from
#                  oommf/app/oxs/ext/demagcoef.cc). Exact to
#                  double precision, fast, default.
#  'quadrature' -- mumax3-style adaptive Gauss-Legendre
#                  integration of the surface-charge formula
#                  (transcribed from mumax3/mag/demagkernel.go).
#                  Used by mumax3 itself; agrees with the
#                  closed form to ~1e-3 at accuracy=4 and
#                  ~1e-5 at accuracy=8. Slower but useful for
#                  side-by-side validation.
# Override at the module level (e.g. `demag_newell._METHOD =
# 'quadrature'` before calling `precompute_demag_kernels`) to
# select the alternative path.
_METHOD = 'closed'


# -----------------------------------------------------------------------------
def _aharoni_demag_factor(a, b, c):
    """Aharoni 1998 closed-form demag factor along the c-axis.

    Parameters
    ----------
    a : float
        Full prism dimension along x (m).
    b : float
        Full prism dimension along y (m).
    c : float
        Full prism dimension along z (m).

    Returns
    -------
    N_c : float
        Demag factor along the c-axis for a uniformly magnetized
        rectangular prism of full dimensions (a, b, c).

    Notes
    -----
    Aharoni, J. Appl. Phys. 83, 3432 (1998), Eq. (1).
    Sum rule: N_a + N_b + N_c = 1; permute arguments cyclically
    to obtain N_a or N_b.
    """
    ah = a / 2.0
    bh = b / 2.0
    ch = c / 2.0
    abc = math.sqrt(ah*ah + bh*bh + ch*ch)
    ab = math.sqrt(ah*ah + bh*bh)
    ac = math.sqrt(ah*ah + ch*ch)
    bc = math.sqrt(bh*bh + ch*ch)
    return (1.0 / math.pi) * (
        (bh*bh - ch*ch) / (2.0*bh*ch)
        * math.log((abc - ah) / (abc + ah))
        + (ah*ah - ch*ch) / (2.0*ah*ch)
        * math.log((abc - bh) / (abc + bh))
        + bh / (2.0*ch) * math.log((ab + ah) / (ab - ah))
        + ah / (2.0*ch) * math.log((ab + bh) / (ab - bh))
        + ch / (2.0*ah) * math.log((bc - bh) / (bc + bh))
        + ch / (2.0*bh) * math.log((ac - ah) / (ac + ah))
        + 2.0 * math.atan2(ah*bh, ch*abc)
        + (ah**3 + bh**3 - 2.0*ch**3) / (3.0*ah*bh*ch)
        + (ah*ah + bh*bh - 2.0*ch*ch) / (3.0*ah*bh*ch) * abc
        + ch / (ah*bh) * (ac + bc)
        - (ah*ah + ch*ch)**1.5 / (3.0*ah*bh*ch)
        - (bh*bh + ch*ch)**1.5 / (3.0*ah*bh*ch)
        - (ah*ah + bh*bh)**1.5 / (3.0*ah*bh*ch)
    )


# -----------------------------------------------------------------------------
def _gl_unit_interval_nodes(n):
    """Gauss-Legendre nodes/weights on [-1, +1] with weights
    normalized to sum to 1.

    Parameters
    ----------
    n : int
        Number of quadrature points per axis. Must be >= 1.

    Returns
    -------
    nodes : numpy.ndarray(1d)
        Nodes in [-1, +1], shape (n,).
    weights : numpy.ndarray(1d)
        Weights summing to 1, shape (n,).
    """
    if n < 1:
        raise RuntimeError(f'_gl_unit_interval_nodes: n must be '
                           f'>= 1, got {n}.')
    nodes, weights = roots_legendre(n)
    return nodes, weights / 2.0


# -----------------------------------------------------------------------------
def _newell_f(x, y, z):
    """Newell-Williams-Dunlop 1993 antiderivative for the
    diagonal demag-tensor element N_xx, transcribed directly
    from `OOMMF/app/oxs/ext/demagcoef.cc::Oxs_Newell_f`.

    f is even in each argument (i.e. f(|x|, |y|, |z|)). The
    27-corner finite-difference of f gives 4 pi dx dy dz N_xx
    between two prisms of dims (dx, dy, dz) at center-to-
    center offset (X, Y, Z).

    Parameters
    ----------
    x, y, z : float
        Cartesian arguments (m).

    Returns
    -------
    f : float
        Antiderivative value at (|x|, |y|, |z|), divided by
        12 (OOMMF normalization).

    Notes
    -----
    Reference: Newell, Williams & Dunlop, J. Geophys. Res. 98,
    9551 (1993), Eq. (32). The OOMMF form writes asinh as a
    log of the squared argument, which is more stable than the
    series expansion near zero:
        2 asinh(y / sqrt(x^2 + z^2))
            = log((y + R)^2 / (x^2 + z^2)).
    """
    x = abs(x)
    y = abs(y)
    z = abs(z)
    xsq = x * x
    ysq = y * y
    zsq = z * z
    R2 = xsq + ysq + zsq
    if R2 <= 0.0:
        return 0.0
    R = math.sqrt(R2)
    sum_ = 0.0
    if z > 0.0:
        sum_ += 2.0 * (2.0 * xsq - ysq - zsq) * R
        temp1 = x * y * z
        if temp1 > 0.0:
            sum_ += -12.0 * temp1 * math.atan2(y * z, x * R)
        temp2 = xsq + zsq
        if y > 0.0 and temp2 > 0.0:
            sum_ += 3.0 * y * (zsq - xsq) * \
                math.log(((y + R) * (y + R)) / temp2)
        temp3 = xsq + ysq
        if temp3 > 0.0:
            sum_ += 3.0 * z * (ysq - xsq) * \
                math.log(((z + R) * (z + R)) / temp3)
    else:
        # z == 0 special case (2D-grid optimisation).
        if x == y:
            K = 2.0 * math.sqrt(2.0) - 6.0 * math.log(
                1.0 + math.sqrt(2.0))
            sum_ += K * xsq * x
        else:
            sum_ += 2.0 * (2.0 * xsq - ysq) * R
            if y > 0.0 and x > 0.0:
                sum_ += -6.0 * y * xsq * math.log((y + R) / x)
    return sum_ / 12.0


# -----------------------------------------------------------------------------
def _newell_g(x, y, z):
    """Newell-Williams-Dunlop 1993 antiderivative for the
    off-diagonal demag-tensor element N_xy, transcribed
    directly from
    `OOMMF/app/oxs/ext/demagcoef.cc::Oxs_Newell_g`.

    g is odd in x, odd in y, even in z. The OOMMF code tracks
    the sign explicitly before taking absolute values.

    Parameters
    ----------
    x, y, z : float
        Cartesian arguments (m).

    Returns
    -------
    g : float
        Antiderivative value, sign-adjusted for the odd-x,
        odd-y, even-z symmetry; divided by 6 (OOMMF
        normalization).

    Notes
    -----
    Reference: Newell, Williams & Dunlop, J. Geophys. Res. 98,
    9551 (1993), Eq. (33).
    """
    result_sign = 1.0
    if x < 0.0:
        result_sign *= -1.0
    if y < 0.0:
        result_sign *= -1.0
    x = abs(x)
    y = abs(y)
    z = abs(z)
    xsq = x * x
    ysq = y * y
    zsq = z * z
    R2 = xsq + ysq + zsq
    if R2 <= 0.0:
        return 0.0
    R = math.sqrt(R2)
    sum_ = -2.0 * x * y * R
    if z > 0.0:
        sum_ += -z * zsq * math.atan2(x * y, z * R)
        sum_ += -3.0 * z * ysq * math.atan2(x * z, y * R)
        sum_ += -3.0 * z * xsq * math.atan2(y * z, x * R)
        temp1 = xsq + ysq
        if temp1 > 0.0:
            sum_ += 3.0 * x * y * z * \
                math.log(((z + R) * (z + R)) / temp1)
        temp2 = ysq + zsq
        if temp2 > 0.0:
            sum_ += 0.5 * y * (3.0 * zsq - ysq) * \
                math.log(((x + R) * (x + R)) / temp2)
        temp3 = xsq + zsq
        if temp3 > 0.0:
            sum_ += 0.5 * x * (3.0 * zsq - xsq) * \
                math.log(((y + R) * (y + R)) / temp3)
    else:
        # z == 0 special case.
        if y > 0.0:
            sum_ += -y * ysq * math.log((x + R) / y)
        if x > 0.0:
            sum_ += -x * xsq * math.log((y + R) / x)
    return result_sign * sum_ / 6.0


# -----------------------------------------------------------------------------
def _newell_27_corner(X, Y, Z, dx, dy, dz, f_func):
    """27-corner Newell-1993 finite-difference of an
    antiderivative.

    The two-cell six-volume integral of 1/r is the
    6-antiderivative F evaluated through three 1st-order
    differences on the source and three on the dest. After
    substituting r = r_dest - r_src and combining same-
    position corners, the 64-corner sum collapses to a
    27-corner sum on the (-1, 0, +1)^3 lattice with separable
    2nd-order finite-difference weights W per axis. The 1D
    convolution of (+1, -1) (source) with (+1, -1) (dest) is
    (-1, +2, -1) at offsets (-1, 0, +1), so:

        W(0) = +2,  W(+-1) = -1.

    Applying an additional analytic 2nd-derivative w.r.t. X
    to F gives the function f used below; the same 27-corner
    sum on f gives the demag tensor element N_xx (and N_yy,
    N_zz by argument permutation; N_xy, N_xz, N_yz when
    `f_func` is the off-diagonal antiderivative g).

    Parameters
    ----------
    X, Y, Z : float
        Center-to-center cell offset (m).
    dx, dy, dz : float
        Cell dimensions along (x, y, z) (m).
    f_func : callable
        Antiderivative `f(x, y, z)` (diagonal) or `g(x, y, z)`
        (off-diagonal). Argument permutations select which
        tensor element is computed:
          N_xx(X,Y,Z; dx,dy,dz) <- f(X, Y, Z; dx, dy, dz)
          N_yy(X,Y,Z; dx,dy,dz) <- f(Y, X, Z; dy, dx, dz)
          N_zz(X,Y,Z; dx,dy,dz) <- f(Z, Y, X; dz, dy, dx)
          N_xy(X,Y,Z; dx,dy,dz) <- g(X, Y, Z; dx, dy, dz)
          N_xz(X,Y,Z; dx,dy,dz) <- g(X, Z, Y; dx, dz, dy)
          N_yz(X,Y,Z; dx,dy,dz) <- g(Y, Z, X; dy, dz, dx)

    Returns
    -------
    N : float
        The demag tensor element, depolarizing-positive sign
        convention so that `H = -mu_0 M_s * N * m`.
    """
    w = (-1.0, 2.0, -1.0)
    val = 0.0
    for i_idx, ix in enumerate((-1, 0, 1)):
        for j_idx, jy in enumerate((-1, 0, 1)):
            for k_idx, kz in enumerate((-1, 0, 1)):
                weight = w[i_idx] * w[j_idx] * w[k_idx]
                val += weight * f_func(
                    X + ix * dx,
                    Y + jy * dy,
                    Z + kz * dz)
    return val / (4.0 * math.pi * dx * dy * dz)


# -----------------------------------------------------------------------------
def _newell_tensor_closed(X, Y, Z, dx, dy, dz):
    """Closed-form demag tensor between two prisms of dims
    (dx, dy, dz) at center-to-center offset (X, Y, Z).

    Replaces the Gauss-Legendre `_compute_one_pair_tensor`
    quadrature with the Newell-Williams-Dunlop 1993 closed-
    form integrals.

    Parameters
    ----------
    X, Y, Z : float
        Center-to-center separation (m).
    dx, dy, dz : float
        Source and dest cell dimensions (m); the formula
        assumes both prisms share these dims.

    Returns
    -------
    N : numpy.ndarray(2d)
        3x3 demag tensor, depolarizing sign convention so that
        H = -mu_0 M_s * N * m. By Maxwell reciprocity N is
        symmetric.

    Notes
    -----
    Diagonal components (N_xx, N_yy, N_zz) are obtained from
    `_newell_f` with cyclic argument permutation:
        N_xx(X,Y,Z) = newell_8corner(X,Y,Z, dx,dy,dz; f(x,y,z))
        N_yy(X,Y,Z) = newell_8corner(Y,X,Z, dy,dx,dz; f(x,y,z))
        N_zz(X,Y,Z) = newell_8corner(Z,Y,X, dz,dy,dx; f(x,y,z))
    Off-diagonals use `_newell_g`:
        N_xy(X,Y,Z) = newell_8corner(X,Y,Z, dx,dy,dz; g(x,y,z))
        N_xz(X,Y,Z) = newell_8corner(X,Z,Y, dx,dz,dy; g(x,y,z))
        N_yz(X,Y,Z) = newell_8corner(Y,Z,X, dy,dz,dx; g(x,y,z))
    """
    out = np.zeros((3, 3))
    out[0, 0] = _newell_27_corner(
        X, Y, Z, dx, dy, dz,
        lambda xx, yy, zz: _newell_f(xx, yy, zz))
    out[1, 1] = _newell_27_corner(
        Y, X, Z, dy, dx, dz,
        lambda xx, yy, zz: _newell_f(xx, yy, zz))
    out[2, 2] = _newell_27_corner(
        Z, Y, X, dz, dy, dx,
        lambda xx, yy, zz: _newell_f(xx, yy, zz))
    out[0, 1] = _newell_27_corner(
        X, Y, Z, dx, dy, dz,
        lambda xx, yy, zz: _newell_g(xx, yy, zz))
    out[1, 0] = out[0, 1]
    out[0, 2] = _newell_27_corner(
        X, Z, Y, dx, dz, dy,
        lambda xx, yy, zz: _newell_g(xx, yy, zz))
    out[2, 0] = out[0, 2]
    out[1, 2] = _newell_27_corner(
        Y, Z, X, dy, dz, dx,
        lambda xx, yy, zz: _newell_g(xx, yy, zz))
    out[2, 1] = out[1, 2]
    return out


# -----------------------------------------------------------------------------
def _dipole_tensor(X, Y, Z, dx, dy, dz):
    """Point-dipole (far-field) demag tensor between two cells.

    Cell-averaged tensor to leading order in (cell size / r):

        N_ab = V / (4 pi r^3) * (delta_ab - 3 r_hat_a r_hat_b)

    with V = dx*dy*dz and r = (X, Y, Z). Same depolarizing sign
    convention as `_newell_tensor_closed` (H = -mu_0 M_s N m);
    relative truncation error is O((cell/r)^2).

    Parameters
    ----------
    X, Y, Z : float
        Center-to-center separation (m). Must be nonzero.
    dx, dy, dz : float
        Cell dimensions (m).

    Returns
    -------
    N : numpy.ndarray(2d)
        3x3 demag tensor.
    """
    r2 = X * X + Y * Y + Z * Z
    if r2 == 0.0:
        raise RuntimeError(
            '_dipole_tensor: zero separation; the dipole '
            'asymptote is undefined at the self-cell.')
    r = math.sqrt(r2)
    pref = dx * dy * dz / (4.0 * math.pi * r2 * r)
    rh = np.array([X, Y, Z]) / r
    return pref * (np.eye(3) - 3.0 * np.outer(rh, rh))


# -----------------------------------------------------------------------------
def _tensor_closed_or_dipole(X, Y, Z, dx, dy, dz, r_c):
    """Closed-form Newell tensor near, dipole asymptote far.

    Parameters
    ----------
    X, Y, Z : float
        Center-to-center separation (m).
    dx, dy, dz : float
        Cell dimensions (m).
    r_c : float
        Crossover radius (m).

    Returns
    -------
    N : numpy.ndarray(2d)
        3x3 demag tensor.
    """
    # Beyond r_c the 27-corner difference cancels catastrophically.
    if math.sqrt(X * X + Y * Y + Z * Z) > r_c:
        return _dipole_tensor(X=X, Y=Y, Z=Z, dx=dx, dy=dy, dz=dz)
    return _newell_tensor_closed(X=X, Y=Y, Z=Z, dx=dx, dy=dy, dz=dz)


# -----------------------------------------------------------------------------
def _delta_lat(idx):
    """Closest edge-to-edge cell distance in lattice units.

    Mirrors mumax3/mag/demagkernel.go:367. For two cells along
    an axis with integer-cell offset `idx`:
      idx = 0  -> 0 (same cell).
      idx = +-1 -> 0 (cells share a face).
      idx = +-k -> k - 1 (k - 1 cells between them).

    Parameters
    ----------
    idx : int
        Signed integer cell offset along one axis.

    Returns
    -------
    d : float
        Closest edge-to-edge distance in cell units.
    """
    idx = abs(int(idx))
    if idx > 0:
        idx -= 1
    return float(idx)


# -----------------------------------------------------------------------------
def _compute_one_pair_tensor(X, Y, Z, cellsize, n_density):
    """Compute the 3x3 demag tensor for one cell pair at
    relative displacement (X, Y, Z) via vectorized 5D
    Gauss-Legendre quadrature.

    Parameters
    ----------
    X : float
        Center-to-center x displacement (m).
    Y : float
        Center-to-center y displacement (m).
    Z : float
        Center-to-center z displacement (m).
    cellsize : tuple[float]
        (dx, dy, dz) prism dimensions (m).
    n_density : tuple[int]
        (n_x, n_y, n_z) quadrature density per dest-volume axis.
        Source-surface density on axes (v, w) is
        (n_density[v] * SURFACE_STAGGER, n_density[w] *
        SURFACE_STAGGER) per the mumax3 stagger convention.

    Returns
    -------
    N : numpy.ndarray(2d)
        3x3 demag tensor, mumax3 sign convention (positive
        N gives positive H per source M).

    Notes
    -----
    Only the upper-triangular entries are computed; by
    Maxwell reciprocity the tensor is symmetric.
    """
    cs = cellsize
    out = np.zeros((3, 3))
    # n_x_d, n_y_d, n_z_d: GL-node counts inside the dest cell.
    n_x_d, n_y_d, n_z_d = n_density
    # GL nodes (in [-1, +1]) and weights for the dest volume.
    rx_n, wrx = _gl_unit_interval_nodes(n_x_d)
    ry_n, wry = _gl_unit_interval_nodes(n_y_d)
    rz_n, wrz = _gl_unit_interval_nodes(n_z_d)
    # Map nodes to physical offsets inside the dest cell.
    rx_off = rx_n * cs[0] / 2.0
    ry_off = ry_n * cs[1] / 2.0
    rz_off = rz_n * cs[2] / 2.0
    # Reshape to rank-5 with one axis per quadrature dimension:
    #   axis 0 -> source-surface v-node index (size n_v)
    #   axis 1 -> source-surface w-node index (size n_w)
    #   axis 2 -> dest-volume x-node index    (size n_x_d)
    #   axis 3 -> dest-volume y-node index    (size n_y_d)
    #   axis 4 -> dest-volume z-node index    (size n_z_d)
    # Each array has length > 1 only along the axis it varies
    # on; the others are 1 so numpy broadcasts the same value
    # across them. Example with n_v=n_w=4, n_x_d=n_y_d=n_z_d=2:
    #   RX shape (1, 1, 2, 1, 1)  ->  2 distinct x-offsets
    #   RY shape (1, 1, 1, 2, 1)  ->  2 distinct y-offsets
    #   RZ shape (1, 1, 1, 1, 2)  ->  2 distinct z-offsets
    #   PV shape (4, 1, 1, 1, 1)  ->  4 distinct v-positions
    #   PW shape (1, 4, 1, 1, 1)  ->  4 distinct w-positions
    # When combined arithmetically the result has shape
    # (4, 4, 2, 2, 2) = 128 entries, one per (source-surface
    # node, dest-volume node) pair, with NO explicit Python
    # loops over those 128 combinations.
    RX = rx_off.reshape(1, 1, n_x_d, 1, 1)
    RY = ry_off.reshape(1, 1, 1, n_y_d, 1)
    RZ = rz_off.reshape(1, 1, 1, 1, n_z_d)
    # Outer product of 1-D weights gives 3-D volume weights.
    Wv_vol = (wrx.reshape(1, 1, n_x_d, 1, 1)
              * wry.reshape(1, 1, 1, n_y_d, 1)
              * wrz.reshape(1, 1, 1, 1, n_z_d))
    # u indexes which Cartesian component of M_source carries
    # the charged faces (the +-u faces of the source cell).
    # We loop over u = 0, 1, 2 to compute the three rows of the
    # tensor N. Each iteration treats a unit source M aligned
    # along the u-axis: M_source = e_u. A uniform M_u inside a
    # rectangular cell produces magnetic "surface charges"
    # sigma = M . n only on the two faces perpendicular to u
    # (the +u face has sigma = +M_u, the -u face has -M_u).
    # The other four faces have sigma = 0 because n is in-plane.
    for u in range(3):
        # v, w: the two axes lying *in* the +-u face (i.e. the
        # face's local coordinates). Cyclic permutation keeps
        # the right-handed frame consistent across u = 0, 1, 2.
        v = (u + 1) % 3
        w = (u + 2) % 3
        # Use a denser GL grid on the source surface than in the
        # dest volume because the integrand 1/r^3 is most singular
        # when source and dest are close (peak contribution comes
        # from the source-side surface). SURFACE_STAGGER = 2
        # follows the mumax3 convention.
        n_v = n_density[v] * SURFACE_STAGGER
        n_w = n_density[w] * SURFACE_STAGGER
        # GL nodes and weights on the unit interval [-1, +1].
        pv_n, wv = _gl_unit_interval_nodes(n_v)
        pw_n, ww = _gl_unit_interval_nodes(n_w)
        # Rescale unit-interval nodes to physical offsets that
        # span the source face from -cs[v]/2 to +cs[v]/2 (idem w).
        pv = pv_n * cs[v] / 2.0
        pw = pw_n * cs[w] / 2.0
        # Reshape to broadcast: PV varies along axis 0 only, PW
        # along axis 1 only. See the RX/RY/RZ block above for
        # the full rank-5 broadcasting layout.
        PV = pv.reshape(n_v, 1, 1, 1, 1)
        PW = pw.reshape(1, n_w, 1, 1, 1)
        # Outer product of 1-D weights => 2-D surface weights.
        # Each (i, j) entry is the GL weight of source-surface
        # node (PV[i], PW[j]).
        Wsurf = (wv.reshape(n_v, 1, 1, 1, 1)
                 * ww.reshape(1, n_w, 1, 1, 1))
        # Combined surface (axes 0, 1) and volume (axes 2, 3, 4)
        # weight tensor — shape (n_v, n_w, n_x_d, n_y_d, n_z_d).
        # A single sum over W_total times the integrand gives the
        # whole 5-D quadrature in one shot.
        W_total = Wsurf * Wv_vol
        # Each pole position is a 3-vector (x, y, z). Two of its
        # components vary across the surface (broadcastable
        # arrays PV, PW); the third is a scalar pinned to the
        # u-face. Storing them as length-3 lists keeps the code
        # axis-agnostic: pole_p[0], pole_p[1], pole_p[2] always
        # mean (x, y, z) regardless of which axis u happens to be.
        pole_p = [None, None, None]
        pole_m = [None, None, None]
        # +M face is at +cs[u]/2 along u (sigma = +M_u = +1).
        # -M face is at -cs[u]/2 along u (sigma = -1).
        pole_p[u] = +cs[u] / 2.0
        pole_m[u] = -cs[u] / 2.0
        # In-plane (v, w) sample positions: identical on both
        # faces because they are parallel, only their u-offset
        # differs by the cell thickness cs[u].
        pole_p[v] = PV
        pole_m[v] = PV
        pole_p[w] = PW
        pole_m[w] = PW
        # Build the dest sample positions: cell-center (X, Y, Z)
        # plus the GL volume-node offset (RX, RY, RZ). The cell-
        # center arrays (X, Y, Z) have shape (1, 1, 1, 1, 1)
        # per source cell, so broadcasting yields the full 5-D
        # field of source-surface x dest-volume sample pairs.
        r_dx = X + RX
        r_dy = Y + RY
        r_dz = Z + RZ
        # Displacement vector r = r_dest - r_pole+ (pointing from
        # the +M pole to the dest sample), per Coulomb's law for
        # magnetic charges:  H = sigma * r / (4 pi r^3).
        dxp = r_dx - pole_p[0]
        dyp = r_dy - pole_p[1]
        dzp = r_dz - pole_p[2]
        # |r|^2 and |r|^3 = |r|^2 * |r|; faster than |r|**3
        # because np.sqrt is one op vs np.power's general path.
        r2p = dxp*dxp + dyp*dyp + dzp*dzp
        rp3 = r2p * np.sqrt(r2p)
        # Face area: the two in-face cell extents. For a 1x1x1
        # cube this is 1; for a thin film cell it can differ
        # depending on which face (u-axis) we sum over.
        surface = cs[v] * cs[w]
        # Common prefactor pulled out of the inner expression:
        #   (quadrature weight per node) x (face area) / (4 pi).
        # Multiplying once now avoids three identical multiplies
        # in the H_x / H_y / H_z sums below.
        factor = W_total * (surface / (4.0 * math.pi))
        # Cache factor / r^3 so each H component below is just a
        # weighted sum of (signed) displacement components.
        f_over_r3p = factor / rp3
        # Repeat the same construction for the -M pole. Position
        # differs only in the u-coordinate (pole_m[u] = -cs[u]/2).
        dxn = r_dx - pole_m[0]
        dyn = r_dy - pole_m[1]
        dzn = r_dz - pole_m[2]
        # |r|^2 and |r|^3 from the -M pole.
        r2n = dxn*dxn + dyn*dyn + dzn*dzn
        rn3 = r2n * np.sqrt(r2n)
        # Prefactor identical to the +M case; the surface-charge
        # sign sigma = -1 is absorbed into the subtraction below.
        f_over_r3n = factor / rn3
        # H_alpha at the dest =
        #   sum_quad [ sigma_+ * r_+_alpha + sigma_- * r_-_alpha ]
        #          / |r|^3
        # = sum_quad ( r_+_alpha / |r_+|^3  -  r_-_alpha / |r_-|^3 )
        # since sigma_+ = +1, sigma_- = -1. .sum() reduces over
        # all 5 quadrature axes, giving one scalar per component.
        Hx = (dxp * f_over_r3p - dxn * f_over_r3n).sum()
        Hy = (dyp * f_over_r3p - dyn * f_over_r3n).sum()
        Hz = (dzp * f_over_r3p - dzn * f_over_r3n).sum()
        # Row u of the tensor: H_alpha produced inside the dest
        # cell by a unit M_u inside the source cell. Looping u
        # over 0, 1, 2 fills the full 3x3 N (Maxwell reciprocity
        # makes it symmetric, so only 6 of 9 entries are unique).
        out[u, 0] = Hx
        out[u, 1] = Hy
        out[u, 2] = Hz
    return out


# -----------------------------------------------------------------------------
def _build_layer_pair_kernel(nx, ny, dx, dy, t_layer,
                             Z_separation, accuracy,
                             images_x, images_y):
    """Build the six unique demag tensor components for one
    layer pair on the (nx, ny) FFT lattice via mumax3-style
    variable-density Gauss-Legendre numerical integration.

    Parameters
    ----------
    nx : int
        Lattice cells along x.
    ny : int
        Lattice cells along y.
    dx : float
        Lateral cell size along x (m).
    dy : float
        Lateral cell size along y (m).
    t_layer : float
        Layer thickness along z (m).
    Z_separation : float
        Center-to-center z displacement between source and
        destination layers (m). Z = 0 for the self-layer
        kernel; Z = t_layer + d_spacer for the inter-layer
        kernel.
    accuracy : float
        Mumax3 accuracy parameter. Quadrature density per axis
        is ceil(cellsize_axis * accuracy / d), where d is the
        edge-to-edge distance between cells (with d = L = min
        cell dimension for touching/coincident cells). Higher
        accuracy -> more integration points per cell pair.
    images_x : int
        Periodic image wraps summed along x: the tensor at
        each offset accumulates the +/- images_x box copies
        (0 = minimum-image truncation). Residual truncation
        scales as ((images_x + 1) * L_x)^-3.
    images_y : int
        Same along y.

    Returns
    -------
    kernel : dict
        Six (ny, nx) real-space arrays keyed by component
        name: 'Nxx', 'Nyy', 'Nzz', 'Nxy', 'Nxz', 'Nyz'. Sign
        convention matches the slab kernel: H_dest = -mu0_Ms *
        N * m_src.

    Notes
    -----
    Self-cell at lattice site (0, 0) for Z = 0 is overridden
    with Aharoni's closed form for the diagonals; off-diagonals
    are zero by symmetry. For Z != 0 (inter-layer), no self-
    cell exists in the cell-cell sense and all components are
    obtained from the quadrature.

    Periodic image contributions (w != 0) are always evaluated
    with the closed/dipole hybrid regardless of `_METHOD`: they
    sit at >= one box length, where the dipole truncation error
    is far below the quadrature accuracy.
    """
    cellsize = (dx, dy, t_layer)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PBC-aware signed lattice positions: indices > nx//2
    # wrap to negative side, matching the FFT convention.
    ix_arr = np.arange(nx)
    iy_arr = np.arange(ny)
    X1d_signed = np.where(ix_arr <= nx // 2, ix_arr, ix_arr - nx)
    Y1d_signed = np.where(iy_arr <= ny // 2, iy_arr, iy_arr - ny)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output kernels (depolarizing-positive sign convention so
    # `H = -mu0 Ms * N * m`; Newell closed-form fills directly
    # in this convention -- no end-of-loop sign flip needed).
    Nxx = np.zeros((ny, nx))
    Nyy = np.zeros((ny, nx))
    Nzz = np.zeros((ny, nx))
    Nxy = np.zeros((ny, nx))
    Nxz = np.zeros((ny, nx))
    Nyz = np.zeros((ny, nx))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-cell loop. Each (i_idx, j_idx) entry corresponds to
    # the tensor between a source cell at the lattice origin
    # and a dest cell at lattice offset (ix_s, iy_s). Two
    # equivalent constructions are supported (see _METHOD):
    #
    # 'closed'    -- OOMMF Newell-Williams-Dunlop 1993 closed
    #                form (transcribed from
    #                oommf/app/oxs/ext/demagcoef.cc). Exact to
    #                double precision; `accuracy` is ignored.
    # 'quadrature' -- mumax3-style adaptive Gauss-Legendre
    #                surface-volume integration with
    #                stagger-2 mumax3 default. `accuracy`
    #                controls the quadrature density.
    #
    # The two paths agree to ~1e-3 at accuracy=4 and ~1e-5 at
    # accuracy=8 (cell-by-cell), verified in
    # tests/test_demag_method_equivalence.py. Default is
    # 'closed' (faster, no accuracy knob, matches OOMMF and
    # Aharoni self-cell exactly).
    if _METHOD == 'closed':
        # Far-field crossover radius: 40 cells.
        r_c = 40.0 * max(dx, dy, t_layer)
        for j_idx in range(ny):
            iy_s = int(Y1d_signed[j_idx])
            for i_idx in range(nx):
                ix_s = int(X1d_signed[i_idx])
                X = ix_s * dx
                Y = iy_s * dy
                Z = Z_separation
                tensor = _tensor_closed_or_dipole(
                    X, Y, Z, dx, dy, t_layer, r_c)
                Nxx[j_idx, i_idx] = tensor[0, 0]
                Nyy[j_idx, i_idx] = tensor[1, 1]
                Nzz[j_idx, i_idx] = tensor[2, 2]
                Nxy[j_idx, i_idx] = tensor[0, 1]
                Nxz[j_idx, i_idx] = tensor[0, 2]
                Nyz[j_idx, i_idx] = tensor[1, 2]
    elif _METHOD == 'quadrature':
        cellsize = (dx, dy, t_layer)
        L = min(cellsize)
        for j_idx in range(ny):
            iy_s = int(Y1d_signed[j_idx])
            for i_idx in range(nx):
                ix_s = int(X1d_signed[i_idx])
                X = ix_s * dx
                Y = iy_s * dy
                Z = Z_separation
                dxe = _delta_lat(ix_s) * dx
                dye = _delta_lat(iy_s) * dy
                if Z_separation == 0.0:
                    dze = 0.0
                else:
                    dze = max(abs(Z_separation) - t_layer, 0.0)
                d = math.sqrt(dxe*dxe + dye*dye + dze*dze)
                if d == 0.0:
                    d = L
                maxSize = d / accuracy
                n_x = max(int(dx / maxSize + 0.5), 1)
                n_y = max(int(dy / maxSize + 0.5), 1)
                n_z = max(int(t_layer / maxSize + 0.5), 1)
                tensor = _compute_one_pair_tensor(
                    X=X, Y=Y, Z=Z, cellsize=cellsize,
                    n_density=(n_x, n_y, n_z))
                # Quadrature returns mumax3 +H/M sign; flip to
                # depolarizing-positive convention to match
                # the closed-form path and the slab kernel.
                Nxx[j_idx, i_idx] = -tensor[0, 0]
                Nyy[j_idx, i_idx] = -tensor[1, 1]
                Nzz[j_idx, i_idx] = -tensor[2, 2]
                Nxy[j_idx, i_idx] = -tensor[0, 1]
                Nxz[j_idx, i_idx] = -tensor[0, 2]
                Nyz[j_idx, i_idx] = -tensor[1, 2]
    else:
        raise RuntimeError(
            f'_build_layer_pair_kernel: unknown _METHOD '
            f'{_METHOD!r}; expected "closed" or "quadrature".')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Aharoni override at (0, 0) for the self-layer kernel.
    # Closed form reproduces Aharoni to ~1e-15, but the direct
    # Aharoni formula sidesteps the 27-corner cancellation
    # noise; quadrature only converges to Aharoni at high
    # accuracy, so the override is essential there.
    if Z_separation == 0.0:
        Nxx[0, 0] = _aharoni_demag_factor(dy, t_layer, dx)
        Nyy[0, 0] = _aharoni_demag_factor(dx, t_layer, dy)
        Nzz[0, 0] = _aharoni_demag_factor(dx, dy, t_layer)
        Nxy[0, 0] = 0.0
        Nxz[0, 0] = 0.0
        Nyz[0, 0] = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Periodic image sum: K(off) += sum_{w != 0} N(off + w L) over
    # the periodic direction(s). Runs after the Aharoni override so
    # the (0, 0) entry gets its image contributions on top of the
    # exact isolated self-term. Images sit at >= one box length,
    # so the closed/dipole hybrid is accurate for every method.
    if images_x > 0 or images_y > 0:
        r_c_img = 40.0 * max(dx, dy, t_layer)
        for j_idx in range(ny):
            iy_s = int(Y1d_signed[j_idx])
            for i_idx in range(nx):
                ix_s = int(X1d_signed[i_idx])
                for wy in range(-images_y, images_y + 1):
                    for wx in range(-images_x, images_x + 1):
                        if wx == 0 and wy == 0:
                            continue
                        X = (ix_s + wx * nx) * dx
                        Y = (iy_s + wy * ny) * dy
                        tensor = _tensor_closed_or_dipole(
                            X, Y, Z_separation, dx, dy,
                            t_layer, r_c_img)
                        Nxx[j_idx, i_idx] += tensor[0, 0]
                        Nyy[j_idx, i_idx] += tensor[1, 1]
                        Nzz[j_idx, i_idx] += tensor[2, 2]
                        Nxy[j_idx, i_idx] += tensor[0, 1]
                        Nxz[j_idx, i_idx] += tensor[0, 2]
                        Nyz[j_idx, i_idx] += tensor[1, 2]
    return {
        'Nxx': Nxx, 'Nyy': Nyy, 'Nzz': Nzz,
        'Nxy': Nxy, 'Nxz': Nxz, 'Nyz': Nyz,
    }


# -----------------------------------------------------------------------------
def _assemble_kernel_dict(p, accuracy, images_x, images_y):
    """Build self- and inter-layer real-space kernels and FFT
    them. Helper shared between the main entry point and the
    convergence check. `images_x`/`images_y` are the periodic
    image wraps summed per direction (0 = minimum image).
    """
    self_real = _build_layer_pair_kernel(
        nx=p.nx, ny=p.ny, dx=p.a, dy=p.a, t_layer=p.t_Co,
        Z_separation=0.0, accuracy=accuracy,
        images_x=images_x, images_y=images_y)
    inter_real = _build_layer_pair_kernel(
        nx=p.nx, ny=p.ny, dx=p.a, dy=p.a, t_layer=p.t_Co,
        Z_separation=(p.t_Co + p.d_Ru), accuracy=accuracy,
        images_x=images_x, images_y=images_y)
    return {
        'Nxx_self':  sfft.fft2(self_real['Nxx'], workers=-1),
        'Nyy_self':  sfft.fft2(self_real['Nyy'], workers=-1),
        'Nzz_self':  sfft.fft2(self_real['Nzz'], workers=-1),
        'Nxy_self':  sfft.fft2(self_real['Nxy'], workers=-1),
        'Nxx_inter': sfft.fft2(inter_real['Nxx'], workers=-1),
        'Nyy_inter': sfft.fft2(inter_real['Nyy'], workers=-1),
        'Nzz_inter': sfft.fft2(inter_real['Nzz'], workers=-1),
        'Nxy_inter': sfft.fft2(inter_real['Nxy'], workers=-1),
        'Nxz_inter': sfft.fft2(inter_real['Nxz'], workers=-1),
        'Nyz_inter': sfft.fft2(inter_real['Nyz'], workers=-1),
        'mu0_Ms': p.mu0 * p.Ms,
        't_Co': p.t_Co,
        'd_Ru': p.d_Ru,
        'shape': (p.ny, p.nx),
    }


# -----------------------------------------------------------------------------
def _validate_kernel_inputs(p, accuracy, tol_conv, who):
    """Shared input validation for the Newell kernel builders.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace; `t_Co`, `d_Ru`, `nx`, `ny` are
        read.
    accuracy : float
        Mumax3 accuracy parameter; must be positive.
    tol_conv : float
        Convergence tolerance; must lie in (0, 1).
    who : str
        Calling builder name, embedded in the error messages.
    """
    # Positive accuracy sets the Gauss-Legendre order; <= 0 is
    # meaningless.
    if accuracy <= 0.0:
        raise RuntimeError(
            f'{who}: accuracy must be positive, got {accuracy}.')
    # A relative tolerance only makes sense strictly inside (0, 1).
    if tol_conv <= 0.0 or tol_conv >= 1.0:
        raise RuntimeError(
            f'{who}: tol_conv must lie in (0, 1), got {tol_conv}.')
    # Zero magnetic thickness would collapse the self-demag term.
    if p.t_Co <= 0.0:
        raise RuntimeError(
            f'{who}: p.t_Co must be positive, got {p.t_Co}.')
    # Negative spacer thickness is unphysical.
    if p.d_Ru < 0.0:
        raise RuntimeError(
            f'{who}: p.d_Ru must be non-negative, got {p.d_Ru}.')
    # A stencil needs at least two cells per in-plane direction.
    if p.nx < 2 or p.ny < 2:
        raise RuntimeError(
            f'{who}: nx, ny must be >= 2, got ({p.nx}, {p.ny}).')


# -----------------------------------------------------------------------------
def _check_kernel_convergence(k_lo, k_hi, accuracy, tol_conv, who):
    """Assert the two accuracy levels agree per k-space component.

    Compares the `accuracy` and `2 * accuracy` kernels
    component-by-component and raises if any relative difference
    exceeds `tol_conv`.

    Parameters
    ----------
    k_lo, k_hi : dict
        Kernel dicts built at `accuracy` and `2 * accuracy`.
    accuracy : float
        Lower accuracy level, reported in the error message.
    tol_conv : float
        Maximum allowed relative difference per component.
    who : str
        Calling builder name, embedded in the error message.
    """
    comp_keys = [
        'Nxx_self', 'Nyy_self', 'Nzz_self', 'Nxy_self',
        'Nxx_inter', 'Nyy_inter', 'Nzz_inter', 'Nxy_inter',
        'Nxz_inter', 'Nyz_inter',
    ]
    for key in comp_keys:
        diff = np.abs(k_hi[key] - k_lo[key])
        denom = np.abs(k_hi[key]).max()
        # Component identically zero at both accuracies; skip the
        # relative test for this key.
        if denom < 1e-30:
            continue
        rel_err = diff.max() / denom
        if rel_err > tol_conv:
            raise RuntimeError(
                f'{who}: kernel did not converge for component '
                f'{key!r} between accuracy={accuracy} and '
                f'accuracy={2.0 * accuracy} (max relative '
                f'difference {rel_err:.4e} exceeds '
                f'tol_conv={tol_conv:.4e}).')


# -----------------------------------------------------------------------------
def precompute_demag_kernels_newell_freebc(p, accuracy, tol_conv):
    """Zero-padded Newell demag kernel for free (open) boundaries.

    Builds the same per-component Newell kernel as
    `precompute_demag_kernels_newell`, but on a doubled
    (2*ny x 2*nx) FFT lattice so that the cyclic convolution
    on the padded grid evaluates the linear (non-periodic)
    convolution restricted to the physical (ny x nx) region.
    Caller-side use: pad the physical m to (2*ny, 2*nx) with
    zeros outside the magnetic region, FFT, multiply by these
    kernels, IFFT, then extract the central (ny x nx) tile.
    Implemented dispatch is handled inside `demag_field` via
    the `kind == 'newell_freebc'` flag in the kernels dict.

    Parameters
    ----------
    p : SimpleNamespace
        Same attributes as `precompute_demag_kernels_newell`.
    accuracy : float
        Mumax3 accuracy parameter.
    tol_conv : float
        Maximum relative kernel error between `accuracy` and
        `2 * accuracy` for the convergence check.

    Returns
    -------
    kernels : dict
        Same schema as the PBC Newell kernel plus
        'kind' = 'newell_freebc' and 'shape_phys' = (ny, nx)
        recording the physical (un-padded) shape.

    Notes
    -----
    Per-call cost: 6 forward + 6 inverse FFTs on real arrays
    of shape (2 ny, 2 nx); 4x the work of the PBC variant.
    The kernels themselves are also 4x larger in memory.
    """
    _validate_kernel_inputs(
        p, accuracy, tol_conv,
        'precompute_demag_kernels_newell_freebc')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Doubled-grid namespace: nx, ny are 2x; all other material
    # and geometry attributes are inherited. The Newell builder
    # only reads `nx, ny, a, t_Co, d_Ru, Ms, mu0`.
    p_pad = SimpleNamespace(
        nx=int(2 * p.nx), ny=int(2 * p.ny),
        a=float(p.a), t_Co=float(p.t_Co), d_Ru=float(p.d_Ru),
        Ms=float(p.Ms), mu0=float(p.mu0))
    # Isolated (free) boundary: no periodic images.
    K_lo = _assemble_kernel_dict(p_pad, accuracy=accuracy,
                                 images_x=0, images_y=0)
    K_hi = _assemble_kernel_dict(p_pad, accuracy=2.0 * accuracy,
                                 images_x=0, images_y=0)
    _check_kernel_convergence(
        K_lo, K_hi, accuracy, tol_conv,
        'precompute_demag_kernels_newell_freebc')
    K_hi['kind'] = 'newell_freebc'
    K_hi['shape_phys'] = (int(p.ny), int(p.nx))
    return K_hi


# -----------------------------------------------------------------------------
def precompute_demag_kernels_racetrack(p, accuracy, tol_conv):
    """Mixed track-BC Newell demag kernel: periodic x, free y.

    Same per-component Newell kernel as the free-BC builder, but
    on a grid that is doubled only along y (2*ny x nx). The
    cyclic convolution is then linear (non-periodic) across the
    width (y, open top/bottom edges) and circular along the
    length (x, periodic). Caller-side use: pad the physical m to
    (2*ny, nx) with zeros in the bottom ny rows, FFT, multiply by
    these kernels, IFFT, extract the top (ny x nx) tile.

    Parameters
    ----------
    p : SimpleNamespace
        Same attributes as `precompute_demag_kernels_newell`.
    accuracy : float
        Mumax3 accuracy parameter.
    tol_conv : float
        Maximum relative kernel error between `accuracy` and
        `2 * accuracy` for the convergence check.

    Returns
    -------
    kernels : dict
        Same schema as the PBC Newell kernel plus
        'kind' = 'racetrack' and 'shape_phys' = (ny, nx).
    """
    _validate_kernel_inputs(
        p, accuracy, tol_conv,
        'precompute_demag_kernels_racetrack')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # y-doubled namespace: ny is 2x (free top/bottom), nx is kept
    # (periodic along the strip length).
    p_pad = SimpleNamespace(
        nx=int(p.nx), ny=int(2 * p.ny),
        a=float(p.a), t_Co=float(p.t_Co), d_Ru=float(p.d_Ru),
        Ms=float(p.Ms), mu0=float(p.mu0))
    # Images along the periodic x direction only; y is free
    # (zero-padded). p.pbc_images is a required attribute.
    K_lo = _assemble_kernel_dict(p_pad, accuracy=accuracy,
                                 images_x=int(p.pbc_images),
                                 images_y=0)
    K_hi = _assemble_kernel_dict(p_pad, accuracy=2.0 * accuracy,
                                 images_x=int(p.pbc_images),
                                 images_y=0)
    _check_kernel_convergence(
        K_lo, K_hi, accuracy, tol_conv,
        'precompute_demag_kernels_racetrack')
    K_hi['kind'] = 'racetrack'
    K_hi['shape_phys'] = (int(p.ny), int(p.nx))
    return K_hi


# -----------------------------------------------------------------------------
def precompute_demag_kernels_newell(p, accuracy, tol_conv):
    """Newell demag kernel via mumax3-style variable-density
    Gauss-Legendre numerical integration of the surface-charge
    formulation. Includes a convergence assertion: the kernel
    is built at `accuracy` and at `2 * accuracy`; if any
    component's relative difference exceeds `tol_conv`, a
    RuntimeError is raised and the higher-accuracy kernel is
    discarded.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace. Required attributes: `nx`, `ny`,
        `a`, `t_Co`, `d_Ru`, `Ms`, `mu0`. Missing attributes
        raise AttributeError at access time.
    accuracy : float
        Mumax3 accuracy parameter (typical 4-8). The kernel is
        also re-built at 2 * accuracy for the convergence
        check.
    tol_conv : float
        Maximum allowed relative difference between the
        accuracy and 2 * accuracy kernels for any component.
        Typical 1e-2 (1%).

    Returns
    -------
    kernels : dict
        k-space demag kernel dict matching the slab kernel
        schema, with two extra components (Nxz_inter,
        Nyz_inter) for the inter-layer cross terms that the
        slab formula treats as zero. The dict returned is the
        higher-accuracy (2 * accuracy) one.
    """
    _validate_kernel_inputs(
        p, accuracy, tol_conv, 'precompute_demag_kernels_newell')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build kernels at requested accuracy and at 2x for the
    # convergence comparison. The 2x kernel is the one we
    # return so the caller gets the better-converged result.
    # Doubly periodic: image sums along both directions.
    K_lo = _assemble_kernel_dict(p, accuracy=accuracy,
                                 images_x=int(p.pbc_images),
                                 images_y=int(p.pbc_images))
    K_hi = _assemble_kernel_dict(p, accuracy=2.0 * accuracy,
                                 images_x=int(p.pbc_images),
                                 images_y=int(p.pbc_images))
    _check_kernel_convergence(
        K_lo, K_hi, accuracy, tol_conv,
        'precompute_demag_kernels_newell')
    return K_hi
