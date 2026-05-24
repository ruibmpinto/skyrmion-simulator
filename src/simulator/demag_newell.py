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

The mumax3 source-charge convention stores N such that
H_dest = +N * M_src; our slab kernel stores N such that
H_dest = -mu0 * Ms * N * m_src. The negation is applied at the
end so the returned kernel matches the slab dict schema.

A convergence assertion compares the kernel at the user-supplied
accuracy and at double that accuracy; if the relative difference
of any component exceeds `tol_conv`, a RuntimeError is raised.

Functions
---------
precompute_demag_kernels_newell
    Build self- and inter-layer demag kernels in k-space using
    the Gauss-Legendre numerical Newell formulation with the
    mumax3 adaptive integration density.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
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
    n_x_d, n_y_d, n_z_d = n_density
    # Volume quadrature nodes/weights shared across source dirs
    rx_n, wrx = _gl_unit_interval_nodes(n_x_d)
    ry_n, wry = _gl_unit_interval_nodes(n_y_d)
    rz_n, wrz = _gl_unit_interval_nodes(n_z_d)
    rx_off = rx_n * cs[0] / 2.0
    ry_off = ry_n * cs[1] / 2.0
    rz_off = rz_n * cs[2] / 2.0
    # Pre-broadcasted volume position offsets and weights;
    # rank-5 with surface (v, w) axes broadcast against later.
    RX = rx_off.reshape(1, 1, n_x_d, 1, 1)
    RY = ry_off.reshape(1, 1, 1, n_y_d, 1)
    RZ = rz_off.reshape(1, 1, 1, 1, n_z_d)
    Wv_vol = (wrx.reshape(1, 1, n_x_d, 1, 1)
              * wry.reshape(1, 1, 1, n_y_d, 1)
              * wrz.reshape(1, 1, 1, 1, n_z_d))
    # Loop over source magnetization direction u in {x, y, z}.
    for u in range(3):
        v = (u + 1) % 3
        w = (u + 2) % 3
        n_v = n_density[v] * SURFACE_STAGGER
        n_w = n_density[w] * SURFACE_STAGGER
        pv_n, wv = _gl_unit_interval_nodes(n_v)
        pw_n, ww = _gl_unit_interval_nodes(n_w)
        pv = pv_n * cs[v] / 2.0
        pw = pw_n * cs[w] / 2.0
        PV = pv.reshape(n_v, 1, 1, 1, 1)
        PW = pw.reshape(1, n_w, 1, 1, 1)
        Wsurf = (wv.reshape(n_v, 1, 1, 1, 1)
                 * ww.reshape(1, n_w, 1, 1, 1))
        W_total = Wsurf * Wv_vol
        # Pole positions: +M on +u face, -M on -u face. Set
        # axis-by-axis so the array dimensionality follows the
        # PV/PW broadcasting.
        pole_p = [None, None, None]
        pole_m = [None, None, None]
        pole_p[u] = +cs[u] / 2.0
        pole_m[u] = -cs[u] / 2.0
        pole_p[v] = PV
        pole_m[v] = PV
        pole_p[w] = PW
        pole_m[w] = PW
        # Destination position
        r_dx = X + RX
        r_dy = Y + RY
        r_dz = Z + RZ
        # +M pole contribution
        dxp = r_dx - pole_p[0]
        dyp = r_dy - pole_p[1]
        dzp = r_dz - pole_p[2]
        r2p = dxp*dxp + dyp*dyp + dzp*dzp
        rp3 = r2p * np.sqrt(r2p)
        surface = cs[v] * cs[w]
        factor = W_total * (surface / (4.0 * math.pi))
        f_over_r3p = factor / rp3
        # -M pole contribution (sigma sign flips)
        dxn = r_dx - pole_m[0]
        dyn = r_dy - pole_m[1]
        dzn = r_dz - pole_m[2]
        r2n = dxn*dxn + dyn*dyn + dzn*dzn
        rn3 = r2n * np.sqrt(r2n)
        f_over_r3n = factor / rn3
        # Net H components, reducing over 5D quadrature axes.
        Hx = (dxp * f_over_r3p - dxn * f_over_r3n).sum()
        Hy = (dyp * f_over_r3p - dyn * f_over_r3n).sum()
        Hz = (dzp * f_over_r3p - dzn * f_over_r3n).sum()
        out[u, 0] = Hx
        out[u, 1] = Hy
        out[u, 2] = Hz
    return out


# -----------------------------------------------------------------------------
def _build_layer_pair_kernel(nx, ny, dx, dy, t_layer,
                             Z_separation, accuracy):
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
    """
    cellsize = (dx, dy, t_layer)
    L = min(cellsize)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # PBC-aware signed lattice positions: indices > nx//2
    # wrap to negative side, matching the FFT convention.
    ix_arr = np.arange(nx)
    iy_arr = np.arange(ny)
    X1d_signed = np.where(ix_arr <= nx // 2, ix_arr, ix_arr - nx)
    Y1d_signed = np.where(iy_arr <= ny // 2, iy_arr, iy_arr - ny)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output kernels
    Nxx = np.zeros((ny, nx))
    Nyy = np.zeros((ny, nx))
    Nzz = np.zeros((ny, nx))
    Nxy = np.zeros((ny, nx))
    Nxz = np.zeros((ny, nx))
    Nyz = np.zeros((ny, nx))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-cell loop. Each cell pair gets an adaptive integration
    # density per mumax3's recipe.
    for j_idx in range(ny):
        iy_s = int(Y1d_signed[j_idx])
        for i_idx in range(nx):
            ix_s = int(X1d_signed[i_idx])
            X = ix_s * dx
            Y = iy_s * dy
            Z = Z_separation
            # Edge-to-edge distance per axis
            dxe = _delta_lat(ix_s) * dx
            dye = _delta_lat(iy_s) * dy
            if Z_separation == 0.0:
                # Same layer: cells overlap in z, no z-gap.
                dze = 0.0
            else:
                # Different layer: gap = |Z| - t_layer if
                # layers are non-overlapping, else 0.
                dze = max(abs(Z_separation) - t_layer, 0.0)
            d = math.sqrt(dxe*dxe + dye*dye + dze*dze)
            if d == 0.0:
                # Touching or coincident cells: fall back to
                # min cell dimension as the length scale.
                d = L
            maxSize = d / accuracy
            n_x = max(int(dx / maxSize + 0.5), 1)
            n_y = max(int(dy / maxSize + 0.5), 1)
            n_z = max(int(t_layer / maxSize + 0.5), 1)
            tensor = _compute_one_pair_tensor(
                X=X, Y=Y, Z=Z, cellsize=cellsize,
                n_density=(n_x, n_y, n_z))
            Nxx[j_idx, i_idx] = tensor[0, 0]
            Nyy[j_idx, i_idx] = tensor[1, 1]
            Nzz[j_idx, i_idx] = tensor[2, 2]
            Nxy[j_idx, i_idx] = tensor[0, 1]
            Nxz[j_idx, i_idx] = tensor[0, 2]
            Nyz[j_idx, i_idx] = tensor[1, 2]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Sign convention flip: mumax3 stores +H/M, slab uses
    # H = -mu0*Ms*N*m. Negate to match the existing slab
    # kernel schema.
    Nxx = -Nxx
    Nyy = -Nyy
    Nzz = -Nzz
    Nxy = -Nxy
    Nxz = -Nxz
    Nyz = -Nyz
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Self-cell override at (0, 0) for the self-layer kernel.
    # Aharoni gives the diagonals exactly; off-diagonals
    # vanish by the cell mirror symmetries.
    if Z_separation == 0.0:
        Nxx[0, 0] = _aharoni_demag_factor(dy, t_layer, dx)
        Nyy[0, 0] = _aharoni_demag_factor(dx, t_layer, dy)
        Nzz[0, 0] = _aharoni_demag_factor(dx, dy, t_layer)
        Nxy[0, 0] = 0.0
        Nxz[0, 0] = 0.0
        Nyz[0, 0] = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return {
        'Nxx': Nxx, 'Nyy': Nyy, 'Nzz': Nzz,
        'Nxy': Nxy, 'Nxz': Nxz, 'Nyz': Nyz,
    }


# -----------------------------------------------------------------------------
def _assemble_kernel_dict(p, accuracy):
    """Build self- and inter-layer real-space kernels and FFT
    them. Helper shared between the main entry point and the
    convergence check.
    """
    self_real = _build_layer_pair_kernel(
        nx=p.nx, ny=p.ny, dx=p.a, dy=p.a, t_layer=p.t_Co,
        Z_separation=0.0, accuracy=accuracy)
    inter_real = _build_layer_pair_kernel(
        nx=p.nx, ny=p.ny, dx=p.a, dy=p.a, t_layer=p.t_Co,
        Z_separation=(p.t_Co + p.d_Ru), accuracy=accuracy)
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
    if accuracy <= 0.0:
        raise RuntimeError(
            f'precompute_demag_kernels_newell: '
            f'accuracy must be positive, got {accuracy}.')
    if tol_conv <= 0.0 or tol_conv >= 1.0:
        raise RuntimeError(
            f'precompute_demag_kernels_newell: '
            f'tol_conv must lie in (0, 1), got {tol_conv}.')
    if p.t_Co <= 0.0:
        raise RuntimeError(
            f'precompute_demag_kernels_newell: '
            f'p.t_Co must be positive, got {p.t_Co}.')
    if p.d_Ru < 0.0:
        raise RuntimeError(
            f'precompute_demag_kernels_newell: '
            f'p.d_Ru must be non-negative, got {p.d_Ru}.')
    if p.nx < 2 or p.ny < 2:
        raise RuntimeError(
            f'precompute_demag_kernels_newell: '
            f'nx, ny must be >= 2, got ({p.nx}, {p.ny}).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build kernels at requested accuracy and at 2x for the
    # convergence comparison. The 2x kernel is the one we
    # return so the caller gets the better-converged result.
    K_lo = _assemble_kernel_dict(p, accuracy=accuracy)
    K_hi = _assemble_kernel_dict(p, accuracy=2.0 * accuracy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Component-by-component relative-difference check on the
    # k-space arrays (real parts; imaginary residuals are
    # numerical FFT noise).
    comp_keys = [
        'Nxx_self', 'Nyy_self', 'Nzz_self', 'Nxy_self',
        'Nxx_inter', 'Nyy_inter', 'Nzz_inter', 'Nxy_inter',
        'Nxz_inter', 'Nyz_inter',
    ]
    for key in comp_keys:
        diff = np.abs(K_hi[key] - K_lo[key])
        denom = np.abs(K_hi[key]).max()
        if denom < 1e-30:
            # Component identically zero at both accuracies;
            # skip the relative test for this key.
            continue
        rel_err = diff.max() / denom
        if rel_err > tol_conv:
            raise RuntimeError(
                f'precompute_demag_kernels_newell: '
                f'kernel did not converge for component '
                f'{key!r} between accuracy={accuracy} and '
                f'accuracy={2.0*accuracy} (max relative '
                f'difference {rel_err:.4e} exceeds '
                f'tol_conv={tol_conv:.4e}).')
    return K_hi
