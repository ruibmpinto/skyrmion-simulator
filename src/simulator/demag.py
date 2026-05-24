"""Magnetostatic (dipolar) field for thin-film SAF stacks.

Computes the long-range demagnetizing field in 2D Fourier
space for two stacked thin Co layers separated by a Ru/Pt
spacer. Two demag formulations are exposed via the `kind`
argument to `precompute_demag_kernels`:

  `kind='slab'`:
    Each layer is treated as a continuous (out-of-plane) slab of 
    thickness `t_Co`; the in-plane lattice has periodic boundary conditions and
    the kernel is computed analytically in k-space via the standard 
    thin-film shape function
        f(k, t) = (1 - exp(-|k|*t)) / (|k|*t).

  `kind='newell'`:
    Each cell is treated as a finite-width rectangular prism (a x a x t_Co). 
    The cell-cell tensor is built by Gauss-Legendre quadrature of the 
    surface-charge formulation, with mumax3-style variable integration
    density and a convergence assertion. 
    Adds inter-layer N_xz, N_yz cross-terms that the slab formulation
    treats as zero.

Both formulations return a dict with the same key schema;
the slab dispatcher pads N_xz_inter, N_yz_inter with zeros
so that `demag_field` is kind-agnostic.

Sign convention: H_demag = -mu0 * Ms * N * m, with N
dimensionless.

Functions
---------
precompute_demag_kernels
    Build self- and inter-layer demag tensors on the FFT
    grid for a given parameter namespace and demag kind.
demag_field
    Evaluate the demag field on both layers from cached
    kernels. Handles both slab and Newell kernels via the
    shared dict schema.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np
import scipy.fft as sfft
# Local
from src.simulator.demag_newell import \
    precompute_demag_kernels_newell

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def precompute_demag_kernels(p, kind, accuracy, tol_conv):
    """Build self- and inter-layer demag kernels in k-space.

    Dispatcher: routes to either the analytic slab kernel or
    the numerical Newell kernel based on `kind`.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace. Must expose `nx`, `ny`, `a`,
        `t_Co`, `d_Ru`, `Ms`, and `mu0`. A missing attribute
        raises `AttributeError` at access time so silent
        defaulting does not corrupt the kernels.
    kind : str
        Demag formulation: 'slab' or 'newell'.
        'slab' uses the analytic thin-film shape factor;
        'newell' uses the mumax3-style finite-prism numerical
        integration. Required, no default.
    accuracy : {float, None}
        Mumax3 accuracy parameter for the Newell kernel
        (typical 4-8). Pass None for kind='slab'.
    tol_conv : {float, None}
        Maximum relative difference between accuracy and
        2*accuracy kernels for the Newell convergence check.
        Pass None for kind='slab'.

    Returns
    -------
    kernels : dict
        Dictionary holding the four self-layer tensor
        components and four inter-layer components on a 2D
        FFT grid of shape (ny, nx), plus the inter-layer
        cross terms `Nxz_inter` and `Nyz_inter` (zero for
        slab, non-zero for newell), the bulk coefficients
        `mu0_Ms`, and the geometry `t_Co`, `d_Ru`, `shape`.

    Notes
    -----
    Convention: in real space H_demag = -mu0 * (N * m), so
    in k-space H(k) = -mu0 * Ms * N(k) * m(k). N is
    dimensionless; the prefactor mu0*Ms is applied inside
    `demag_field`.
    """
    if kind == 'slab':
        if accuracy is not None or tol_conv is not None:
            raise RuntimeError(
                f"precompute_demag_kernels: kind='slab' does "
                f"not accept accuracy/tol_conv; got "
                f"accuracy={accuracy!r}, tol_conv={tol_conv!r}.")
        return _precompute_slab(p)
    if kind == 'newell':
        if accuracy is None or tol_conv is None:
            raise RuntimeError(
                f"precompute_demag_kernels: kind='newell' "
                f"requires accuracy and tol_conv; got "
                f"accuracy={accuracy!r}, tol_conv={tol_conv!r}.")
        return precompute_demag_kernels_newell(
            p, accuracy=accuracy, tol_conv=tol_conv)
    raise RuntimeError(
        f"precompute_demag_kernels: kind must be 'slab' or "
        f"'newell', got {kind!r}.")


# ---------------------------------------------------------------------
def _precompute_slab(p):
    """Slab-approximation demag kernel (analytic thin-film
    shape factor in k-space).

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace, as for `precompute_demag_kernels`.

    Returns
    -------
    kernels : dict
        Same schema as the Newell dispatcher, with
        Nxz_inter, Nyz_inter padded to zero (the slab kernel
        does not couple in-plane source to out-of-plane field
        between layers).
    """
    nx, ny = p.nx, p.ny
    a = p.a
    t = p.t_Co
    d_Ru = p.d_Ru
    if t <= 0.0:
        raise RuntimeError(f'p.t_Co must be positive, got {t}.')
    if d_Ru < 0.0:
        raise RuntimeError(f'p.d_Ru must be non-negative, got {d_Ru}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2D k-grid in rad/m
    # 2*pi factor converts numpy's cycle/sample frequencies to rad/m.
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=a)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=a)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX * KX + KY * KY
    K = np.sqrt(K2)
    Kt = K * t
    Kd = K * d_Ru
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1/K^2 with safe handling of k=0
    # Zero the inverse at k=0 to avoid div-by-zero; k=0 set separately.
    inv_K2 = np.zeros_like(K2)
    nz = K2 > 0.0
    inv_K2[nz] = 1.0 / K2[nz]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Self-layer thin-film shape function
    # f(k,t) -> 1 as Kt -> 0 (uniform); -> 0 as Kt -> inf (short wave).
    f_self = np.ones_like(K)
    nz_kt = Kt > 1e-12
    f_self[nz_kt] = (1.0 - np.exp(-Kt[nz_kt])) / Kt[nz_kt]
    one_m_f = 1.0 - f_self
    # In-plane components share the (1 - f) shape factor.
    Nxx_self = one_m_f * KX * KX * inv_K2
    Nyy_self = one_m_f * KY * KY * inv_K2
    Nxy_self = one_m_f * KX * KY * inv_K2
    # Out-of-plane component uses f directly (complementary projection).
    Nzz_self = f_self.copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k=0 mode: pure uniform shape anisotropy (Nzz=1, in-plane=0)
    # An infinite uniformly magnetized film: only z gets a demag field.
    Nxx_self[~nz] = 0.0
    Nyy_self[~nz] = 0.0
    Nxy_self[~nz] = 0.0
    Nzz_self[~nz] = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inter-layer kernel from surface magnetic charges.
    # (1 - exp(-Kt))^2 / (2 Kt) * exp(-K d_Ru):
    # one (1 - exp(-Kt)) is the source-slab structure factor for
    # the +/- M_z surface charges, the second is the observer-slab
    # thickness average of the exp(-K z) potential, and exp(-K d_Ru)
    # is the decay across the spacer. Vanishes as Kt at small Kt,
    # since an infinite uniformly magnetized slab has no field
    # outside itself.
    S = np.zeros_like(K)
    S[nz_kt] = (
        ((1.0 - np.exp(-Kt[nz_kt])) ** 2)
        / (2.0 * Kt[nz_kt])
        * np.exp(-Kd[nz_kt])
    )
    Nxx_inter = S * KX * KX * inv_K2
    Nyy_inter = S * KY * KY * inv_K2
    Nxy_inter = S * KX * KY * inv_K2
    # Out-of-plane cross term: opposite sign of in-plane.
    # An antiparallel m_z in the other layer attracts (lowers
    # demag energy), so the field on this layer from the
    # other layer's m_z is +mu0*Ms*S*m_z_other (positive when
    # mz_other > 0). With our sign convention H = -mu0 Ms N m
    # this requires N_zz_inter = -S.
    Nzz_inter = -S
    # k=0: an infinite slab produces no external field.
    # All inter-layer components vanish at k=0 (no DC coupling).
    Nxx_inter[~nz] = 0.0
    Nyy_inter[~nz] = 0.0
    Nxy_inter[~nz] = 0.0
    Nzz_inter[~nz] = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Newell-schema cross terms: identically zero for the
    # slab kernel because each layer is treated as infinite
    # in-plane, which decouples in-plane M from out-of-plane
    # H between layers.
    Nxz_inter = np.zeros_like(Nzz_inter)
    Nyz_inter = np.zeros_like(Nzz_inter)
    return {
        'Nxx_self': Nxx_self, 'Nyy_self': Nyy_self,
        'Nxy_self': Nxy_self, 'Nzz_self': Nzz_self,
        'Nxx_inter': Nxx_inter, 'Nyy_inter': Nyy_inter,
        'Nxy_inter': Nxy_inter, 'Nzz_inter': Nzz_inter,
        'Nxz_inter': Nxz_inter, 'Nyz_inter': Nyz_inter,
        'mu0_Ms': p.mu0 * p.Ms,
        't_Co': t, 'd_Ru': d_Ru,
        'shape': (ny, nx),
    }


# ---------------------------------------------------------------------
def demag_field(m_top, m_bot, kernels):
    """Evaluate the demag field on both layers (Tesla).

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    kernels : dict
        Output of `precompute_demag_kernels`.

    Returns
    -------
    H_top : numpy.ndarray(3d)
        Demag field on the top layer in Tesla.
    H_bot : numpy.ndarray(3d)
        Demag field on the bottom layer in Tesla.

    Notes
    -----
    Per-step cost: 6 forward FFTs + 6 inverse FFTs on
    real arrays of shape (ny, nx). Memory cost is
    dominated by the eight cached kernels of the same
    shape.
    """
    mu0_Ms = kernels['mu0_Ms']
    # Forward FFT each component (top and bot)
    # 6 forward FFTs per call; dominant per-step cost of demag.
    # scipy.fft uses pocketfft (same algorithm as np.fft) but
    # supports thread-pool parallelism through `workers`.
    # Setting workers=-1 lets pocketfft use every available
    # core, which gives ~5-8x speed-up on 256x256 grids
    # compared with the single-threaded np.fft.
    # Top
    Mxt = sfft.fft2(m_top[..., 0], workers=-1)
    Myt = sfft.fft2(m_top[..., 1], workers=-1)
    Mzt = sfft.fft2(m_top[..., 2], workers=-1)
    # Bottom
    Mxb = sfft.fft2(m_bot[..., 0], workers=-1)
    Myb = sfft.fft2(m_bot[..., 1], workers=-1)
    Mzb = sfft.fft2(m_bot[..., 2], workers=-1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Self interaction
    Nxx_s = kernels['Nxx_self']
    Nyy_s = kernels['Nyy_self']
    Nxy_s = kernels['Nxy_self']
    Nzz_s = kernels['Nzz_self']
    # Inter-layer interaction
    Nxx_i = kernels['Nxx_inter']
    Nyy_i = kernels['Nyy_inter']
    Nxy_i = kernels['Nxy_inter']
    Nzz_i = kernels['Nzz_inter']
    # Inter-layer cross terms (Newell only; zero for slab).
    # These couple in-plane M of one layer to out-of-plane H
    # of the other and vice versa. They are odd in the z
    # displacement: the kernel is built for source-to-dest
    # displacement +Z (e.g. bot -> top), so the opposite
    # direction (top -> bot) uses the negated kernel.
    Nxz_i = kernels['Nxz_inter']
    Nyz_i = kernels['Nyz_inter']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # H = -mu0*Ms * (N_self * m_self + N_inter * m_other) (k-space)
    # Convolution becomes multiplication in k-space (PBC-justified).
    # Top dest from bottom source uses +Nxz_i, +Nyz_i (source
    # at z=0, dest at z=+Z); bottom dest from top source uses
    # the negated kernel because the z-displacement flips sign.
    # Top
    Hx_t_k = -mu0_Ms * (
        Nxx_s * Mxt + Nxy_s * Myt
        + Nxx_i * Mxb + Nxy_i * Myb + Nxz_i * Mzb)
    Hy_t_k = -mu0_Ms * (
        Nxy_s * Mxt + Nyy_s * Myt
        + Nxy_i * Mxb + Nyy_i * Myb + Nyz_i * Mzb)
    Hz_t_k = -mu0_Ms * (
        Nzz_s * Mzt
        + Nzz_i * Mzb + Nxz_i * Mxb + Nyz_i * Myb)
    # Bottom (sign flip on Nxz_i, Nyz_i for the opposite
    # source-to-dest z-displacement).
    Hx_b_k = -mu0_Ms * (
        Nxx_s * Mxb + Nxy_s * Myb
        + Nxx_i * Mxt + Nxy_i * Myt - Nxz_i * Mzt)
    Hy_b_k = -mu0_Ms * (
        Nxy_s * Mxb + Nyy_s * Myb
        + Nxy_i * Mxt + Nyy_i * Myt - Nyz_i * Mzt)
    Hz_b_k = -mu0_Ms * (
        Nzz_s * Mzb
        + Nzz_i * Mzt - Nxz_i * Mxt - Nyz_i * Myt)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inverse FFT (take real part; imaginary residual is
    # numerical noise from finite precision)
    # 6 inverse FFTs to complete the per-step demag evaluation.
    # Top
    H_top = np.empty_like(m_top)
    H_top[..., 0] = np.real(sfft.ifft2(Hx_t_k, workers=-1))
    H_top[..., 1] = np.real(sfft.ifft2(Hy_t_k, workers=-1))
    H_top[..., 2] = np.real(sfft.ifft2(Hz_t_k, workers=-1))
    # Bottom
    H_bot = np.empty_like(m_bot)
    H_bot[..., 0] = np.real(sfft.ifft2(Hx_b_k, workers=-1))
    H_bot[..., 1] = np.real(sfft.ifft2(Hy_b_k, workers=-1))
    H_bot[..., 2] = np.real(sfft.ifft2(Hz_b_k, workers=-1))
    # Return
    return H_top, H_bot
