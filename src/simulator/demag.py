"""Magnetostatic (dipolar) field for thin-film SAF stacks.

Computes the long-range demagnetizing field in 2D Fourier
space for two stacked thin Co layers separated by a Ru
spacer. Each layer is treated as a slab of thickness `t_Co`;
the in-plane lattice has periodic boundary conditions, so
plane-wave Fourier kernels apply directly.

The kernel uses the standard thin-film shape function
    f(k, t) = (1 - exp(-|k|*t)) / (|k|*t),
which yields the correct limits:
    |k|*t -> 0  : N_zz = 1 (uniform shape anisotropy
                  -mu0*Ms*m_z ).
    |k|*t -> inf: N_zz = 0 (no demag for textures much
                  shorter than the thickness).
The off-diagonal in-plane components reduce to the
divergence-free magnetostatic Green's tensor.

The interlayer kernel between two slabs of thickness `t_Co`
separated by a center-to-center gap `d_Ru` uses the same
shape factor squared multiplied by exp(-|k|*d_Ru); at k=0 it
vanishes (an infinite slab produces no field outside itself).

Functions
---------
precompute_demag_kernels
    Build self- and inter-layer demag tensors on the FFT
    grid for a given parameter namespace.
demag_field
    Evaluate the demag field on both layers from cached
    kernels.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def precompute_demag_kernels(p):
    """Build self- and inter-layer demag kernels in k-space.

    Parameters
    ----------
    p : SimpleNamespace
        Parameters namespace. Must expose `nx`, `ny`, `a`,
        `t_Co`, `d_Ru`, `Ms`, and `mu0`. A missing attribute
        raises `AttributeError` at access time so silent
        defaulting does not corrupt the kernels.

    Returns
    -------
    kernels : dict
        Dictionary holding the four self-layer tensor
        components and four inter-layer components on a 2D
        FFT grid of shape (ny, nx), plus the bulk
        coefficients `Ms` and `mu0` needed at field-eval
        time.

    Notes
    -----
    Convention: in real space H_demag = -mu0 * (N * m), so
    in k-space H(k) = -mu0 * Ms * N(k) * m(k). N is
    dimensionless; the prefactor mu0*Ms is applied inside
    `demag_field`.
    """
    nx, ny = p.nx, p.ny
    a = p.a
    t = p.t_Co
    d_Ru = p.d_Ru
    if t <= 0.0:
        raise RuntimeError(
            f'p.t_Co must be positive, got {t}.'
        )
    if d_Ru < 0.0:
        raise RuntimeError(
            f'p.d_Ru must be non-negative, got {d_Ru}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2D k-grid in rad/m
    kx = 2.0 * np.pi * np.fft.fftfreq(nx, d=a)
    ky = 2.0 * np.pi * np.fft.fftfreq(ny, d=a)
    KX, KY = np.meshgrid(kx, ky, indexing='xy')
    K2 = KX * KX + KY * KY
    K = np.sqrt(K2)
    Kt = K * t
    Kd = K * d_Ru
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 1/K^2 with safe handling of k=0
    inv_K2 = np.zeros_like(K2)
    nz = K2 > 0.0
    inv_K2[nz] = 1.0 / K2[nz]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Self-layer thin-film shape function
    f_self = np.ones_like(K)
    nz_kt = Kt > 1e-12
    f_self[nz_kt] = (1.0 - np.exp(-Kt[nz_kt])) / Kt[nz_kt]
    one_m_f = 1.0 - f_self
    Nxx_self = one_m_f * KX * KX * inv_K2
    Nyy_self = one_m_f * KY * KY * inv_K2
    Nxy_self = one_m_f * KX * KY * inv_K2
    Nzz_self = f_self.copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # k=0 mode: pure uniform shape anisotropy (Nzz=1, in-plane=0)
    Nxx_self[~nz] = 0.0
    Nyy_self[~nz] = 0.0
    Nxy_self[~nz] = 0.0
    Nzz_self[~nz] = 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inter-layer kernel: same shape factor squared * exp(-K*d_Ru)
    S = np.zeros_like(K)
    S[nz_kt] = (
        ((1.0 - np.exp(-Kt[nz_kt])) ** 2)
        / (Kt[nz_kt] ** 2)
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
    Nxx_inter[~nz] = 0.0
    Nyy_inter[~nz] = 0.0
    Nxy_inter[~nz] = 0.0
    Nzz_inter[~nz] = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    return {
        'Nxx_self': Nxx_self, 'Nyy_self': Nyy_self,
        'Nxy_self': Nxy_self, 'Nzz_self': Nzz_self,
        'Nxx_inter': Nxx_inter, 'Nyy_inter': Nyy_inter,
        'Nxy_inter': Nxy_inter, 'Nzz_inter': Nzz_inter,
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
    Mxt = np.fft.fft2(m_top[..., 0])
    Myt = np.fft.fft2(m_top[..., 1])
    Mzt = np.fft.fft2(m_top[..., 2])
    Mxb = np.fft.fft2(m_bot[..., 0])
    Myb = np.fft.fft2(m_bot[..., 1])
    Mzb = np.fft.fft2(m_bot[..., 2])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    Nxx_s = kernels['Nxx_self']
    Nyy_s = kernels['Nyy_self']
    Nxy_s = kernels['Nxy_self']
    Nzz_s = kernels['Nzz_self']
    Nxx_i = kernels['Nxx_inter']
    Nyy_i = kernels['Nyy_inter']
    Nxy_i = kernels['Nxy_inter']
    Nzz_i = kernels['Nzz_inter']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # H = -mu0*Ms * (N_self * m_self + N_inter * m_other) (k-space)
    Hx_t_k = -mu0_Ms * (
        Nxx_s * Mxt + Nxy_s * Myt
        + Nxx_i * Mxb + Nxy_i * Myb
    )
    Hy_t_k = -mu0_Ms * (
        Nxy_s * Mxt + Nyy_s * Myt
        + Nxy_i * Mxb + Nyy_i * Myb
    )
    Hz_t_k = -mu0_Ms * (
        Nzz_s * Mzt + Nzz_i * Mzb
    )
    Hx_b_k = -mu0_Ms * (
        Nxx_s * Mxb + Nxy_s * Myb
        + Nxx_i * Mxt + Nxy_i * Myt
    )
    Hy_b_k = -mu0_Ms * (
        Nxy_s * Mxb + Nyy_s * Myb
        + Nxy_i * Mxt + Nyy_i * Myt
    )
    Hz_b_k = -mu0_Ms * (
        Nzz_s * Mzb + Nzz_i * Mzt
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Inverse FFT (take real part; imaginary residual is
    # numerical noise from finite precision)
    H_top = np.empty_like(m_top)
    H_top[..., 0] = np.real(np.fft.ifft2(Hx_t_k))
    H_top[..., 1] = np.real(np.fft.ifft2(Hy_t_k))
    H_top[..., 2] = np.real(np.fft.ifft2(Hz_t_k))
    H_bot = np.empty_like(m_bot)
    H_bot[..., 0] = np.real(np.fft.ifft2(Hx_b_k))
    H_bot[..., 1] = np.real(np.fft.ifft2(Hy_b_k))
    H_bot[..., 2] = np.real(np.fft.ifft2(Hz_b_k))
    return H_top, H_bot
