// Analytic slab demag kernel built on the full (ny, nx) c2c grid.
// Mirrors src/simulator/demag.py _precompute_slab, with 1/(ny*nx)
// pre-scaling baked in so the unnormalised c2c inverse transform
// yields the physical real-space field.
#include "skyrmion/demag.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

namespace skyrmion {

namespace {

constexpr Real kPi = 3.14159265358979323846;

inline Real fft_freq(int i, int n) {
    // Matches numpy.fft.fftfreq * n: for even n, index n/2 maps to -n/2.
    const int half = (n + 1) / 2;
    return (i < half) ? static_cast<Real>(i) : static_cast<Real>(i - n);
}

} // namespace

DemagKernels precompute_demag_slab(const Params& p) {
    if (p.t_Co <= 0.0) {
        throw std::runtime_error("demag_slab: t_Co must be positive.");
    }
    if (p.d_Ru < 0.0) {
        throw std::runtime_error("demag_slab: d_Ru must be non-negative.");
    }
    const int ny = p.ny, nx = p.nx;
    const Real norm = 1.0 / (static_cast<Real>(ny) * nx);

    DemagKernels K;
    K.ny = ny; K.nx = nx;
    K.mu0_Ms = p.mu0 * p.Ms;
    K.t_Co = p.t_Co;
    K.d_Ru = p.d_Ru;

    const std::size_t n = static_cast<std::size_t>(ny) * nx;
    K.Nxx_self.assign(n, Complex{0, 0});
    K.Nyy_self.assign(n, Complex{0, 0});
    K.Nxy_self.assign(n, Complex{0, 0});
    K.Nzz_self.assign(n, Complex{0, 0});
    K.Nxx_inter.assign(n, Complex{0, 0});
    K.Nyy_inter.assign(n, Complex{0, 0});
    K.Nxy_inter.assign(n, Complex{0, 0});
    K.Nzz_inter.assign(n, Complex{0, 0});
    K.Nxz_inter.assign(n, Complex{0, 0});
    K.Nyz_inter.assign(n, Complex{0, 0});

    const Real dkx = 2.0 * kPi / (nx * p.a);
    const Real dky = 2.0 * kPi / (ny * p.a);

    for (int i = 0; i < ny; ++i) {
        const Real ky = fft_freq(i, ny) * dky;
        for (int j = 0; j < nx; ++j) {
            const Real kx = fft_freq(j, nx) * dkx;
            const Real k2 = kx * kx + ky * ky;
            const Real k  = std::sqrt(k2);
            const Real kt = k * p.t_Co;
            const Real kd = k * p.d_Ru;

            Real Nxx_s, Nyy_s, Nxy_s, Nzz_s;
            Real Nxx_i, Nyy_i, Nxy_i, Nzz_i;
            Real Nxz_i_im, Nyz_i_im;

            if (k2 > 0.0) {
                const Real inv_k2 = 1.0 / k2;
                const Real f_self = (kt > 1e-12) ? (1.0 - std::exp(-kt)) / kt : 1.0;
                const Real one_m_f = 1.0 - f_self;
                Nxx_s = one_m_f * kx * kx * inv_k2;
                Nyy_s = one_m_f * ky * ky * inv_k2;
                Nxy_s = one_m_f * kx * ky * inv_k2;
                Nzz_s = f_self;

                Real S = 0.0;
                if (kt > 1e-12) {
                    const Real one_m_e = 1.0 - std::exp(-kt);
                    S = (one_m_e * one_m_e) / (2.0 * kt) * std::exp(-kd);
                }
                Nxx_i = S * kx * kx * inv_k2;
                Nyy_i = S * ky * ky * inv_k2;
                Nxy_i = S * kx * ky * inv_k2;
                Nzz_i = -S;
                // Analytic inter-layer cross terms N_xz = i (kx/k) S,
                // N_yz = i (ky/k) S: same magnitude S as the retained
                // terms, imaginary (lateral-shift kernel), built for
                // the +Z (bot -> top) direction like the Newell kernel.
                const Real inv_k = 1.0 / k;
                Nxz_i_im = kx * inv_k * S;
                Nyz_i_im = ky * inv_k * S;
            } else {
                Nxx_s = 0.0; Nyy_s = 0.0; Nxy_s = 0.0; Nzz_s = 1.0;
                Nxx_i = 0.0; Nyy_i = 0.0; Nxy_i = 0.0; Nzz_i = 0.0;
                Nxz_i_im = 0.0; Nyz_i_im = 0.0;
            }

            const std::size_t idx = static_cast<std::size_t>(i) * nx + j;
            K.Nxx_self[idx]  = Complex{Nxx_s * norm, 0.0};
            K.Nyy_self[idx]  = Complex{Nyy_s * norm, 0.0};
            K.Nxy_self[idx]  = Complex{Nxy_s * norm, 0.0};
            K.Nzz_self[idx]  = Complex{Nzz_s * norm, 0.0};
            K.Nxx_inter[idx] = Complex{Nxx_i * norm, 0.0};
            K.Nyy_inter[idx] = Complex{Nyy_i * norm, 0.0};
            K.Nxy_inter[idx] = Complex{Nxy_i * norm, 0.0};
            K.Nzz_inter[idx] = Complex{Nzz_i * norm, 0.0};
            K.Nxz_inter[idx] = Complex{0.0, Nxz_i_im * norm};
            K.Nyz_inter[idx] = Complex{0.0, Nyz_i_im * norm};
        }
    }
    return K;
}

} // namespace skyrmion
