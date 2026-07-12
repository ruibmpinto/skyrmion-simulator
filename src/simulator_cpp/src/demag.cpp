#include "skyrmion/demag.hpp"

#include <cstring>
#include <stdexcept>

namespace skyrmion {

DemagKernels precompute_demag_kernels(const Params& p) {
    switch (p.demag_kind) {
        case DemagKind::Slab:   return precompute_demag_slab(p);
        case DemagKind::Newell: return precompute_demag_newell(p, p.demag_accuracy, p.demag_tol_conv);
        case DemagKind::NewellFreeBC:
            return precompute_demag_newell_freebc(p, p.demag_accuracy, p.demag_tol_conv);
        case DemagKind::Racetrack:
            return precompute_demag_racetrack(p, p.demag_accuracy, p.demag_tol_conv);
        case DemagKind::None:
            throw std::runtime_error("precompute_demag_kernels: demag_kind == None.");
    }
    throw std::runtime_error("precompute_demag_kernels: invalid demag_kind.");
}

DemagState::DemagState(const Params& p, int fft_threads)
    // FFT grid = kernel grid: ny*nx for periodic, the doubled 2N grid for
    // free-BC (k_ is initialized before fft_ by member-declaration order).
    : k_(precompute_demag_kernels(p)), fft_(k_.ny, k_.nx, fft_threads) {
    const std::size_t n = static_cast<std::size_t>(k_.ny) * k_.nx;
    Mxt_.assign(n, Complex{0, 0});
    Myt_.assign(n, Complex{0, 0});
    Mzt_.assign(n, Complex{0, 0});
    Mxb_.assign(n, Complex{0, 0});
    Myb_.assign(n, Complex{0, 0});
    Mzb_.assign(n, Complex{0, 0});
}

void DemagState::transform_layer(const Field3& m,
                                 std::vector<Complex>& Mx,
                                 std::vector<Complex>& My,
                                 std::vector<Complex>& Mz) {
    const int gny = fft_.ny(), gnx = fft_.nx();   // FFT grid (2N if freebc)
    const int pny = m.ny, pnx = m.nx;             // physical field size
    const std::size_t n = static_cast<std::size_t>(gny) * gnx;
    for (int comp = 0; comp < 3; ++comp) {
        fftw_complex* in = fft_.scratch_in();
        // Zero the grid, then place the physical field in the top-left
        // tile. For the periodic kinds gny/gnx == pny/pnx (fills all); for
        // free-BC the remaining cells stay zero (magnetic vacuum).
        for (std::size_t k = 0; k < n; ++k) { in[k][0] = 0.0; in[k][1] = 0.0; }
        for (int i = 0; i < pny; ++i) {
            for (int j = 0; j < pnx; ++j) {
                in[i * gnx + j][0] = m(i, j, comp);
            }
        }
        fft_.execute_fwd();
        std::vector<Complex>& dst = (comp == 0) ? Mx : (comp == 1 ? My : Mz);
        const fftw_complex* src = fft_.scratch_out();
        for (std::size_t k = 0; k < n; ++k) {
            dst[k] = Complex{src[k][0], src[k][1]};
        }
    }
}

void DemagState::inverse_to_layer(std::vector<Complex>& Hx_k,
                                  std::vector<Complex>& Hy_k,
                                  std::vector<Complex>& Hz_k,
                                  Field3& H) {
    const int gny = fft_.ny(), gnx = fft_.nx();   // FFT grid (2N if freebc)
    const int pny = H.ny, pnx = H.nx;             // physical field size
    const std::size_t n = static_cast<std::size_t>(gny) * gnx;
    std::vector<Complex>* arrays[3] = {&Hx_k, &Hy_k, &Hz_k};
    for (int comp = 0; comp < 3; ++comp) {
        fftw_complex* dst = fft_.scratch_in();
        const std::vector<Complex>& src = *arrays[comp];
        for (std::size_t k = 0; k < n; ++k) {
            dst[k][0] = src[k].real();
            dst[k][1] = src[k].imag();
        }
        fft_.execute_inv();
        const fftw_complex* out = fft_.scratch_out();
        // Crop the top-left physical tile (the whole grid for periodic).
        for (int i = 0; i < pny; ++i) {
            for (int j = 0; j < pnx; ++j) {
                // Take real part; imaginary residual is numerical noise
                // plus the non-Hermitian-symmetric Nyquist-row component
                // that Python discards via np.real(ifft2(...)).
                H(i, j, comp) = out[i * gnx + j][0];
            }
        }
    }
}

void DemagState::compute(const Field3& m_top, const Field3& m_bot,
                         Field3& H_top, Field3& H_bot) {
    const int ny = k_.ny;
    const std::size_t n = static_cast<std::size_t>(ny) * k_.nx;
    const Real mu0_Ms = k_.mu0_Ms;

    transform_layer(m_top, Mxt_, Myt_, Mzt_);
    transform_layer(m_bot, Mxb_, Myb_, Mzb_);

    std::vector<Complex> Hx_t(n), Hy_t(n), Hz_t(n);
    std::vector<Complex> Hx_b(n), Hy_b(n), Hz_b(n);

    for (std::size_t k = 0; k < n; ++k) {
        const Complex Nxx_s = k_.Nxx_self[k];
        const Complex Nyy_s = k_.Nyy_self[k];
        const Complex Nxy_s = k_.Nxy_self[k];
        const Complex Nzz_s = k_.Nzz_self[k];
        const Complex Nxx_i = k_.Nxx_inter[k];
        const Complex Nyy_i = k_.Nyy_inter[k];
        const Complex Nxy_i = k_.Nxy_inter[k];
        const Complex Nzz_i = k_.Nzz_inter[k];
        const Complex Nxz_i = k_.Nxz_inter[k];
        const Complex Nyz_i = k_.Nyz_inter[k];

        const Complex Mxt = Mxt_[k], Myt = Myt_[k], Mzt = Mzt_[k];
        const Complex Mxb = Mxb_[k], Myb = Myb_[k], Mzb = Mzb_[k];

        // Top destination: source at bottom uses +Nxz, +Nyz.
        Hx_t[k] = -mu0_Ms * (Nxx_s * Mxt + Nxy_s * Myt
                             + Nxx_i * Mxb + Nxy_i * Myb + Nxz_i * Mzb);
        Hy_t[k] = -mu0_Ms * (Nxy_s * Mxt + Nyy_s * Myt
                             + Nxy_i * Mxb + Nyy_i * Myb + Nyz_i * Mzb);
        Hz_t[k] = -mu0_Ms * (Nzz_s * Mzt
                             + Nzz_i * Mzb + Nxz_i * Mxb + Nyz_i * Myb);
        // Bottom destination: opposite z-displacement -> -Nxz, -Nyz.
        Hx_b[k] = -mu0_Ms * (Nxx_s * Mxb + Nxy_s * Myb
                             + Nxx_i * Mxt + Nxy_i * Myt - Nxz_i * Mzt);
        Hy_b[k] = -mu0_Ms * (Nxy_s * Mxb + Nyy_s * Myb
                             + Nxy_i * Mxt + Nyy_i * Myt - Nyz_i * Mzt);
        Hz_b[k] = -mu0_Ms * (Nzz_s * Mzb
                             + Nzz_i * Mzt - Nxz_i * Mxt - Nyz_i * Myt);
    }

    inverse_to_layer(Hx_t, Hy_t, Hz_t, H_top);
    inverse_to_layer(Hx_b, Hy_b, Hz_b, H_bot);
}

} // namespace skyrmion
