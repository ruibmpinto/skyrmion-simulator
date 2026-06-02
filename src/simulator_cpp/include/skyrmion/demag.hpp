// Magnetostatic (dipolar) field for thin-film SAF stacks.
//
// Kernel full-spectrum storage of shape (ny, nx) complex matches FFTW
// c2c output. Each kernel value is pre-scaled by 1/(ny*nx) so the
// inverse c2c transform yields the physical real-space field with no
// extra normalization step. Full c2c (rather than r2c/c2r) is required
// because the analytic slab kernel for the off-diagonal components is
// not Hermitian-conjugate-symmetric at the Nyquist row/column.
//
// Sign convention: H_demag = -mu0 * Ms * N * m (matches Python slab kernel).
#pragma once

#include "skyrmion/fft2d.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <complex>
#include <memory>
#include <vector>

namespace skyrmion {

using Complex = std::complex<Real>;

struct DemagKernels {
    int ny = 0, nx = 0;
    Real mu0_Ms = 0.0;
    Real t_Co = 0.0, d_Ru = 0.0;
    // Each (ny * nx) entries, full k-space spectrum.
    std::vector<Complex> Nxx_self, Nyy_self, Nxy_self, Nzz_self;
    std::vector<Complex> Nxx_inter, Nyy_inter, Nxy_inter, Nzz_inter;
    std::vector<Complex> Nxz_inter, Nyz_inter;
};

class DemagState {
public:
    explicit DemagState(const Params& p, int fft_threads);
    // Computes the demag field for both layers. H_top, H_bot are
    // allocated by the caller with shape (ny, nx).
    void compute(const Field3& m_top, const Field3& m_bot,
                 Field3& H_top, Field3& H_bot);
private:
    DemagKernels k_;
    FFT2D fft_;
    // Cached spectra of m for both layers.
    std::vector<Complex> Mxt_, Myt_, Mzt_, Mxb_, Myb_, Mzb_;
    void transform_layer(const Field3& m,
                         std::vector<Complex>& Mx,
                         std::vector<Complex>& My,
                         std::vector<Complex>& Mz);
    void inverse_to_layer(std::vector<Complex>& Hx_k,
                          std::vector<Complex>& Hy_k,
                          std::vector<Complex>& Hz_k,
                          Field3& H);
};

// Build the analytic slab-approximation demag kernel.
DemagKernels precompute_demag_slab(const Params& p);

// Newell kernel: not implemented in this C++ port. Throws.
DemagKernels precompute_demag_newell(const Params& p, Real accuracy, Real tol_conv);

// Dispatcher matching the Python signature.
DemagKernels precompute_demag_kernels(const Params& p);

} // namespace skyrmion
