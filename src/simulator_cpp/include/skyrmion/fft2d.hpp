// Thin FFTW3 wrapper for 2D complex-to-complex FFTs.
//
// Full c2c is used (rather than r2c/c2r) so the demag kernel can be
// stored on the full (ny, nx) spectrum. The slab analytic kernel is
// not Hermitian-conjugate-symmetric at the Nyquist row/column for the
// antisymmetric off-diagonal components (Nxy, Nxz, Nyz); c2r-based
// reconstruction would force a different "effective" kernel there and
// diverge from Python's full-spectrum + np.real() convention.
#pragma once

#include "skyrmion/types.hpp"

#include <complex>
#include <fftw3.h>

namespace skyrmion {

class FFT2D {
public:
    FFT2D(int ny, int nx, int threads);
    ~FFT2D();

    FFT2D(const FFT2D&) = delete;
    FFT2D& operator=(const FFT2D&) = delete;

    int ny() const { return ny_; }
    int nx() const { return nx_; }

    // Scratch buffers used by execute_fwd / execute_inv.
    fftw_complex* scratch_in()  { return scratch_in_;  }
    fftw_complex* scratch_out() { return scratch_out_; }

    void execute_fwd();
    void execute_inv();

private:
    int ny_, nx_;
    fftw_complex* scratch_in_  = nullptr;
    fftw_complex* scratch_out_ = nullptr;
    fftw_plan plan_fwd_        = nullptr;
    fftw_plan plan_inv_        = nullptr;
};

} // namespace skyrmion
