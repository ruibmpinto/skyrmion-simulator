/// \file
/// Thin FFTW3 wrapper for 2D complex-to-complex FFTs.
///
/// Full c2c is used (rather than r2c/c2r) so the demag kernel can be
/// stored on the full (ny, nx) spectrum. The slab analytic kernel is
/// not Hermitian-conjugate-symmetric at the Nyquist row/column for the
/// antisymmetric off-diagonal components (Nxy, Nxz, Nyz); c2r-based
/// reconstruction would force a different "effective" kernel there and
/// diverge from Python's full-spectrum + np.real() convention.
#pragma once

#include "skyrmion/types.hpp"

#include <complex>
#include <fftw3.h>

namespace skyrmion {

/// Owns a pair of FFTW plans and their scratch buffers for one fixed
/// (ny, nx) transform size. Non-copyable: the plans and buffers are raw
/// FFTW resources released in the destructor.
class FFT2D {
public:
    /// Create forward and inverse plans for an ny-by-nx c2c transform.
    /// \param ny Number of rows.
    /// \param nx Number of columns.
    /// \param threads FFTW thread count; 1 disables threading.
    FFT2D(int ny, int nx, int threads);
    /// Destroy the plans and free the scratch buffers.
    ~FFT2D();

    FFT2D(const FFT2D&) = delete;
    FFT2D& operator=(const FFT2D&) = delete;

    /// Number of rows of the transform.
    int ny() const { return ny_; }
    /// Number of columns of the transform.
    int nx() const { return nx_; }

    /// Input scratch buffer, (ny * nx) complex entries in row-major
    /// order. Fill this before calling execute_fwd or execute_inv.
    fftw_complex* scratch_in()  { return scratch_in_;  }
    /// Output scratch buffer holding the result of the last execute
    /// call, (ny * nx) complex entries in row-major order.
    fftw_complex* scratch_out() { return scratch_out_; }

    /// Transform scratch_in into scratch_out, unnormalized.
    void execute_fwd();
    /// Inverse-transform scratch_in into scratch_out. FFTW leaves the
    /// result scaled by ny * nx; the 1 / (ny * nx) factor is folded
    /// into the precomputed demag kernel instead of being applied
    /// here.
    void execute_inv();

private:
    int ny_, nx_;
    fftw_complex* scratch_in_  = nullptr;
    fftw_complex* scratch_out_ = nullptr;
    fftw_plan plan_fwd_        = nullptr;
    fftw_plan plan_inv_        = nullptr;
};

} // namespace skyrmion
