#include "skyrmion/fft2d.hpp"

#include <cstdlib>
#include <mutex>
#include <stdexcept>

namespace skyrmion {

namespace { std::once_flag g_threads_init; }

FFT2D::FFT2D(int ny, int nx, int threads) : ny_(ny), nx_(nx) {
    if (ny <= 0 || nx <= 0) {
        throw std::runtime_error("FFT2D: ny, nx must be positive.");
    }
#ifdef SKYRMION_FFTW_THREADS
    std::call_once(g_threads_init, []() {
        if (!fftw_init_threads()) {
            throw std::runtime_error("FFT2D: fftw_init_threads failed.");
        }
    });
    if (threads > 0) fftw_plan_with_nthreads(threads);
#else
    (void)threads;
#endif
    const std::size_t n = static_cast<std::size_t>(ny) * nx;
    scratch_in_  = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    scratch_out_ = static_cast<fftw_complex*>(fftw_malloc(sizeof(fftw_complex) * n));
    if (!scratch_in_ || !scratch_out_) {
        throw std::runtime_error("FFT2D: fftw_malloc failed.");
    }
    plan_fwd_ = fftw_plan_dft_2d(ny, nx, scratch_in_, scratch_out_,
                                 FFTW_FORWARD, FFTW_MEASURE);
    plan_inv_ = fftw_plan_dft_2d(ny, nx, scratch_in_, scratch_out_,
                                 FFTW_BACKWARD, FFTW_MEASURE);
    if (!plan_fwd_ || !plan_inv_) {
        throw std::runtime_error("FFT2D: fftw_plan creation failed.");
    }
}

FFT2D::~FFT2D() {
    if (plan_fwd_) fftw_destroy_plan(plan_fwd_);
    if (plan_inv_) fftw_destroy_plan(plan_inv_);
    if (scratch_in_)  fftw_free(scratch_in_);
    if (scratch_out_) fftw_free(scratch_out_);
}

void FFT2D::execute_fwd() { fftw_execute(plan_fwd_); }
void FFT2D::execute_inv() { fftw_execute(plan_inv_); }

} // namespace skyrmion
