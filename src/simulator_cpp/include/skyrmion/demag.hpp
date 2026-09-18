/// \file
/// Magnetostatic (dipolar) field for thin-film SAF stacks.
///
/// Kernel full-spectrum storage of shape (ny, nx) complex matches FFTW
/// c2c output. Each kernel value is pre-scaled by 1/(ny*nx) so the
/// inverse c2c transform yields the physical real-space field with no
/// extra normalization step. Full c2c (rather than r2c/c2r) is required
/// because the analytic slab kernel for the off-diagonal components is
/// not Hermitian-conjugate-symmetric at the Nyquist row/column.
///
/// Sign convention: H_demag = -mu0 * Ms * N * m (matches Python slab kernel).
#pragma once

#include "skyrmion/fft2d.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <complex>
#include <memory>
#include <vector>

namespace skyrmion {

/// Complex scalar used for the k-space kernels and spectra.
using Complex = std::complex<Real>;

/// Precomputed k-space demag tensor for one SAF stack: the intra-layer
/// (self) block and the inter-layer block, both pre-scaled by
/// 1/(ny*nx) for the unnormalized inverse c2c transform.
struct DemagKernels {
    /// ny/nx are the FFT grid size: equal to the physical grid for the
    /// periodic kinds, but the DOUBLED (2*phys) grid for free-BC.
    int ny = 0, nx = 0;
    /// Free-boundary (zero-padded) kernel: ny/nx are 2*phys and the demag
    /// is the isolated (no-image) field; ny_phys/nx_phys are the physical
    /// size to pad into / crop out. freebc=false => periodic (ny==ny_phys).
    bool freebc = false;
    int ny_phys = 0, nx_phys = 0;  ///< Physical grid to pad into / crop.
    Real mu0_Ms = 0.0;   ///< mu0 * Ms (T), the demag field scale.
    Real t_Co = 0.0, d_Ru = 0.0;  ///< Layer and spacer thickness (m).
    /// Each (ny * nx) entries, full k-space spectrum.
    std::vector<Complex> Nxx_self, Nyy_self, Nxy_self, Nzz_self;
    /// Inter-layer in-plane and out-of-plane diagonal blocks.
    std::vector<Complex> Nxx_inter, Nyy_inter, Nxy_inter, Nzz_inter;
    /// Inter-layer cross terms; they change sign for the bottom layer.
    std::vector<Complex> Nxz_inter, Nyz_inter;
};

/// Owns the demag kernel, the FFTW plans and the scratch spectra, and
/// evaluates the magnetostatic field of both layers per call.
class DemagState {
public:
    /// Builds the kernel selected by `p.demag_kind` and the matching
    /// FFT plans. The FFT grid equals the kernel grid (the doubled 2N
    /// grid for the free-boundary kinds).
    /// \param p Parameter set: geometry, Ms, thicknesses, demag_kind.
    /// \param fft_threads Number of FFTW threads to plan for.
    /// \throws std::runtime_error If `p.demag_kind` is None, or if the
    ///         kernel builder rejects the parameters.
    explicit DemagState(const Params& p, int fft_threads);
    /// Computes the demag field for both layers. H_top, H_bot are
    /// allocated by the caller with shape (ny, nx).
    /// \param m_top Top-layer magnetisation direction field.
    /// \param m_bot Bottom-layer magnetisation direction field.
    /// \param H_top Top-layer demag field (T), overwritten.
    /// \param H_bot Bottom-layer demag field (T), overwritten.
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

/// Build the analytic slab-approximation demag kernel.
/// \param p Parameter set: grid, cell size, Ms, t_Co, d_Ru.
/// \return The k-space kernel on the ny*nx periodic grid.
/// \throws std::runtime_error If t_Co <= 0 or d_Ru < 0.
DemagKernels precompute_demag_slab(const Params& p);

/// Newell kernel, periodic (circular-convolution) on the ny*nx grid.
/// Periodic images are summed `p.pbc_images` wraps deep along both
/// directions. With DemagMethod::Closed the kernel is built once; with
/// Quadrature it is built at `accuracy` and at `2 * accuracy` and the
/// two are compared before the finer one is returned.
/// \param p Parameter set: grid, cell size, Ms, thicknesses, method.
/// \param accuracy Quadrature node density (nodes per edge-to-edge
///        distance); larger is finer. Ignored by the closed form.
/// \param tol_conv Maximum relative difference allowed between the
///        two quadrature refinements, in (0, 1).
/// \return The k-space kernel on the ny*nx periodic grid.
/// \throws std::runtime_error If accuracy <= 0, tol_conv is outside
///         (0, 1), t_Co <= 0, d_Ru < 0, nx or ny < 2, or the
///         quadrature convergence check fails.
DemagKernels precompute_demag_newell(const Params& p, Real accuracy, Real tol_conv);

/// Newell kernel, free-boundary: built on a 2N zero-padded grid for an
/// isolated (no periodic image) linear convolution. Port of Python
/// precompute_demag_kernels_newell_freebc.
/// \param p Parameter set: grid, cell size, Ms, thicknesses, method.
/// \param accuracy Quadrature node density; ignored by the closed form.
/// \param tol_conv Maximum relative difference between the two
///        quadrature refinements, in (0, 1).
/// \return The k-space kernel on the doubled (2*ny, 2*nx) grid.
/// \throws std::runtime_error On the same invalid parameters as
///         precompute_demag_newell, or on a failed convergence check.
DemagKernels precompute_demag_newell_freebc(const Params& p, Real accuracy,
                                            Real tol_conv);

/// Newell kernel, mixed track BC: periodic along x (grid width nx,
/// circular convolution) and free/isolated along y (grid height 2*ny,
/// zero-padded). Strip infinite along x, open across its width.
/// Images are summed along x only, `p.pbc_images` wraps deep.
/// \param p Parameter set: grid, cell size, Ms, thicknesses, method.
/// \param accuracy Quadrature node density; ignored by the closed form.
/// \param tol_conv Maximum relative difference between the two
///        quadrature refinements, in (0, 1).
/// \return The k-space kernel on the (2*ny, nx) grid.
/// \throws std::runtime_error On the same invalid parameters as
///         precompute_demag_newell, or on a failed convergence check.
DemagKernels precompute_demag_racetrack(const Params& p, Real accuracy,
                                              Real tol_conv);

/// Dispatcher matching the Python signature.
/// Selects the kernel builder from `p.demag_kind`, passing
/// `p.demag_accuracy` and `p.demag_tol_conv` to the Newell variants.
/// \param p Parameter set.
/// \return The k-space kernel for the configured geometry.
/// \throws std::runtime_error If `p.demag_kind` is None, or if the
///         selected builder rejects the parameters.
DemagKernels precompute_demag_kernels(const Params& p);

} // namespace skyrmion
