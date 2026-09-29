/// \file
/// Largest-connected-component (LCC) core diagnostics under periodic
/// boundary conditions. Port of the LCC helpers in
/// src/skyrmion_simulator/stochastic_llgs/diagnostics.py: a 4-connectivity labelling with
/// union-find fusion across both wraps, then the single largest
/// component. These strip thermal-noise blobs from the
/// diameter/centre estimates.
#pragma once

#include "skyrmion/observables.hpp"   // Center2D
#include "skyrmion/types.hpp"

#include <cstdint>
#include <vector>

namespace skyrmion {
namespace stochastic {

/// Boolean mask (row-major ny*nx, 1 = in component) of the largest
/// connected core component. core = (core_polarity * m_z < 0).
/// 4-connected, PBC-fused. Empty core => all zeros.
///
/// The raster-order labelling matches scipy.ndimage.label numbering
/// and the union-find root is the smallest label in a group, so the
/// largest-component tie-break picks the same component as Python.
/// \param m Magnetization field, shape (ny, nx, 3).
/// \param core_polarity +1 or -1; the core is where
///        core_polarity * m_z < 0.
/// \return Row-major ny*nx mask, 1 inside the largest component.
/// \throws std::runtime_error if core_polarity is neither +1 nor -1.
std::vector<std::uint8_t> largest_core_mask_pbc(const Field3& m,
                                                int core_polarity);

/// Equivalent-disk diameter of the LCC (m); 0 if no core cells.
/// \param m Magnetization field, shape (ny, nx, 3).
/// \param a Lattice constant (m); must be > 0.
/// \param core_polarity +1 or -1, as for largest_core_mask_pbc.
/// \return 2 * sqrt(area / pi) with area = n_cells * a^2, in metres.
/// \throws std::runtime_error if a is not positive, or if
///         core_polarity is neither +1 nor -1.
Real skyrmion_diameter_lcc(const Field3& m, Real a, int core_polarity);

/// PBC circular-mean centre restricted to the LCC (weight = LCC
/// indicator). Raises on an empty core mask.
/// \param m Magnetization field, shape (ny, nx, 3).
/// \param a Lattice constant (m); must be > 0.
/// \param core_polarity +1 or -1, as for largest_core_mask_pbc.
/// \return Centre (cx, cy) in metres, wrapped into [0, nx*a) and
///         [0, ny*a).
/// \throws std::runtime_error if a is not positive, if core_polarity
///         is invalid, or if the core mask is empty (the skyrmion may
///         have annihilated).
Center2D skyrmion_center_lcc_pbc(const Field3& m, Real a, int core_polarity);

/// Major/minor diameters (D1 >= D2) and major-axis angle of the LCC
/// via PBC-aware second moments. Thermal-noise-robust counterpart of
/// skyrmion_ellipse. Raises if the LCC has fewer than 3 sites. Port of
/// src/skyrmion_simulator/stochastic_llgs/diagnostics.py::skyrmion_ellipse_lcc.
///
/// Site positions are unwrapped about the per-axis circular mean, so a
/// component straddling a wrap collapses into one contiguous window
/// before the covariance is taken.
/// \param m Magnetization field, shape (ny, nx, 3).
/// \param a Lattice constant (m); must be > 0.
/// \param core_polarity +1 or -1, as for largest_core_mask_pbc.
/// \return D1 and D2 in metres (4*sqrt of the covariance eigenvalues)
///         and theta, the major-axis angle wrt +x in radians.
/// \throws std::runtime_error if a is not positive, if core_polarity
///         is invalid, or if the largest component has fewer than 3
///         sites.
Ellipse skyrmion_ellipse_lcc(const Field3& m, Real a, int core_polarity);

} // namespace stochastic
} // namespace skyrmion
