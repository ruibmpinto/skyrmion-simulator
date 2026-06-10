// Largest-connected-component (LCC) core diagnostics under periodic
// boundary conditions. Port of the LCC helpers in
// src/stochastic_llgs/diagnostics.py: a 4-connectivity labelling with
// union-find fusion across both wraps, then the single largest
// component. These strip thermal-noise blobs from the diameter/centre
// estimates.
#pragma once

#include "skyrmion/observables.hpp"   // Center2D
#include "skyrmion/types.hpp"

#include <cstdint>
#include <vector>

namespace skyrmion {
namespace stochastic {

// Boolean mask (row-major ny*nx, 1 = in component) of the largest
// connected core component. core = (core_polarity * m_z < 0).
// 4-connected, PBC-fused. Empty core => all zeros.
std::vector<std::uint8_t> largest_core_mask_pbc(const Field3& m,
                                                int core_polarity);

// Equivalent-disk diameter of the LCC (m); 0 if no core cells.
Real skyrmion_diameter_lcc(const Field3& m, Real a, int core_polarity);

// PBC circular-mean centre restricted to the LCC (weight = LCC
// indicator). Raises on an empty core mask.
Center2D skyrmion_center_lcc_pbc(const Field3& m, Real a, int core_polarity);

// Major/minor diameters (D1 >= D2) and major-axis angle of the LCC via
// PBC-aware second moments. Thermal-noise-robust counterpart of
// skyrmion_ellipse. Raises if the LCC has fewer than 3 sites. Port of
// src/stochastic_llgs/diagnostics.py::skyrmion_ellipse_lcc.
Ellipse skyrmion_ellipse_lcc(const Field3& m, Real a, int core_polarity);

} // namespace stochastic
} // namespace skyrmion
