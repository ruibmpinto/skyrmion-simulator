// Skyrmion observables: topological charge, center, diameter, ellipse,
// domain-wall angle. Mirrors src/simulator/main.py::topological_charge
// and src/simulator/analysis.py.
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {

// (1/4pi) integral m . (dx m x dy m) dA, central differences with PBC.
Real topological_charge(const Field3& m, Real a);

struct Center2D { Real cx; Real cy; };
Center2D skyrmion_center(const Field3& m, Real a, int core_polarity);

// PBC-safe centroid via the phase of the weighted circular mean.
// Returns (cx, cy) modulo (nx*a, ny*a). Matches
// stochastic_llgs.diagnostics.skyrmion_center_pbc.
Center2D skyrmion_center_pbc(const Field3& m, Real a, int core_polarity);

Real skyrmion_diameter(const Field3& m, Real a, int core_polarity);

struct Ellipse {
    Real D1;     // major-axis diameter (m)
    Real D2;     // minor-axis diameter (m)
    Real theta;  // major-axis angle wrt +x (radians)
};
Ellipse skyrmion_ellipse(const Field3& m, Real a, int core_polarity);

// In-plane angle at the right-half-plane DW, sign-folded onto +x.
Real dw_angle(const Field3& m, Real a, int core_polarity, Real mz_thresh);

} // namespace skyrmion
