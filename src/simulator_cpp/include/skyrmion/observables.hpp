/// \file
/// Skyrmion observables: topological charge, center, diameter, ellipse,
/// domain-wall angle. Mirrors src/skyrmion_simulator/simulator/main.py::topological_charge
/// and src/skyrmion_simulator/simulator/analysis.py.
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {

/// (1/4pi) integral m . (dx m x dy m) dA, central differences with PBC.
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \return Topological charge Q (dimensionless).
Real topological_charge(const Field3& m, Real a);

/// A point in the lattice plane: cx, cy are x and y in metres.
struct Center2D { Real cx; Real cy; };
/// Core centroid weighted by 0.5 * (1 - core_polarity * m_z), so the
/// weight peaks where m_z opposes the background.
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \param core_polarity +1 or -1, the sign of the background m_z.
/// \return The weighted centroid, in metres, without PBC unwrapping.
/// \throws std::runtime_error If core_polarity is neither +1 nor -1,
///         or if the total weight is zero.
Center2D skyrmion_center(const Field3& m, Real a, int core_polarity);

/// PBC-safe centroid via the phase of the weighted circular mean.
/// Returns (cx, cy) modulo (nx*a, ny*a). Matches
/// stochastic_llgs.diagnostics.skyrmion_center_pbc.
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \param core_polarity +1 or -1, the sign of the background m_z.
/// \return The wrapped centroid, in metres.
/// \throws std::runtime_error If core_polarity is neither +1 nor -1,
///         if a <= 0, or if the total weight is non-positive (the
///         skyrmion may have annihilated).
Center2D skyrmion_center_pbc(const Field3& m, Real a, int core_polarity);

/// Equivalent-circle diameter of the reversed-core region, from the
/// area of the cells with core_polarity * m_z < 0.
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \param core_polarity +1 or -1, the sign of the background m_z.
/// \return The diameter 2 sqrt(area / pi), in metres.
/// \throws std::runtime_error If core_polarity is neither +1 nor -1.
Real skyrmion_diameter(const Field3& m, Real a, int core_polarity);

/// Second-moment ellipse of the reversed-core region.
struct Ellipse {
    Real D1;     ///< major-axis diameter (m)
    Real D2;     ///< minor-axis diameter (m)
    Real theta;  ///< major-axis angle wrt +x (radians)
};
/// Diagonalises the covariance of the cells with core_polarity * m_z
/// < 0; the diameters are 4 sqrt(eigenvalue).
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \param core_polarity +1 or -1, the sign of the background m_z.
/// \return Major/minor diameters (m) and orientation (radians).
/// \throws std::runtime_error If core_polarity is neither +1 nor -1,
///         or if fewer than three cells lie inside the core.
Ellipse skyrmion_ellipse(const Field3& m, Real a, int core_polarity);

/// In-plane angle at the right-half-plane DW, sign-folded onto +x.
/// Averages (m_x, m_y) over the cells right of the core centre whose
/// |m_z| is below `mz_thresh`.
/// \param m Magnetisation direction field.
/// \param a Cell size in metres.
/// \param core_polarity +1 or -1, the sign of the background m_z.
/// \param mz_thresh Wall selector: only |m_z| < mz_thresh contributes.
/// \return The in-plane angle, in radians.
/// \throws std::runtime_error If no wall site lies in the +x
///         half-plane, or if core_polarity is neither +1 nor -1.
Real dw_angle(const Field3& m, Real a, int core_polarity, Real mz_thresh);

} // namespace skyrmion
