/// \file
/// Trajectory post-processing: PBC unwrap, annihilation detection, and
/// the linear-fit Hall angle. Port of the corresponding helpers in
/// src/stochastic_llgs/diagnostics.py.
#pragma once

#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace stochastic {

/// Unwrapped centre history: `cx` and `cy` are equal-length series of
/// x and y positions in metres, free of periodic-image jumps.
struct Trajectory2D { std::vector<double> cx; std::vector<double> cy; };

/// Remove +/-L jumps from a wrapped centre history: displacements
/// larger than L/2 between consecutive samples are treated as PBC
/// wraps. periodic_y == false (free-y racetrack) leaves the y series
/// untouched.
/// \param cx Wrapped x centre samples (m).
/// \param cy Wrapped y centre samples (m); same length as cx.
/// \param L_x Box extent along x (m); finite and > 0.
/// \param L_y Box extent along y (m); finite and > 0.
/// \param periodic_y True to unwrap y as well; false leaves y as is.
/// \return The unwrapped series, first sample copied through.
/// \throws std::runtime_error if L_x or L_y is non-finite or
///         non-positive, or if cx and cy differ in length.
Trajectory2D unwrap_trajectory(const std::vector<double>& cx,
                               const std::vector<double>& cy,
                               double L_x, double L_y, bool periodic_y);

/// First sample index of the first run of `k_consecutive` samples with
/// |Q| < q_threshold; -1 if the skyrmion never annihilates.
/// \param Q Topological-charge samples (dimensionless).
/// \param q_threshold Charge magnitude below which the core counts as
///        gone; finite and > 0.
/// \param k_consecutive Run length required to declare annihilation;
///        must be positive.
/// \return Index of the last sample of the first qualifying run, or -1
///         when no such run exists (including fewer samples than
///         k_consecutive).
/// \throws std::runtime_error if q_threshold is non-finite or
///         non-positive, or if k_consecutive is not positive.
int detect_annihilation(const std::vector<double>& Q,
                        double q_threshold, int k_consecutive);

/// Drift-velocity fit of an unwrapped trajectory: `v_x` and `v_y` are
/// the fitted drift velocities in m/s and `theta_deg` the Hall angle
/// atan2(v_y, |v_x|) in degrees.
struct HallFit { double v_x; double v_y; double theta_deg; };

/// Linear least-squares drift velocity over the final `half` fraction
/// of an unwrapped trajectory, plus the Hall angle atan2(v_y, |v_x|)
/// (deg): the transverse deflection off the drive (x) axis, range
/// (-90, 90].
/// \param t Sample times (s).
/// \param cx Unwrapped x positions (m); same length as t.
/// \param cy Unwrapped y positions (m); same length as t.
/// \param half Trailing fraction of the series to fit, in (0, 1).
/// \return The fitted velocities (m/s) and Hall angle (degrees).
/// \throws std::runtime_error on a length mismatch between t, cx and
///         cy, if half is outside (0, 1), or with fewer than 4
///         samples.
HallFit hall_angle(const std::vector<double>& t,
                   const std::vector<double>& cx,
                   const std::vector<double>& cy, double half);

} // namespace stochastic
} // namespace skyrmion
