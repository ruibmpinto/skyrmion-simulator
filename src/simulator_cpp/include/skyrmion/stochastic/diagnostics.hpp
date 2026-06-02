// Trajectory post-processing: PBC unwrap, annihilation detection, and
// the linear-fit Hall angle. Port of the corresponding helpers in
// src/stochastic_llgs/diagnostics.py.
#pragma once

#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace stochastic {

struct Trajectory2D { std::vector<double> cx; std::vector<double> cy; };

// Remove +/-L jumps from a wrapped centre history: displacements larger
// than L/2 between consecutive samples are treated as PBC wraps.
Trajectory2D unwrap_trajectory(const std::vector<double>& cx,
                               const std::vector<double>& cy,
                               double L_x, double L_y);

// First sample index of the first run of `k_consecutive` samples with
// |Q| < q_threshold; -1 if the skyrmion never annihilates.
int detect_annihilation(const std::vector<double>& Q,
                        double q_threshold, int k_consecutive);

struct HallFit { double v_x; double v_y; double theta_deg; };

// Linear least-squares drift velocity over the final `half` fraction of
// an unwrapped trajectory, plus the Hall angle atan2(v_y, v_x) (deg).
HallFit hall_angle(const std::vector<double>& t,
                   const std::vector<double>& cx,
                   const std::vector<double>& cy, double half);

} // namespace stochastic
} // namespace skyrmion
