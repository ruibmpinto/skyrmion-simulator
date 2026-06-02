// Two-skyrmion initial condition for the inter-skyrmion potential scan.
// Port of the IC helpers in
// src/stochastic_llgs/production/pair_potential.py.
#pragma once

#include "skyrmion/initial_conditions.hpp"   // SAFPair
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

// Single Neel skyrmion centred at (cx, cy) in metres (cf.
// skyrmion_profile, which centres at the box midpoint).
Field3 skyrmion_at_position(int nx, int ny, Real a, Real R, Real dw,
                            int polarity, Real cx, Real cy);

// SAF pair containing two skyrmions at c1 and c2. The two single-
// skyrmion fields are merged per site by the core-favouring z rule
// used in pair_potential.py.
SAFPair two_skyrmion_pair_ic(int nx, int ny, Real a, Real R, Real dw,
                             int polarity_top,
                             Real c1x, Real c1y, Real c2x, Real c2y);

// Pair separation r(t): split the top layer into left/right column
// halves at j = nx/2, take the PBC core centroid of each (the right
// centroid shifted back by the split column), and return their
// distance. Sets `alive_both` false and returns NaN if either half has
// no core (sum(1 - m_z) <= 0). Port of the measurement in
// pair_potential.py.
Real pair_separation(const Field3& m_top, Real a, bool& alive_both);

} // namespace stochastic
} // namespace skyrmion
