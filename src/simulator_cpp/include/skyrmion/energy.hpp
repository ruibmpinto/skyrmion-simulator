// Total magnetic energy of the SAF stack.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {

// E = -0.5 V_cell Ms sum_layers sum_sites m.H_internal - V_cell Ms sum m.H_zeeman.
// Uses bare-K anisotropy + explicit FFT demag (matches Python total_energy).
Real total_energy(const Field3& m_top, const Field3& m_bot,
                  const Params& p, DemagState& demag);

} // namespace skyrmion
