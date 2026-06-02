// Thermal-noise parameters and Joule self-heating for the stochastic
// LLGS solver. Mirrors src/stochastic_llgs/parameters_thermal.py and
// src/stochastic_llgs/joule_heating.py.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

// Attach thermal-noise fields to p (mutated in place). Computes
//   V_cell      = a^2 * t_Co
//   sigma_noise = sqrt(2 * alpha * k_B * T / (gamma * Ms * V_cell))
// (Tesla*sqrt(s)). T = 0 is allowed and yields sigma_noise = 0 (the
// deterministic limit). Raises on invalid T, R_th, or non-positive
// V_cell.
void attach_thermal(Params& p, Real T, Real R_th, long long seed);

// Uniform global Joule heating: T(j) = T_sub + R_th * j^2.
Real T_of_j(Real j, Real T_sub, Real R_th);

} // namespace stochastic
} // namespace skyrmion
