/// \file
/// Thermal-noise parameters and Joule self-heating for the stochastic
/// LLGS solver. Mirrors src/stochastic_llgs/parameters_thermal.py and
/// src/stochastic_llgs/joule_heating.py.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

/// Attach thermal-noise fields to p (mutated in place). Computes
///   V_cell      = a^2 * t_Co
///   sigma_noise = sqrt(2 * alpha * k_B * T / (gamma * Ms * V_cell))
/// (Tesla*sqrt(s)). T = 0 is allowed and yields sigma_noise = 0 (the
/// deterministic limit). Raises on invalid T, R_th, or non-positive
/// V_cell.
/// \param p Parameter set; T, R_th, seed, k_B, V_cell and sigma_noise
///        are overwritten. a, t_Co, alpha, gamma_ and Ms must already
///        be set.
/// \param T Temperature (K); finite and >= 0.
/// \param R_th Joule self-heating coefficient (K m^4 / A^2); finite
///        and >= 0.
/// \param seed Thermal-RNG stream key, stored in p.seed.
/// \throws std::runtime_error if T or R_th is non-finite or negative,
///         if V_cell = a^2 * t_Co is not positive, or if the computed
///         sigma_noise^2 is negative.
void attach_thermal(Params& p, Real T, Real R_th, long long seed);

/// Uniform global Joule heating: T(j) = T_sub + R_th * j^2.
/// \param j Current density (A/m^2); must be finite.
/// \param T_sub Substrate temperature (K); finite and > 0.
/// \param R_th Heating coefficient (K m^4 / A^2); finite and >= 0.
/// \return Effective temperature (K).
/// \throws std::runtime_error if j is non-finite, T_sub is non-finite
///         or non-positive, or R_th is non-finite or negative.
Real T_of_j(Real j, Real T_sub, Real R_th);

} // namespace stochastic
} // namespace skyrmion
