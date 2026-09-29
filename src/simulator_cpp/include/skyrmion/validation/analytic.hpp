/// \file
/// Closed-form targets for the stochastic-LLGS validation gates. These
/// mirror the analytic formulas in src/skyrmion_simulator/stochastic_llgs/validation/*.py
/// (Langevin magnetization, Neel-Brown reversal time, discrete magnon
/// stiffness). They are deterministic and bit-checkable against Python.
#pragma once

#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace validation {

/// Langevin function L(x) = coth(x) - 1/x. Uses the small-argument
/// series L(x) ~ x/3 for |x| < 1e-4 to avoid catastrophic cancellation.
/// \param x Reduced field mu B / (k_B T); dimensionless.
/// \return L(x), the equilibrium <m_z> of a Zeeman macrospin.
Real langevin_function(Real x);

/// Neel-Brown high-barrier mean reversal (first-passage) time:
///   tau = (1 + alpha^2) / (alpha * gamma) * sqrt(pi / Delta) *
///   exp(Delta)
/// with reduced barrier Delta = K V_cell / (k_B T). gamma is the bare
/// gyromagnetic ratio (rad s^-1 T^-1).
/// \param alpha Gilbert damping; dimensionless.
/// \param gamma Bare gyromagnetic ratio, in rad s^-1 T^-1.
/// \param Delta Reduced energy barrier K V_cell / (k_B T);
///        dimensionless.
/// \return The mean reversal time, in seconds.
Real brown_tau(Real alpha, Real gamma, Real Delta);

/// Discrete magnon stiffness H_k (Tesla) on the np.fft.fft2 frequency
/// grid, returned row-major over (ny, nx):
///   H_k = B_z + C_ex * (4 - 2 cos(2 pi mx / nx) - 2 cos(2 pi my / ny)),
/// where mx, my are the signed FFT frequency indices. The lattice
/// constant cancels (k_x a = 2 pi mx / nx), so it is not an argument.
/// \param ny Lattice rows.
/// \param nx Lattice columns.
/// \param B_z Uniform field along z, in tesla (the k = 0 stiffness).
/// \param C_ex Exchange field prefactor 2 A_ex / (Ms a^2), in tesla.
/// \return The ny*nx stiffness values, row-major, in tesla.
std::vector<Real> magnon_stiffness_grid(int ny, int nx, Real B_z,
                                        Real C_ex);

} // namespace validation
} // namespace skyrmion
