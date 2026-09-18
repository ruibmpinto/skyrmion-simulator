/// \file
/// Macrospin ensemble driver for the stochastic-LLGS validation gates.
/// Port of src/stochastic_llgs/validation/macrospin.py.
///
/// Builds a (ny = n_traj, nx = 1) lattice with all spatial couplings
/// zeroed (C_ex = C_dmi = 0, H_RKKY = 0), so each y-row is an
/// independent macrospin and one Heun step advances the whole ensemble.
/// Used by the Langevin and Brown-reversal gates.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace validation {

/// (n_traj, 1) macrospin parameters: H_RKKY = 0, K_bot = 0, J = 0,
/// C_ex = C_dmi = 0, bare anisotropy C_anis_top = 2 K / Ms (no
/// thin-film -mu0 Ms correction), gamma_p = gamma / (1 + alpha^2).
/// attach_thermal is applied so p.sigma_noise is populated. Raises on
/// invalid input.
/// \param T Bath temperature, in kelvin.
/// \param alpha Gilbert damping; must be finite and > 0.
/// \param H_ext Uniform external field, in tesla.
/// \param K Uniaxial anisotropy of the top layer, in J/m^3; must be
///        finite and >= 0.
/// \param a Lattice constant, in metres.
/// \param t_Co Magnetic layer thickness, in metres.
/// \param Ms Saturation magnetisation, in A/m.
/// \param gamma Bare gyromagnetic ratio, in rad s^-1 T^-1.
/// \param seed Master noise seed passed to attach_thermal.
/// \param n_traj Number of independent trajectories (lattice rows).
/// \return Params describing the decoupled macrospin ensemble.
/// \throws std::runtime_error if alpha, K, a, t_Co, Ms or gamma is
///         non-finite or out of range, or if n_traj <= 0.
Params make_macrospin_params(Real T, Real alpha, Vec3 H_ext, Real K,
                             Real a, Real t_Co, Real Ms, Real gamma,
                             long long seed, int n_traj);

/// Top-layer magnetization history over the sampled steps. `m_top` is
/// row-major over (n_samples, n_traj, 3).
struct MacrospinHistory {
    std::vector<double> times;   ///< Sample times (s), (n_samples,)
    std::vector<double> m_top;   ///< (n_samples * n_traj * 3)
    int n_samples = 0;           ///< Number of recorded samples
    int n_traj = 0;              ///< Number of trajectories
};

/// Advance n_traj independent macrospins for n_steps Heun steps,
/// recording m_top once every sample_every steps. m0_top is broadcast
/// across all trajectories; m_bot starts at +z. Requires C_ex = C_dmi =
/// 0, H_RKKY = 0, nx = 1, and |m0_top| = 1.
///
/// The stepper is stochastic::HeunStochasticStepper with no demag and
/// amplitude p.sigma_noise, seeded from p.seed.
/// \param p Macrospin parameters, typically from
///        make_macrospin_params.
/// \param m0_top Initial top-layer direction; must be a unit vector.
/// \param dt Step size, in seconds.
/// \param n_steps Number of Heun steps; must be > 0.
/// \param sample_every Steps between recorded samples; must be > 0.
/// \param tol_norm Norm-drift tolerance of the Heun step.
/// \return The sampled times and top-layer magnetization history.
/// \throws std::runtime_error if p.C_ex, p.C_dmi or p.H_RKKY is
///         non-zero, p.nx != 1, n_steps or sample_every is
///         non-positive, or |m0_top| differs from 1 by more than 1e-10.
MacrospinHistory run_macrospin_ensemble(Params& p, Vec3 m0_top, Real dt,
                                        int n_steps, int sample_every,
                                        Real tol_norm);

} // namespace validation
} // namespace skyrmion
