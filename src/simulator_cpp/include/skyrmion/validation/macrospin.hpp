// Macrospin ensemble driver for the stochastic-LLGS validation gates.
// Port of src/stochastic_llgs/validation/macrospin.py.
//
// Builds a (ny = n_traj, nx = 1) lattice with all spatial couplings
// zeroed (C_ex = C_dmi = 0, H_RKKY = 0), so each y-row is an
// independent macrospin and one Heun step advances the whole ensemble.
// Used by the Langevin and Brown-reversal gates.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace validation {

// (n_traj, 1) macrospin parameters: H_RKKY = 0, K_bot = 0, J = 0,
// C_ex = C_dmi = 0, bare anisotropy C_anis_top = 2 K / Ms (no thin-film
// -mu0 Ms correction), gamma_p = gamma / (1 + alpha^2). attach_thermal
// is applied so p.sigma_noise is populated. Raises on invalid input.
Params make_macrospin_params(Real T, Real alpha, Vec3 H_ext, Real K,
                             Real a, Real t_Co, Real Ms, Real gamma,
                             long long seed, int n_traj);

// Top-layer magnetization history over the sampled steps. `m_top` is
// row-major over (n_samples, n_traj, 3).
struct MacrospinHistory {
    std::vector<double> times;   // (n_samples,)
    std::vector<double> m_top;   // (n_samples * n_traj * 3)
    int n_samples = 0;
    int n_traj = 0;
};

// Advance n_traj independent macrospins for n_steps Heun steps,
// recording m_top once every sample_every steps. m0_top is broadcast
// across all trajectories; m_bot starts at +z. Requires C_ex = C_dmi =
// 0, H_RKKY = 0, nx = 1, and |m0_top| = 1.
MacrospinHistory run_macrospin_ensemble(Params& p, Vec3 m0_top, Real dt,
                                        int n_steps, int sample_every,
                                        Real tol_norm);

} // namespace validation
} // namespace skyrmion
