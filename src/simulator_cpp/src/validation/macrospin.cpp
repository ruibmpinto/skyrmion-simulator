#include "skyrmion/validation/macrospin.hpp"

#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace skyrmion {
namespace validation {

Params make_macrospin_params(Real T, Real alpha, Vec3 H_ext, Real K,
                             Real a, Real t_Co, Real Ms, Real gamma,
                             long long seed, int n_traj) {
    if (!(std::isfinite(alpha) && alpha > 0.0)) {
        throw std::runtime_error(
            "make_macrospin_params: alpha must be finite and > 0.");
    }
    if (!(std::isfinite(K) && K >= 0.0)) {
        throw std::runtime_error(
            "make_macrospin_params: K must be finite and >= 0.");
    }
    if (!(std::isfinite(a) && a > 0.0 && std::isfinite(t_Co) && t_Co > 0.0
          && std::isfinite(Ms) && Ms > 0.0 && std::isfinite(gamma)
          && gamma > 0.0)) {
        throw std::runtime_error(
            "make_macrospin_params: a, t_Co, Ms, gamma must be finite "
            "and > 0.");
    }
    if (n_traj <= 0) {
        throw std::runtime_error(
            "make_macrospin_params: n_traj must be a positive int.");
    }
    Params p = make_default_params();
    p.nx = 1;
    p.ny = n_traj;
    p.alpha = alpha;
    p.gamma_ = gamma;
    p.Ms = Ms;
    p.a = a;
    p.t_Co = t_Co;
    p.H_ext = H_ext;
    p.K_top = K;
    p.K_bot = 0.0;
    p.H_RKKY = 0.0;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    precompute(p);
    // Zero spatial couplings so the y-axis stacks independent
    // trajectories; bare uniaxial anisotropy (no thin-film correction).
    p.C_ex = 0.0;
    p.C_dmi = 0.0;
    p.C_anis_top = 2.0 * p.K_top / p.Ms;
    p.C_anis_bot = 0.0;
    p.gamma_p = p.gamma_ / (1.0 + p.alpha * p.alpha);
    stochastic::attach_thermal(p, T, 0.0, seed);
    return p;
}

MacrospinHistory run_macrospin_ensemble(Params& p, Vec3 m0_top, Real dt,
                                        int n_steps, int sample_every,
                                        Real tol_norm) {
    if (p.C_ex != 0.0 || p.C_dmi != 0.0) {
        throw std::runtime_error(
            "run_macrospin_ensemble: requires C_ex = C_dmi = 0.");
    }
    if (p.H_RKKY != 0.0) {
        throw std::runtime_error(
            "run_macrospin_ensemble: requires H_RKKY = 0.");
    }
    if (p.nx != 1) {
        throw std::runtime_error(
            "run_macrospin_ensemble: requires p.nx = 1.");
    }
    if (n_steps <= 0 || sample_every <= 0) {
        throw std::runtime_error(
            "run_macrospin_ensemble: n_steps and sample_every must be "
            "positive.");
    }
    const Real n0 = std::sqrt(m0_top[0] * m0_top[0] + m0_top[1] * m0_top[1]
                              + m0_top[2] * m0_top[2]);
    if (std::abs(n0 - 1.0) > 1.0e-10) {
        throw std::runtime_error(
            "run_macrospin_ensemble: m0_top must have unit norm.");
    }
    const int n_traj = p.ny;
    Field3 m_top(n_traj, 1), m_bot(n_traj, 1);
    for (int i = 0; i < n_traj; ++i) {
        m_top(i, 0, 0) = m0_top[0];
        m_top(i, 0, 1) = m0_top[1];
        m_top(i, 0, 2) = m0_top[2];
        m_bot(i, 0, 2) = 1.0;
    }
    stochastic::ThermalRng rng(static_cast<std::uint64_t>(p.seed));
    const Real sigma = p.sigma_noise;
    stochastic::HeunStochasticStepper stepper(p, nullptr, rng, sigma,
                                              tol_norm, /*mask=*/nullptr);

    const int n_samples = (n_steps + sample_every - 1) / sample_every;
    MacrospinHistory h;
    h.n_traj = n_traj;
    h.times.reserve(n_samples);
    h.m_top.reserve(static_cast<std::size_t>(n_samples) * n_traj * 3);
    Real t = 0.0;
    for (int step = 0; step < n_steps; ++step) {
        stepper.step(m_top, m_bot, t, dt, p);
        t += dt;
        if (step % sample_every == 0) {
            h.times.push_back(step * dt);
            for (int i = 0; i < n_traj; ++i)
                for (int k = 0; k < 3; ++k)
                    h.m_top.push_back(m_top(i, 0, k));
        }
    }
    h.n_samples = static_cast<int>(h.times.size());
    return h;
}

} // namespace validation
} // namespace skyrmion
