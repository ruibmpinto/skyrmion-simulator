#include "skyrmion/stochastic/thermal.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

void attach_thermal(Params& p, Real T, Real R_th, long long seed) {
    if (!std::isfinite(T) || T < 0.0) {
        throw std::runtime_error(
            "attach_thermal: T must be finite and non-negative "
            "(T = 0 gives the deterministic limit).");
    }
    if (!std::isfinite(R_th) || R_th < 0.0) {
        throw std::runtime_error(
            "attach_thermal: R_th must be finite and non-negative.");
    }
    p.T = T;
    p.R_th = R_th;
    p.seed = seed;
    p.k_B = 1.380649e-23;
    p.V_cell = p.a * p.a * p.t_Co;
    if (p.V_cell <= 0.0) {
        throw std::runtime_error(
            "attach_thermal: V_cell = a^2 * t_Co must be positive.");
    }
    const Real sigma2 = 2.0 * p.alpha * p.k_B * p.T
                        / (p.gamma_ * p.Ms * p.V_cell);
    if (sigma2 < 0.0) {
        throw std::runtime_error(
            "attach_thermal: computed sigma_noise^2 is negative; "
            "check alpha, gamma, Ms, a, t_Co.");
    }
    p.sigma_noise = std::sqrt(sigma2);
}

Real T_of_j(Real j, Real T_sub, Real R_th) {
    if (!std::isfinite(j)) {
        throw std::runtime_error("T_of_j: j must be finite.");
    }
    if (!std::isfinite(T_sub) || T_sub <= 0.0) {
        throw std::runtime_error(
            "T_of_j: T_sub must be finite and strictly positive.");
    }
    if (!std::isfinite(R_th) || R_th < 0.0) {
        throw std::runtime_error(
            "T_of_j: R_th must be finite and non-negative.");
    }
    return T_sub + R_th * j * j;
}

} // namespace stochastic
} // namespace skyrmion
