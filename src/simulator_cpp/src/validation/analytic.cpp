#include "skyrmion/validation/analytic.hpp"

#include <cmath>

namespace skyrmion {
namespace validation {

namespace { constexpr Real kPi = 3.14159265358979323846; }

Real langevin_function(Real x) {
    if (std::abs(x) < 1.0e-4) {
        return x / 3.0;
    }
    return 1.0 / std::tanh(x) - 1.0 / x;
}

Real brown_tau(Real alpha, Real gamma, Real Delta) {
    return (1.0 + alpha * alpha) / (alpha * gamma)
           * std::sqrt(kPi / Delta) * std::exp(Delta);
}

std::vector<Real> magnon_stiffness_grid(int ny, int nx, Real B_z,
                                        Real C_ex) {
    std::vector<Real> H_k(static_cast<std::size_t>(ny) * nx);
    const Real two_pi = 2.0 * kPi;
    for (int i = 0; i < ny; ++i) {
        const int my = (i < (ny + 1) / 2) ? i : i - ny;
        const Real cy = std::cos(two_pi * my / ny);
        for (int j = 0; j < nx; ++j) {
            const int mx = (j < (nx + 1) / 2) ? j : j - nx;
            const Real cx = std::cos(two_pi * mx / nx);
            H_k[static_cast<std::size_t>(i) * nx + j] =
                B_z + C_ex * (4.0 - 2.0 * cx - 2.0 * cy);
        }
    }
    return H_k;
}

} // namespace validation
} // namespace skyrmion
