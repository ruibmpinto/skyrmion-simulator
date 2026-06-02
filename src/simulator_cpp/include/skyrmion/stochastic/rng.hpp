// Thermal-noise RNG: std::mt19937_64 + std::normal_distribution.
// Mirrors src/stochastic_llgs/thermal_field.py::sample_thermal_field.
//
// One engine per trajectory, seeded explicitly (seed = seed_base +
// traj_idx by convention). The instantaneous field has std
// sigma/sqrt(dt), so the integrated noise increment over a step has
// variance sigma^2 * dt (the discrete Wiener representation).
//
// Note: std::normal_distribution is not bit-portable across standard
// library implementations, so the exact noise stream is reproducible
// within a build (same seed -> same trajectory) but not guaranteed
// identical across platforms. Validation is statistical + T=0, so this
// is sufficient.
#pragma once

#include "skyrmion/types.hpp"

#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

class ThermalRng {
public:
    explicit ThermalRng(std::uint64_t seed) : engine_(seed) {}

    // Fill f (shape (ny, nx, 3)) with independent Gaussian samples of
    // std sigma/sqrt(dt). Raises on non-positive sigma or dt, matching
    // the Python sampler; the caller decides the sigma == 0 (T = 0)
    // zero-noise case (it must not draw, to keep the deterministic
    // stream).
    void sample_thermal_field(Field3& f, Real sigma, Real dt) {
        if (!std::isfinite(sigma) || sigma <= 0.0) {
            throw std::runtime_error(
                "sample_thermal_field: sigma must be finite and "
                "strictly positive.");
        }
        if (!std::isfinite(dt) || dt <= 0.0) {
            throw std::runtime_error(
                "sample_thermal_field: dt must be finite and "
                "strictly positive.");
        }
        const Real std_dev = sigma / std::sqrt(dt);
        for (Real& v : f.data) {
            v = dist_(engine_) * std_dev;
        }
    }

private:
    std::mt19937_64 engine_;
    std::normal_distribution<Real> dist_{0.0, 1.0};
};

} // namespace stochastic
} // namespace skyrmion
