/// \file
/// Thermal-noise RNG: stateless counter-based per-site generator.
/// Mirrors src/stochastic_llgs/thermal_field.py::sample_thermal_field.
///
/// Each Gaussian sample is derived from a splitmix64 hash of the counter
/// (seed, draw_index, site_index) via Box-Muller, so the noise is
/// independent per site and DETERMINISTIC regardless of the OpenMP
/// thread count (the fill parallelizes trivially). One draw_index is
/// consumed per sample_thermal_field call (so successive steps and the
/// two SAF layers draw independent streams). The instantaneous field
/// has std sigma/sqrt(dt) (the discrete Wiener increment has variance
/// sigma^2*dt).
///
/// Note: the exact stream differs from the previous mt19937
/// implementation but is statistically identical; validation is
/// FDT/equipartition + T=0, so this is sufficient.
#pragma once

#include "skyrmion/types.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

/// Counter-based Gaussian generator for the thermal field.
///
/// Holds only a seed and a draw counter, so the stream depends on
/// neither call order across threads nor the OpenMP thread count.
class ThermalRng {
public:
    /// Construct a stream from its seed, with the draw counter at zero.
    /// \param seed Stream key; distinct seeds give independent noise.
    explicit ThermalRng(std::uint64_t seed) : key_(seed), draw_index_(0) {}

    /// Fill f with independent Gaussian samples of std sigma/sqrt(dt)
    /// and advance the draw counter by one.
    /// \param f Destination field, shape (ny, nx, 3).
    /// \param sigma Noise amplitude; must be finite and > 0.
    /// \param dt Time step; must be finite and > 0.
    /// \throws std::runtime_error if sigma or dt is non-finite or
    ///         non-positive.
    /// \note The caller handles the sigma == 0 (T = 0) zero-noise case
    ///       and must not call this, so that the deterministic stream
    ///       and the draw_index stay in sync.
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
        const std::uint64_t key = key_;
        const std::uint64_t di = draw_index_;
        const std::size_t n = f.data.size();
        // 2^-53, for mapping a 53-bit integer into the unit interval.
        const double inv_2p53 = 1.0 / 9007199254740992.0;
        const double two_pi = 6.283185307179586476925286766559;
#ifdef SKYRMION_OPENMP
        #pragma omp parallel for schedule(static)
#endif
        for (std::size_t k = 0; k < n; ++k) {
            // Two independent 64-bit hashes per site (sub = 0, 1).
            const std::uint64_t c0 =
                (static_cast<std::uint64_t>(k) << 1) | 0ull;
            const std::uint64_t c1 =
                (static_cast<std::uint64_t>(k) << 1) | 1ull;
            const std::uint64_t h1 = mix(key ^ mix(di ^ mix(c0)));
            const std::uint64_t h2 = mix(key ^ mix(di ^ mix(c1)));
            // u1 in (0,1] (offset by 1 to exclude 0 for the log), u2 in
            // [0,1).
            const double u1 =
                (static_cast<double>(h1 >> 11) + 1.0) * inv_2p53;
            const double u2 = static_cast<double>(h2 >> 11) * inv_2p53;
            const double g =
                std::sqrt(-2.0 * std::log(u1)) * std::cos(two_pi * u2);
            f.data[k] = static_cast<Real>(g) * std_dev;
        }
        ++draw_index_;
    }

private:
    // splitmix64 finalizer; strong avalanche mixing of a 64-bit counter.
    static std::uint64_t mix(std::uint64_t z) {
        z += 0x9E3779B97F4A7C15ull;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }

    std::uint64_t key_;
    std::uint64_t draw_index_;
};

} // namespace stochastic
} // namespace skyrmion
