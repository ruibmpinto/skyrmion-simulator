#include "skyrmion/stochastic/heun.hpp"

#include "skyrmion/fields.hpp"
#include "skyrmion/integrator.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

HeunStochasticStepper::HeunStochasticStepper(const Params& p,
                                             DemagState* demag,
                                             ThermalRng& rng,
                                             Real sigma, Real tol_norm,
                                             const std::uint8_t* mask)
    : demag_(demag), rng_(rng), sigma_(sigma), tol_norm_(tol_norm),
      mask_(mask),
      H_top_(p.ny, p.nx), H_bot_(p.ny, p.nx),
      h_top_(p.ny, p.nx), h_bot_(p.ny, p.nx),
      f1_top_(p.ny, p.nx), f1_bot_(p.ny, p.nx),
      f2_top_(p.ny, p.nx), f2_bot_(p.ny, p.nx),
      mp_top_(p.ny, p.nx), mp_bot_(p.ny, p.nx) {
    if (!std::isfinite(tol_norm) || tol_norm <= 0.0) {
        throw std::runtime_error(
            "HeunStochasticStepper: tol_norm must be finite and > 0.");
    }
    if (!std::isfinite(sigma) || sigma < 0.0) {
        throw std::runtime_error(
            "HeunStochasticStepper: sigma must be finite and >= 0.");
    }
}

void HeunStochasticStepper::field_plus_noise(const Field3& m_top,
                                             const Field3& m_bot,
                                             Params& p, bool single) {
    if (single) {
        // Lone ferromagnet: the layer is its own RKKY partner (harmless
        // when H_RKKY = 0). No demag (a bilayer coupling).
        effective_field(m_top, m_top, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top_, mask_);
    } else if (demag_) {
        effective_field_demag(m_top, m_bot, p, *demag_, H_top_, H_bot_, mask_);
    } else {
        effective_field(m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top_, mask_);
        effective_field(m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY, H_bot_, mask_);
    }
    // Add the pre-sampled thermal field to the assembled H (never
    // inside field assembly, so the exchange/DMI difference operators
    // act on m only).
    const std::size_t n = H_top_.data.size();
    for (std::size_t k = 0; k < n; ++k) H_top_.data[k] += h_top_.data[k];
    if (!single) {
        for (std::size_t k = 0; k < n; ++k) H_bot_.data[k] += h_bot_.data[k];
    }
}

void HeunStochasticStepper::step(Field3& m_top, Field3& m_bot,
                                 Real t, Real dt, Params& p) {
    // Single-layer mode: an empty bottom layer evolves a lone ferromagnet.
    // Demag is a bilayer coupling, so a kernel is rejected.
    const bool single = (m_bot.n_sites() == 0);
    if (single && demag_) {
        throw std::runtime_error(
            "HeunStochasticStepper: single-layer mode (empty m_bot) "
            "requires demag == nullptr (demag is a bilayer coupling).");
    }

    // Sample the thermal field once per step; reuse it in predictor
    // and corrector (Stratonovich). sigma == 0 => zero noise, no draw.
    if (sigma_ > 0.0) {
        rng_.sample_thermal_field(h_top_, sigma_, dt);
        if (!single) rng_.sample_thermal_field(h_bot_, sigma_, dt);
    } else {
        std::fill(h_top_.data.begin(), h_top_.data.end(), Real{0});
        if (!single) std::fill(h_bot_.data.begin(), h_bot_.data.end(), Real{0});
    }
    // Zero the noise outside the magnetic region so frozen cells do not
    // random-walk (free-boundary geometries).
    if (mask_) {
        const int ny = m_top.ny, nx = m_top.nx;
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                if (mask_[static_cast<std::size_t>(i) * nx + j]) continue;
                h_top_(i, j, 0) = 0; h_top_(i, j, 1) = 0; h_top_(i, j, 2) = 0;
                if (!single) {
                    h_bot_(i, j, 0) = 0; h_bot_(i, j, 1) = 0;
                    h_bot_(i, j, 2) = 0;
                }
            }
        }
    }

    // Predictor: f1 = LLGS(m, H(m) + h); m~ = m + dt*f1 (no renorm).
    field_plus_noise(m_top, m_bot, p, single);
    llgs_rhs(m_top, H_top_, p, t, f1_top_);
    if (!single) llgs_rhs(m_bot, H_bot_, p, t, f1_bot_);
    const std::size_t n = m_top.data.size();
    for (std::size_t k = 0; k < n; ++k) {
        mp_top_.data[k] = m_top.data[k] + dt * f1_top_.data[k];
    }
    if (!single) {
        for (std::size_t k = 0; k < n; ++k) {
            mp_bot_.data[k] = m_bot.data[k] + dt * f1_bot_.data[k];
        }
    }

    // Corrector: f2 = LLGS(m~, H(m~) + h) with the SAME noise h.
    field_plus_noise(mp_top_, mp_bot_, p, single);
    llgs_rhs(mp_top_, H_top_, p, t + dt, f2_top_);
    if (!single) llgs_rhs(mp_bot_, H_bot_, p, t + dt, f2_bot_);

    // Heun average, then end-of-step norm-drift check + renorm.
    const int ny = m_top.ny, nx = m_top.nx;
    Real drift = 0.0;
    auto combine = [&](Field3& m, const Field3& f1, const Field3& f2) {
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                const Real x = m(i, j, 0) + 0.5 * dt * (f1(i, j, 0) + f2(i, j, 0));
                const Real y = m(i, j, 1) + 0.5 * dt * (f1(i, j, 1) + f2(i, j, 1));
                const Real z = m(i, j, 2) + 0.5 * dt * (f1(i, j, 2) + f2(i, j, 2));
                const Real nm = std::sqrt(x * x + y * y + z * z);
                const Real d = std::abs(nm - 1.0);
                if (d > drift) drift = d;
                m(i, j, 0) = x; m(i, j, 1) = y; m(i, j, 2) = z;
            }
        }
    };
    // Write the un-normalised Heun update in place, tracking drift.
    combine(m_top, f1_top_, f2_top_);
    if (!single) combine(m_bot, f1_bot_, f2_bot_);
    if (drift > tol_norm_) {
        throw std::runtime_error(
            "HeunStochasticStepper: end-of-step norm drift exceeds "
            "tol_norm; reduce dt or check the thermal amplitude.");
    }
    last_drift_ = drift;
    // Single end-of-step projection back onto the unit sphere.
    normalize_inplace(m_top);
    if (!single) normalize_inplace(m_bot);
}

} // namespace stochastic
} // namespace skyrmion
