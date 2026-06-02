#include "skyrmion/pulses.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {

SquarePulse::SquarePulse(Real J0, Real t_start, Real t_end)
    : J0_(J0), t_start_(t_start), t_end_(t_end) {
    if (t_end <= t_start) {
        throw std::runtime_error("SquarePulse: t_end must exceed t_start.");
    }
}

Real SquarePulse::operator()(Real t) const {
    return (t >= t_start_ && t <= t_end_) ? J0_ : Real{0};
}

static constexpr Real kFwhmToSigma() {
    // 1 / (2 * sqrt(2 * ln 2))
    return 1.0 / (2.0 * 1.1774100225154746910115693264596996377473856893858);
}

GaussianPulse::GaussianPulse(Real J0, Real t_center, Real fwhm)
    : J0_(J0), t_center_(t_center) {
    if (fwhm <= 0.0) {
        throw std::runtime_error("GaussianPulse: FWHM must be positive.");
    }
    sigma_ = fwhm * kFwhmToSigma();
}

Real GaussianPulse::operator()(Real t) const {
    Real z = (t - t_center_) / sigma_;
    return J0_ * std::exp(-0.5 * z * z);
}

SuperpositionPulse::SuperpositionPulse(std::vector<std::unique_ptr<Pulse>> pulses)
    : pulses_(std::move(pulses)) {
    if (pulses_.empty()) {
        throw std::runtime_error("SuperpositionPulse requires at least one pulse.");
    }
}

Real SuperpositionPulse::operator()(Real t) const {
    Real total = 0.0;
    for (const auto& p : pulses_) total += (*p)(t);
    return total;
}

} // namespace skyrmion
