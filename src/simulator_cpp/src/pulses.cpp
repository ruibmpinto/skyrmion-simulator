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

TrianglePulse::TrianglePulse(Real J0, Real t_start, Real t_peak, Real t_end)
    : J0_(J0), t_start_(t_start), t_peak_(t_peak), t_end_(t_end) {
    if (t_end <= t_start) {
        throw std::runtime_error("TrianglePulse: t_end must exceed t_start.");
    }
    if (!(t_start <= t_peak && t_peak <= t_end)) {
        throw std::runtime_error(
            "TrianglePulse: t_peak must lie within [t_start, t_end].");
    }
}

Real TrianglePulse::operator()(Real t) const {
    // Branch structure mirrors the Python implementation so the two
    // agree to the last bit on the shared reference grid.
    if (t < t_start_ || t > t_end_) return Real{0};
    if (t < t_peak_) {
        return J0_ * ((t - t_start_) / (t_peak_ - t_start_));
    }
    // A peak pinned to t_end has no falling edge; hold the amplitude.
    if (t_end_ == t_peak_) return J0_;
    return J0_ * ((t_end_ - t) / (t_end_ - t_peak_));
}

static constexpr Real kPi() {
    return 3.14159265358979323846264338327950288419716939937510;
}

HalfSinePulse::HalfSinePulse(Real J0, Real t_start, Real t_end)
    : J0_(J0), t_start_(t_start), t_end_(t_end) {
    if (t_end <= t_start) {
        throw std::runtime_error("HalfSinePulse: t_end must exceed t_start.");
    }
}

Real HalfSinePulse::operator()(Real t) const {
    if (t < t_start_ || t > t_end_) return Real{0};
    const Real phase = kPi() * (t - t_start_) / (t_end_ - t_start_);
    return J0_ * std::sin(phase);
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
