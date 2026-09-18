/// \file
/// Time profiles for the SOT-driving current density J(t).
#pragma once

#include "skyrmion/types.hpp"

#include <memory>
#include <vector>

namespace skyrmion {

/// Abstract time profile of the driving current density J(t).
class Pulse {
public:
    virtual ~Pulse() = default;
    /// \param t Time in seconds.
    /// \return Current density at time `t`, in A/m^2.
    virtual Real operator()(Real t) const = 0;
};

/// Time-independent current density, J(t) = J0 for all t.
class ConstantPulse : public Pulse {
public:
    /// \param J0 Constant current density (A/m^2).
    explicit ConstantPulse(Real J0) : J0_(J0) {}
    /// \return J0, independent of the time argument.
    Real operator()(Real /*t*/) const override { return J0_; }
    /// \return The constant amplitude J0 (A/m^2).
    Real J0() const { return J0_; }
private:
    Real J0_;
};

/// Rectangular pulse: J0 on the closed interval [t_start, t_end],
/// zero outside it.
class SquarePulse : public Pulse {
public:
    /// \param J0 Plateau current density (A/m^2).
    /// \param t_start Rising edge, in seconds.
    /// \param t_end Falling edge, in seconds.
    /// \throws std::runtime_error If t_end <= t_start.
    SquarePulse(Real J0, Real t_start, Real t_end);
    /// \param t Time in seconds.
    /// \return J0 inside [t_start, t_end], else zero.
    Real operator()(Real t) const override;
private:
    Real J0_, t_start_, t_end_;
};

/// Gaussian pulse of peak amplitude J0 centred on t_center. The pulse
/// has unbounded support: it never reaches exactly zero.
class GaussianPulse : public Pulse {
public:
    /// \param J0 Peak current density (A/m^2).
    /// \param t_center Time of the peak, in seconds.
    /// \param fwhm Full width at half maximum, in seconds; the stored
    ///        sigma is fwhm / (2 sqrt(2 ln 2)).
    /// \throws std::runtime_error If fwhm is not positive.
    GaussianPulse(Real J0, Real t_center, Real fwhm);
    /// \param t Time in seconds.
    /// \return J0 * exp(-0.5 * ((t - t_center) / sigma)^2).
    Real operator()(Real t) const override;
private:
    Real J0_, t_center_, sigma_;
};

/// Linear rise to J0 at t_peak, then linear fall. Pin t_peak to
/// t_start for an instantaneous rise, to t_end for an instantaneous
/// fall, or midway for a symmetric ramp. Charge and action are both
/// independent of t_peak.
class TrianglePulse : public Pulse {
public:
    /// \param J0 Peak current density (A/m^2).
    /// \param t_start Start of the rising ramp, in seconds.
    /// \param t_peak Time at which J0 is reached, in seconds.
    /// \param t_end End of the falling ramp, in seconds.
    /// \throws std::runtime_error If t_end <= t_start, or if t_peak
    ///         lies outside [t_start, t_end].
    TrianglePulse(Real J0, Real t_start, Real t_peak, Real t_end);
    /// \param t Time in seconds.
    /// \return The ramped amplitude inside [t_start, t_end], else zero.
    ///         A peak pinned to t_end has no falling edge and holds J0.
    Real operator()(Real t) const override;
private:
    Real J0_, t_start_, t_peak_, t_end_;
};

/// Half-period sine lobe: vanishes at both edges, peaks at midpoint.
class HalfSinePulse : public Pulse {
public:
    /// \param J0 Peak current density (A/m^2).
    /// \param t_start Leading zero crossing, in seconds.
    /// \param t_end Trailing zero crossing, in seconds.
    /// \throws std::runtime_error If t_end <= t_start.
    HalfSinePulse(Real J0, Real t_start, Real t_end);
    /// \param t Time in seconds.
    /// \return J0 * sin(pi * (t - t_start) / (t_end - t_start)) inside
    ///         [t_start, t_end], else zero.
    Real operator()(Real t) const override;
private:
    Real J0_, t_start_, t_end_;
};

/// Sum of several pulses, all evaluated at the same time. Used to build
/// pulse trains out of the elementary shapes above.
class SuperpositionPulse : public Pulse {
public:
    /// \param pulses Component pulses; ownership is taken.
    /// \throws std::runtime_error If `pulses` is empty.
    explicit SuperpositionPulse(std::vector<std::unique_ptr<Pulse>> pulses);
    /// \param t Time in seconds.
    /// \return Sum of the component amplitudes at `t`, in A/m^2.
    Real operator()(Real t) const override;
private:
    std::vector<std::unique_ptr<Pulse>> pulses_;
};

} // namespace skyrmion
