// Time profiles for the SOT-driving current density J(t).
#pragma once

#include "skyrmion/types.hpp"

#include <memory>
#include <vector>

namespace skyrmion {

class Pulse {
public:
    virtual ~Pulse() = default;
    virtual Real operator()(Real t) const = 0;
};

class ConstantPulse : public Pulse {
public:
    explicit ConstantPulse(Real J0) : J0_(J0) {}
    Real operator()(Real /*t*/) const override { return J0_; }
    Real J0() const { return J0_; }
private:
    Real J0_;
};

class SquarePulse : public Pulse {
public:
    SquarePulse(Real J0, Real t_start, Real t_end);
    Real operator()(Real t) const override;
private:
    Real J0_, t_start_, t_end_;
};

class GaussianPulse : public Pulse {
public:
    GaussianPulse(Real J0, Real t_center, Real fwhm);
    Real operator()(Real t) const override;
private:
    Real J0_, t_center_, sigma_;
};

class SuperpositionPulse : public Pulse {
public:
    explicit SuperpositionPulse(std::vector<std::unique_ptr<Pulse>> pulses);
    Real operator()(Real t) const override;
private:
    std::vector<std::unique_ptr<Pulse>> pulses_;
};

} // namespace skyrmion
