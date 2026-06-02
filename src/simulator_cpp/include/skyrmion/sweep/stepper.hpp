// Integrator-step abstraction for the sweep driver. Mirrors the
// step_drive / step_relax callables in src/sweeps/integrators.py so
// deterministic, demag-aware, and (later) stochastic backends flow
// through the same driver.
//
// A future HeunStochasticStepper plugs in here with no driver changes.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

class Stepper {
public:
    virtual ~Stepper() = default;
    // Advance both layers by one step of size dt at time t.
    virtual void step(Field3& m_top, Field3& m_bot,
                      Real t, Real dt, Params& p) = 0;
};

// Plain RK4 with local-K_eff anisotropy (no explicit FFT demag).
// `mask` (row-major ny*nx, or nullptr for the periodic path) is forwarded
// to the field assembly for free-boundary geometries.
class RK4LocalKeffStepper : public Stepper {
public:
    RK4LocalKeffStepper(const Params& p, const std::uint8_t* mask)
        : rhs_(p, mask) {}
    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override {
        rk4_step(rhs_, m_top, m_bot, t, dt, p);
    }
private:
    RHSLocalKeff rhs_;
};

// RK4 with explicit FFT demag (slab or Newell kernel).
class RK4DemagStepper : public Stepper {
public:
    RK4DemagStepper(const Params& p, DemagState& demag,
                    const std::uint8_t* mask)
        : rhs_(p, demag, mask) {}
    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override {
        rk4_step(rhs_, m_top, m_bot, t, dt, p);
    }
private:
    RHSDemag rhs_;
};

} // namespace sweep
} // namespace skyrmion
