/// \file
/// Integrator-step abstraction for the sweep driver. Mirrors the
/// step_drive / step_relax callables in src/sweeps/integrators.py so
/// deterministic, demag-aware, and (later) stochastic backends flow
/// through the same driver.
///
/// A future HeunStochasticStepper plugs in here with no driver changes.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

/// Interface every sweep time-integration backend implements.
class Stepper {
public:
    virtual ~Stepper() = default;
    /// Advance both layers by one step of size dt at time t.
    /// \param m_top Top-layer magnetization, updated in place.
    /// \param m_bot Bottom-layer magnetization, updated in place.
    /// \param t Current time, in seconds (the pulse is evaluated at it).
    /// \param dt Step size, in seconds.
    /// \param p Run configuration.
    virtual void step(Field3& m_top, Field3& m_bot,
                      Real t, Real dt, Params& p) = 0;
};

/// Plain RK4 with local-K_eff anisotropy (no explicit FFT demag).
/// `mask` (row-major ny*nx, or nullptr for the periodic path) is
/// forwarded to the field assembly for free-boundary geometries.
class RK4LocalKeffStepper : public Stepper {
public:
    /// \param p Run configuration, borrowed by the RHS functor.
    /// \param mask Row-major ny*nx geometry mask, or nullptr for the
    ///        periodic path; the array must outlive this stepper.
    RK4LocalKeffStepper(const Params& p, const std::uint8_t* mask)
        : rhs_(p, mask) {}
    /// One RK4 step of both layers on the local-K_eff field.
    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override {
        rk4_step(rhs_, m_top, m_bot, t, dt, p);
    }
private:
    RHSLocalKeff rhs_;
};

/// RK4 with explicit FFT demag (slab or Newell kernel).
class RK4DemagStepper : public Stepper {
public:
    /// \param p Run configuration, borrowed by the RHS functor.
    /// \param demag Demag state holding the precomputed kernel and FFT
    ///        plans; borrowed, so it must outlive this stepper.
    /// \param mask Row-major ny*nx geometry mask, or nullptr for the
    ///        periodic path; the array must outlive this stepper.
    RK4DemagStepper(const Params& p, DemagState& demag,
                    const std::uint8_t* mask)
        : rhs_(p, demag, mask) {}
    /// One RK4 step of both layers on the bare-K + FFT-demag field.
    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override {
        rk4_step(rhs_, m_top, m_bot, t, dt, p);
    }
private:
    RHSDemag rhs_;
};

} // namespace sweep
} // namespace skyrmion
