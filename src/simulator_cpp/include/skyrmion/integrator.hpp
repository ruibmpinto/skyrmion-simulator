// LLGS time integration. Mirrors src/simulator/integrator.py.
//
// Two RHS variants:
//   RHSLocalKeff : local-K_eff effective field (no explicit demag).
//   RHSDemag     : bare-K effective field + FFT demag (slab kernel).
//
// Both implement the protocol:
//   void operator()(const Field3& m_top, const Field3& m_bot,
//                   Real t, Field3& dmdt_top, Field3& dmdt_bot);
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <cstdint>
#include <memory>

namespace skyrmion {

void normalize_inplace(Field3& m);

// dm/dt for a single layer from the explicit LLGS equation with SOT and
// optional topological spin Hall torque. H_eff is provided by the caller.
// `mask` (row-major ny*nx, 1 = inside, or nullptr = periodic): vacuum
// sites get dm/dt = 0 exactly -- the SOT/TSH torques do not depend on
// H_eff and would otherwise rotate vacuum spins. TSH + mask throws.
void llgs_rhs(const Field3& m, const Field3& H_eff, const Params& p, Real t,
              Field3& dmdt, const std::uint8_t* mask);

// `mask` (row-major ny*nx, 1 = inside the magnetic region, or nullptr for
// the periodic path) is forwarded to effective_field for free-boundary
// geometries. Stored as a borrowed pointer; the array must outlive the
// functor.
class RHSLocalKeff {
public:
    RHSLocalKeff(const Params& p, const std::uint8_t* mask);
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    const std::uint8_t* mask_;
    Field3 H_top_;
    Field3 H_bot_;
};

class RHSDemag {
public:
    RHSDemag(const Params& p, DemagState& demag, const std::uint8_t* mask);
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    DemagState& demag_;
    const std::uint8_t* mask_;
    Field3 H_top_;
    Field3 H_bot_;
};

// Single-layer local-K_eff RHS (matches Python relaxation._rhs_single).
// A lone ferromagnet: the layer is its own RKKY partner (harmless when
// the caller sets H_RKKY = 0). Implements the rk4_step_single protocol:
//   void operator()(const Field3& m, Real t, Field3& dmdt);
class RHSSingleKeff {
public:
    RHSSingleKeff(const Params& p, const std::uint8_t* mask);
    void operator()(const Field3& m, Real t, Field3& dmdt);
private:
    const Params& p_;
    const std::uint8_t* mask_;
    Field3 H_;
};

// One classical RK4 step, both layers in lockstep. Intermediate stages
// are renormalised to keep |m| = 1.
template <typename RHS>
void rk4_step(RHS& rhs, Field3& m_top, Field3& m_bot,
              Real t, Real dt, const Params& p);

// One classical RK4 step for a single layer (matches Python
// integrator.rk4_step_single). `rhs` is a callable
//   void operator()(const Field3& m, Real t, Field3& dmdt);
// Intermediate stages are renormalised to keep |m| = 1.
template <typename RHS>
void rk4_step_single(RHS& rhs, Field3& m, Real t, Real dt, const Params& p);

} // namespace skyrmion
