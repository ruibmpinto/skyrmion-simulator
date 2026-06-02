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

#include <memory>

namespace skyrmion {

void normalize_inplace(Field3& m);

// dm/dt for a single layer from the explicit LLGS equation with SOT and
// optional topological spin Hall torque. H_eff is provided by the caller.
void llgs_rhs(const Field3& m, const Field3& H_eff, const Params& p, Real t,
              Field3& dmdt);

class RHSLocalKeff {
public:
    explicit RHSLocalKeff(const Params& p);
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    Field3 H_top_;
    Field3 H_bot_;
};

class RHSDemag {
public:
    RHSDemag(const Params& p, DemagState& demag);
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    DemagState& demag_;
    Field3 H_top_;
    Field3 H_bot_;
};

// One classical RK4 step, both layers in lockstep. Intermediate stages
// are renormalised to keep |m| = 1.
template <typename RHS>
void rk4_step(RHS& rhs, Field3& m_top, Field3& m_bot,
              Real t, Real dt, const Params& p);

} // namespace skyrmion
