// Micro-benchmark: time RK4 LLGS steps on a 64x64 lattice for the two
// field models (local-K_eff and slab FFT demag). Reports microseconds
// per step so the C++ and Python integrators can be compared on the
// same workload. Integration only; no I/O, no observables.
#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/stepper.hpp"

#include <chrono>
#include <cstdio>
#include <memory>
#include <utility>

using namespace skyrmion;
using namespace skyrmion::sweep;

namespace {

double time_steps(Stepper& stepper, Field3 m_top, Field3 m_bot,
                  Params& p, int n_warmup, int n_steps) {
    Real t = 0.0;
    for (int s = 0; s < n_warmup; ++s) {
        stepper.step(m_top, m_bot, t, p.dt, p);
        t += p.dt;
    }
    const auto t0 = std::chrono::steady_clock::now();
    for (int s = 0; s < n_steps; ++s) {
        stepper.step(m_top, m_bot, t, p.dt, p);
        t += p.dt;
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

} // namespace

int main() {
    const int nx = 64, ny = 64;
    const int n_warmup = 50;
    const int n_steps = 2000;

    Params p = make_default_params();
    p.nx = nx; p.ny = ny; p.dt = 5.0e-14;
    p.J_current = 4.0e11;
    p.pulse = std::make_shared<ConstantPulse>(p.J_current);
    precompute(p);

    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);

    std::printf("C++ profile: %dx%d lattice, %d steps (after %d warmup)\n",
                nx, ny, n_steps, n_warmup);

    // ---- local-K_eff path ---------------------------------------------------
    {
        RK4LocalKeffStepper stepper(p, /*mask=*/nullptr);
        const double sec = time_steps(stepper, ic.m_top, ic.m_bot,
                                      p, n_warmup, n_steps);
        std::printf("  keff   : %8.2f us/step  (%6.1f steps/s, %.3f s total)\n",
                    sec / n_steps * 1e6, n_steps / sec, sec);
    }
    // ---- slab FFT demag path ------------------------------------------------
    {
        Params pd = p;
        pd.demag_kind = DemagKind::Slab;
        pd.pulse = std::make_shared<ConstantPulse>(pd.J_current);
        precompute(pd);
        DemagState demag(pd, /*threads=*/0);
        RK4DemagStepper stepper(pd, demag, /*mask=*/nullptr);
        const double sec = time_steps(stepper, ic.m_top, ic.m_bot,
                                      pd, n_warmup, n_steps);
        std::printf("  slab   : %8.2f us/step  (%6.1f steps/s, %.3f s total)\n",
                    sec / n_steps * 1e6, n_steps / sec, sec);
    }
    return 0;
}
