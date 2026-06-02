// Stratonovich-Heun stepper for the stochastic SAF LLGS. Port of
// src/stochastic_llgs/integrator_sllg.py::heun_stochastic_step.
//
// A predictor-corrector RK2 (NOT rk4_step): assemble H from m only,
// add the pre-sampled thermal field, evaluate llgs_rhs; the SAME noise
// sample is reused in predictor and corrector (this is what makes the
// scheme converge to the Stratonovich SDE). The predictor is not
// renormalised; one end-of-step renorm restores |m| = 1 after a loud
// norm-drift check.
//
// Implements sweep::Stepper, so it also drops into sweep::run_trace /
// sweep::relax unchanged. demag == nullptr selects the local-K_eff
// field; non-null selects bare-K + FFT demag. sigma == 0 draws no
// noise (the deterministic T = 0 limit, bit-parity with Python).
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/sweep/stepper.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

class HeunStochasticStepper : public sweep::Stepper {
public:
    HeunStochasticStepper(const Params& p, DemagState* demag,
                          ThermalRng& rng, Real sigma, Real tol_norm);

    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override;

    Real last_norm_drift() const { return last_drift_; }

private:
    DemagState* demag_;
    ThermalRng& rng_;
    Real sigma_;
    Real tol_norm_;
    Real last_drift_ = 0.0;
    // Scratch (sized at construction).
    Field3 H_top_, H_bot_;
    Field3 h_top_, h_bot_;
    Field3 f1_top_, f1_bot_, f2_top_, f2_bot_;
    Field3 mp_top_, mp_bot_;

    // Assemble H_top_/H_bot_ from (m_top, m_bot) using the configured
    // field model, then add the current noise sample h_top_/h_bot_.
    void field_plus_noise(const Field3& m_top, const Field3& m_bot, Params& p);
};

} // namespace stochastic
} // namespace skyrmion
