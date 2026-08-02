// Thermal equilibration to an LCC-size plateau. Shared by the drive
// runner (run_trajectory) and the standalone stage-2 equilibration
// binary so the loop logic lives in one place. Mirrors the Python
// run_single._equilibrate.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

struct EquilResult {
    int    n_used = 0;       // stochastic steps taken
    bool   converged = true; // plateau criterion met
    double d1_relaxed = 0.0; // major LCC axis, final-window mean (m)
    double d2_relaxed = 0.0; // minor LCC axis, final-window mean (m)
    // Per-check series (one entry every check_every steps; NaN where
    // the LCC ellipse was undefined).
    std::vector<int>    step_series;
    std::vector<double> d1_series;  // m
    std::vector<double> d2_series;  // m
};

// Run J=0 noisy dynamics (caller must have zeroed the pulse/SOT) in
// blocks of `check_every`, sampling the top-layer LCC diameter, until
// the mean over the last `window` checks matches the previous `window`
// within relative `tol` for `k_consec` consecutive checks, or until
// `max_steps`. Mutates `m_top`/`m_bot` in place. `progress_every` > 0
// prints a timestamped progress line every that-many size-checks
// (step / sim-ps / wall-s / diameter); 0 is silent.
EquilResult equilibrate_to_plateau(
    Field3& m_top, Field3& m_bot, HeunStochasticStepper& stepper,
    Params& p, int check_every, int window, Real tol,
    int k_consec, int max_steps, int progress_every);

}  // namespace stochastic
}  // namespace skyrmion
