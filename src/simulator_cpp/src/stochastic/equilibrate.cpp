#include "skyrmion/stochastic/equilibrate.hpp"

#include "skyrmion/observables.hpp"        // Ellipse
#include "skyrmion/stochastic/lcc.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <vector>

namespace skyrmion {
namespace stochastic {

namespace { constexpr double kNaN = std::numeric_limits<double>::quiet_NaN(); }

EquilResult equilibrate_to_plateau(
    Field3& m_top, Field3& m_bot, HeunStochasticStepper& stepper,
    Params& p, int check_every, int window, Real tol,
    int k_consec, int max_steps, int progress_every) {
    const Real a = p.a;
    Real t = 0.0;
    std::vector<double> diam_hist, d1_hist, d2_hist;
    int stable = 0;
    int check = 0;
    const auto t_start = std::chrono::steady_clock::now();
    EquilResult r;
    r.converged = true;
    while (r.n_used < max_steps) {
        for (int s = 0; s < check_every; ++s) {
            stepper.step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
        }
        r.n_used += check_every;
        ++check;
        const double d_now = skyrmion_diameter_lcc(m_top, a, +1);
        diam_hist.push_back(d_now);
        if (progress_every > 0 && check % progress_every == 0) {
            const double wall =
                std::chrono::duration<double>(
                    std::chrono::steady_clock::now() - t_start).count();
            std::printf(
                "    equil step %d (%.0f ps), wall %.0fs, d=%.1f nm\n",
                r.n_used, r.n_used * p.dt * 1e12, wall, d_now * 1e9);
            std::fflush(stdout);
        }
        try {
            const Ellipse e = skyrmion_ellipse_lcc(m_top, a, +1);
            d1_hist.push_back(e.D1);
            d2_hist.push_back(e.D2);
        } catch (const std::exception&) {
            d1_hist.push_back(kNaN);
            d2_hist.push_back(kNaN);
        }
        const int n = static_cast<int>(diam_hist.size());
        if (n >= 2 * window) {
            double nm = 0.0, om = 0.0;
            for (int q = 0; q < window; ++q) {
                nm += diam_hist[n - 1 - q];
                om += diam_hist[n - 1 - window - q];
            }
            nm /= window;
            om /= window;
            if (om > 0.0 && std::fabs(nm - om) < tol * om) {
                if (++stable >= k_consec) {
                    r.converged = true;
                    break;
                }
            } else {
                stable = 0;
                r.converged = false;
            }
        }
    }
    // Relaxed size: mean over the final window of checks.
    const int n = static_cast<int>(d1_hist.size());
    const int tail = (window < n) ? window : n;
    double s1 = 0.0, s2 = 0.0;
    int c1 = 0, c2 = 0;
    for (int q = 0; q < tail; ++q) {
        const double v1 = d1_hist[n - 1 - q];
        const double v2 = d2_hist[n - 1 - q];
        if (std::isfinite(v1)) { s1 += v1; ++c1; }
        if (std::isfinite(v2)) { s2 += v2; ++c2; }
    }
    r.d1_relaxed = (c1 > 0) ? s1 / c1 : kNaN;
    r.d2_relaxed = (c2 > 0) ? s2 / c2 : kNaN;
    return r;
}

}  // namespace stochastic
}  // namespace skyrmion
