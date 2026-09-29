// Brown reversal-time macrospin validation gate. Port of
// src/skyrmion_simulator/stochastic_llgs/validation/test_brown_reversal.py.
//
// Uniaxial macrospin (no field), each trajectory starts at +z and runs
// until m_z falls below mz_flip_threshold or n_steps elapses. The
// right-censored mean first-passage time is compared to Brown's
// high-barrier formula tau = (1+a^2)/(a gamma) sqrt(pi/Delta) exp(Delta).
// Pass: |log10(tau_sim / tau_brown)| < tol_log10 at each barrier.
#include "skyrmion/parameters.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/types.hpp"
#include "skyrmion/validation/analytic.hpp"
#include "skyrmion/validation/macrospin.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}
npy::tensor<double> arr1d(const std::vector<double>& v) {
    npy::tensor<double> t({v.size()});
    t.copy_from(v.data(), v.size());
    return t;
}

// Run an ensemble until each trajectory's m_z first drops below
// `thresh`, or n_steps elapses (censored at n_steps). Returns the
// number flipped and the summed flip-step count (censored entries
// contribute n_steps).
long long run_brown_ensemble(Params& p, Vec3 m0, double dt, long long n_steps,
                             double tol_norm, double thresh,
                             int& n_flipped_out) {
    const int n_traj = p.ny;
    Field3 m_top(n_traj, 1), m_bot(n_traj, 1);
    for (int i = 0; i < n_traj; ++i) {
        m_top(i, 0, 0) = m0[0]; m_top(i, 0, 1) = m0[1];
        m_top(i, 0, 2) = m0[2]; m_bot(i, 0, 2) = 1.0;
    }
    stochastic::ThermalRng rng(static_cast<std::uint64_t>(p.seed));
    stochastic::HeunStochasticStepper stepper(p, nullptr, rng,
                                              p.sigma_noise, tol_norm,
                                              /*mask=*/nullptr);
    std::vector<long long> flip_step(n_traj, n_steps);
    std::vector<char> flipped(n_traj, 0);
    int n_flipped = 0;
    double t = 0.0;
    for (long long step = 0; step < n_steps; ++step) {
        stepper.step(m_top, m_bot, t, dt, p);
        t += dt;
        for (int i = 0; i < n_traj; ++i) {
            if (!flipped[i] && m_top(i, 0, 2) < thresh) {
                flip_step[i] = step;
                flipped[i] = 1;
                ++n_flipped;
            }
        }
        if (n_flipped == n_traj) break;
    }
    long long sum_steps = 0;
    for (int i = 0; i < n_traj; ++i) sum_steps += flip_step[i];
    n_flipped_out = n_flipped;
    return sum_steps;
}

} // namespace

int main() {
    // =========================== Run configuration ==========================
    const std::vector<double> delta_list = {3.0, 5.0};
    const double alpha = 0.5;
    const double K_top = 1.294e6;
    const double a = 2.0e-9;
    const double t_co = 1.3e-9;
    const double ms = 1.43e6;
    const double gamma = 194.8e9;
    const double dt = 5.0e-14;
    const int n_traj = 256;
    const long long seed_base = 31;
    const double tol_norm = 5.0e-3;
    const double mz_flip_thresh = -0.5;
    const int n_steps_per_tau = 30;
    const double tol_log10 = 0.5;
    const std::string out_dir = "output/stochastic_llgs/validation";
    const std::string out_npz = "brown_reversal.npz";
    // ======================= End run configuration ==========================

    const double k_B = 1.380649e-23;
    const double V_cell = a * a * t_co;
    const int n_D = static_cast<int>(delta_list.size());
    std::vector<double> tau_sim(n_D), tau_th(n_D), frac_flip(n_D);

    std::printf("Brown reversal macrospin gate\n");
    for (int i = 0; i < n_D; ++i) {
        const double delta = delta_list[i];
        const double T = K_top * V_cell / (delta * k_B);
        const double tau_b = validation::brown_tau(alpha, gamma, delta);
        const long long n_steps =
            static_cast<long long>(n_steps_per_tau * tau_b / dt);
        const long long seed = seed_base + 1000LL * i;
        Params p = validation::make_macrospin_params(
            T, alpha, Vec3{0.0, 0.0, 0.0}, K_top, a, t_co, ms, gamma,
            seed, n_traj);
        int n_flipped = 0;
        const long long sum_steps = run_brown_ensemble(
            p, Vec3{0.0, 0.0, 1.0}, dt, n_steps, tol_norm, mz_flip_thresh,
            n_flipped);
        if (n_flipped < n_traj / 2) {
            std::printf("  WARN: Delta=%.1f: only %d/%d trajectories "
                        "flipped within %d*tau_brown.\n",
                        delta, n_flipped, n_traj, n_steps_per_tau);
        }
        const double tau_hat = (sum_steps * dt)
                               / static_cast<double>(std::max(n_flipped, 1));
        tau_sim[i] = tau_hat;
        tau_th[i] = tau_b;
        frac_flip[i] = static_cast<double>(n_flipped) / n_traj;
        std::printf("  Delta=%.1f T=%7.2f K tau_Brown=%.2e s "
                    "tau_sim=%.2e s log10(ratio)=%+.2f flipped=%d/%d\n",
                    delta, T, tau_b, tau_hat,
                    std::log10(tau_hat / tau_b), n_flipped, n_traj);
    }

    std::vector<double> log10_ratio(n_D);
    double max_log10 = 0.0;
    for (int i = 0; i < n_D; ++i) {
        log10_ratio[i] = std::log10(tau_sim[i] / tau_th[i]);
        max_log10 = std::max(max_log10, std::abs(log10_ratio[i]));
    }

    std::filesystem::create_directories(out_dir);
    const std::string path = out_dir + "/" + out_npz;
    npy::npzfilewriter w(path);
    w.write("delta", arr1d(delta_list));
    w.write("tau_sim", arr1d(tau_sim));
    w.write("tau_brown", arr1d(tau_th));
    w.write("log10_ratio", arr1d(log10_ratio));
    w.write("max_log10", sc(max_log10));
    w.write("tol_log10", sc(tol_log10));
    w.write("fraction_flipped", arr1d(frac_flip));
    w.write("n_traj", sc(n_traj)); w.write("dt", sc(dt));
    w.close();

    std::printf("max |log10(tau_sim/tau_Brown)| = %.2f; tol = %.2f\n",
                max_log10, tol_log10);
    std::printf("Saved %s\n", path.c_str());
    if (max_log10 >= tol_log10) {
        std::printf("Brown reversal gate FAILED.\n");
        throw std::runtime_error("Brown reversal gate FAILED.");
    }
    std::printf("Brown reversal gate PASSED.\n");
    return 0;
}
