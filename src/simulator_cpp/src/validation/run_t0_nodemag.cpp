// T = 0 reproducibility gate, no demag. Port of
// src/skyrmion_simulator/stochastic_llgs/validation/test_t0_limit_nodemag.py.
//
// Drives a SAF skyrmion twice: deterministic RK4 (rhs_local_keff) and
// zero-noise Heun (sigma = 0, so the predictor-corrector degenerates to
// a deterministic order-2 Heun). Heun is order 2 and RK4 order 4, so a
// small discretization mismatch is expected. Pass: relative deviation
// < tol_rel on diameter, drift velocity, and topological charge.
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"

#include <npy/npy.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

constexpr double kRad2Deg = 57.29577951308232;

struct Track {
    double diameter, cx, cy, Q, velocity, hall_deg;
};

npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}

// Relax (J = 0) then drive (J = p.J_current) the SAF skyrmion, applying
// `step` once per step. The SOT is gated by p.pulse, so relax uses a
// zero pulse and drive a constant-J pulse. Centres are recorded at
// 0.75*n_drive and n_drive to estimate the drift velocity.
Track run_phase(Params& p, int n_relax, int n_drive, double dt,
                const std::function<void(Field3&, Field3&, double)>& step) {
    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    Field3 m_top = std::move(ic.m_top);
    Field3 m_bot = std::move(ic.m_bot);

    p.pulse = std::make_shared<ConstantPulse>(0.0);
    double t = 0.0;
    for (int s = 0; s < n_relax; ++s) { step(m_top, m_bot, t); t += dt; }

    p.pulse = std::make_shared<ConstantPulse>(p.J_current);
    const int s0 = static_cast<int>(0.75 * n_drive);
    const int s1 = n_drive;
    double x0 = 0, y0 = 0, t0 = 0, x1 = 0, y1 = 0, t1 = 0;
    t = 0.0;
    for (int s = 1; s <= n_drive; ++s) {
        step(m_top, m_bot, t);
        t += dt;
        if (s == s0 || s == s1) {
            Center2D c = skyrmion_center(m_top, p.a, +1);
            if (s == s0) { x0 = c.cx; y0 = c.cy; t0 = s * dt; }
            else         { x1 = c.cx; y1 = c.cy; t1 = s * dt; }
        }
    }
    const double vx = (x1 - x0) / (t1 - t0);
    const double vy = (y1 - y0) / (t1 - t0);
    Track tr;
    tr.diameter = skyrmion_diameter(m_top, p.a, +1);
    tr.cx = x1; tr.cy = y1;
    tr.Q = topological_charge(m_top, p.a);
    tr.velocity = std::sqrt(vx * vx + vy * vy);
    tr.hall_deg = kRad2Deg * std::atan2(std::abs(vy), std::abs(vx));
    return tr;
}

} // namespace

int main() {
    // =========================== Run configuration ==========================
    const int n_relax = 6000;        // 300 ps
    const int n_drive = 10000;       // 500 ps
    const double dt = 5.0e-14;
    const long long seed = 1;
    const double tol_norm = 5.0e-3;
    const double tol_rel = 0.02;     // 2% per observable
    const std::string out_dir = "output/stochastic_llgs/validation";
    const std::string out_npz = "t0_limit_nodemag.npz";
    // ======================= End run configuration ==========================

    std::printf("T = 0 reproducibility (no demag)\n");

    // ---- Reference: deterministic RK4 -----------------------------------
    Params p_det = make_default_params();
    p_det.dt = dt;
    RHSLocalKeff rhs(p_det, /*mask=*/nullptr);
    std::printf("Running deterministic RK4 reference "
                "(%d relax + %d drive)...\n", n_relax, n_drive);
    Track ref = run_phase(p_det, n_relax, n_drive, dt,
        [&](Field3& a, Field3& b, double tt) {
            rk4_step(rhs, a, b, tt, dt, p_det);
        });
    std::printf("  RK4: d=%.1f nm v=%.1f m/s theta=%.2f deg Q=%.4f\n",
                ref.diameter * 1e9, ref.velocity, ref.hall_deg, ref.Q);

    // ---- Simulation: zero-noise Heun (T = 0) ----------------------------
    Params p_sim = make_default_params();
    p_sim.dt = dt;
    stochastic::attach_thermal(p_sim, 0.0, 0.0, seed);
    if (p_sim.sigma_noise != 0.0) {
        throw std::runtime_error("t0_nodemag: sigma_noise must be 0 at T=0.");
    }
    stochastic::ThermalRng rng(static_cast<std::uint64_t>(seed));
    stochastic::HeunStochasticStepper stepper(p_sim, nullptr, rng, 0.0,
                                              tol_norm, /*mask=*/nullptr);
    std::printf("Running stochastic Heun at T = 0...\n");
    Track sim = run_phase(p_sim, n_relax, n_drive, dt,
        [&](Field3& a, Field3& b, double tt) {
            stepper.step(a, b, tt, dt, p_sim);
        });
    std::printf("  Heun: d=%.1f nm v=%.1f m/s theta=%.2f deg Q=%.4f\n",
                sim.diameter * 1e9, sim.velocity, sim.hall_deg, sim.Q);

    // ---- Compare diameter, velocity, Q within tol_rel -------------------
    auto rel = [](double ref_v, double sim_v) {
        return (ref_v == 0.0) ? (sim_v != 0.0 ? 1e30 : 0.0)
                              : std::abs(sim_v - ref_v) / std::abs(ref_v);
    };
    const double rel_d = rel(ref.diameter, sim.diameter);
    const double rel_v = rel(ref.velocity, sim.velocity);
    const double rel_Q = rel(ref.Q, sim.Q);
    const double dtheta = std::abs(sim.hall_deg - ref.hall_deg);

    std::filesystem::create_directories(out_dir);
    const std::string path = out_dir + "/" + out_npz;
    npy::npzfilewriter w(path);
    w.write("ref_diameter", sc(ref.diameter));
    w.write("ref_velocity", sc(ref.velocity));
    w.write("ref_hall_deg", sc(ref.hall_deg));
    w.write("ref_Q", sc(ref.Q));
    w.write("sim_diameter", sc(sim.diameter));
    w.write("sim_velocity", sc(sim.velocity));
    w.write("sim_hall_deg", sc(sim.hall_deg));
    w.write("sim_Q", sc(sim.Q));
    w.write("rel_diameter", sc(rel_d));
    w.write("rel_velocity", sc(rel_v));
    w.write("rel_Q", sc(rel_Q));
    w.write("hall_abs_diff", sc(dtheta));
    w.write("tol_rel", sc(tol_rel));
    w.close();

    std::printf("rel_diameter=%.4f rel_velocity=%.4f rel_Q=%.4f "
                "hall_abs_diff=%.4f (tol_rel=%.2f)\n",
                rel_d, rel_v, rel_Q, dtheta, tol_rel);
    std::printf("Saved %s\n", path.c_str());
    if (rel_d >= tol_rel || rel_v >= tol_rel || rel_Q >= tol_rel) {
        std::printf("T=0 no-demag gate FAILED.\n");
        throw std::runtime_error("T=0 no-demag gate FAILED.");
    }
    std::printf("T=0 no-demag gate PASSED.\n");
    return 0;
}
