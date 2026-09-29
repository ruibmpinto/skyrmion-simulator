// T = 0 reproducibility gate, with slab demag. Port of
// src/skyrmion_simulator/stochastic_llgs/validation/test_t0_limit_demag.py.
//
// Drives a SAF skyrmion (stabilised by H_z = 0.20 T, R = 40 nm) twice:
// deterministic RK4-with-demag (RHSDemag) and zero-noise Heun-with-demag.
// Gate: relative deviation < tol on diameter and topological charge.
// The per-cell field error is recorded as a diagnostic only -- in the
// demag-driven regime two integrators converge to the same morphology
// but place solitons at slightly different positions (O(dt^2) growth).
#include "skyrmion/demag.hpp"
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

npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}

npy::tensor<float> field_f32(const Field3& m) {
    npy::tensor<float> t({static_cast<std::size_t>(m.ny),
                          static_cast<std::size_t>(m.nx), 3});
    std::vector<float> buf(m.data.size());
    for (std::size_t k = 0; k < m.data.size(); ++k) {
        buf[k] = static_cast<float>(m.data[k]);
    }
    t.copy_from(buf.data(), buf.size());
    return t;
}

// Relax (J = 0) then drive (J = p.J_current); SOT gated by p.pulse.
SAFPair run_phase(Params& p, int n_relax, int n_drive, double dt,
                  const std::function<void(Field3&, Field3&, double)>& step) {
    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    Field3 m_top = std::move(ic.m_top);
    Field3 m_bot = std::move(ic.m_bot);
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    double t = 0.0;
    for (int s = 0; s < n_relax; ++s) { step(m_top, m_bot, t); t += dt; }
    p.pulse = std::make_shared<ConstantPulse>(p.J_current);
    t = 0.0;
    for (int s = 0; s < n_drive; ++s) { step(m_top, m_bot, t); t += dt; }
    return {std::move(m_top), std::move(m_bot)};
}

} // namespace

int main() {
    // =========================== Run configuration ==========================
    const double h_z_stabilising = 0.20;   // Tesla
    const double sk_R = 40.0e-9;
    const double sk_dw = 15.0e-9;
    const int n_relax = 2000;              // 100 ps
    const int n_drive = 2000;              // 100 ps
    const double dt = 5.0e-14;
    const long long seed = 1;
    const double tol_norm = 5.0e-3;
    const double tol_rel_d = 0.02;
    const double tol_rel_Q = 0.02;
    const std::string out_dir = "output/stochastic_llgs/validation";
    const std::string out_npz = "t0_limit_demag.npz";
    // ======================= End run configuration ==========================

    std::printf("T = 0 reproducibility (with demag)\n");
    std::printf("H_z=%.2f T R_sk=%.1f nm dw=%.1f nm n_relax=%d n_drive=%d\n",
                h_z_stabilising, sk_R * 1e9, sk_dw * 1e9, n_relax, n_drive);

    Params p = make_default_params();
    p.dt = dt;
    p.H_ext = {0.0, 0.0, h_z_stabilising};
    p.skyrmion_R = sk_R;
    p.skyrmion_dw = sk_dw;
    p.demag_kind = DemagKind::Slab;
    precompute(p);
    DemagState demag(p, 0);

    // ---- Reference: deterministic RK4 with demag ------------------------
    RHSDemag rhs(p, demag, /*mask=*/nullptr);
    std::printf("Running deterministic RK4-with-demag reference...\n");
    SAFPair ref = run_phase(p, n_relax, n_drive, dt,
        [&](Field3& a, Field3& b, double tt) {
            rk4_step(rhs, a, b, tt, dt, p);
        });
    const double d_ref = skyrmion_diameter(ref.m_top, p.a, +1);
    const double Q_ref = topological_charge(ref.m_top, p.a);
    std::printf("  RK4: d=%.1f nm Q=%+.4f\n", d_ref * 1e9, Q_ref);

    // ---- Simulation: zero-noise Heun with demag (T = 0) -----------------
    stochastic::attach_thermal(p, 0.0, 0.0, seed);
    if (p.sigma_noise != 0.0) {
        throw std::runtime_error("t0_demag: sigma_noise must be 0 at T=0.");
    }
    stochastic::ThermalRng rng(static_cast<std::uint64_t>(seed));
    stochastic::HeunStochasticStepper stepper(p, &demag, rng, 0.0, tol_norm,
                                              /*mask=*/nullptr);
    std::printf("Running stochastic Heun-with-demag at T = 0...\n");
    SAFPair sim = run_phase(p, n_relax, n_drive, dt,
        [&](Field3& a, Field3& b, double tt) {
            stepper.step(a, b, tt, dt, p);
        });
    const double d_sim = skyrmion_diameter(sim.m_top, p.a, +1);
    const double Q_sim = topological_charge(sim.m_top, p.a);
    std::printf("  Heun: d=%.1f nm Q=%+.4f\n", d_sim * 1e9, Q_sim);

    // ---- Per-cell field error (diagnostic) ------------------------------
    const int ny = p.ny, nx = p.nx;
    double err_max = 0.0, sum_top = 0.0, sum_bot = 0.0;
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            double dt2 = 0.0, db2 = 0.0;
            for (int k = 0; k < 3; ++k) {
                const double et = sim.m_top(i, j, k) - ref.m_top(i, j, k);
                const double eb = sim.m_bot(i, j, k) - ref.m_bot(i, j, k);
                dt2 += et * et; db2 += eb * eb;
            }
            sum_top += dt2; sum_bot += db2;
            err_max = std::max(err_max, std::sqrt(std::max(dt2, db2)));
        }
    const double ncell = static_cast<double>(ny) * nx;
    const double err_rms = std::sqrt(sum_top / ncell / 2.0
                                     + sum_bot / ncell / 2.0);

    // ---- Morphology gates -----------------------------------------------
    auto rel = [](double r, double s) {
        return (r == 0.0) ? (s != 0.0 ? 1e30 : 0.0)
                          : std::abs(s - r) / std::abs(r);
    };
    const double rel_d = rel(d_ref, d_sim);
    const double rel_Q = rel(Q_ref, Q_sim);

    std::filesystem::create_directories(out_dir);
    const std::string path = out_dir + "/" + out_npz;
    npy::npzfilewriter w(path);
    w.write("h_z_stabilising", sc(h_z_stabilising));
    w.write("sk_R", sc(sk_R)); w.write("sk_dw", sc(sk_dw));
    w.write("n_relax", sc(n_relax)); w.write("n_drive", sc(n_drive));
    w.write("d_ref", sc(d_ref)); w.write("d_sim", sc(d_sim));
    w.write("Q_ref", sc(Q_ref)); w.write("Q_sim", sc(Q_sim));
    w.write("rel_d", sc(rel_d)); w.write("rel_Q", sc(rel_Q));
    w.write("err_max", sc(err_max)); w.write("err_rms", sc(err_rms));
    w.write("tol_rel_d", sc(tol_rel_d)); w.write("tol_rel_Q", sc(tol_rel_Q));
    w.write("m_top_ref", field_f32(ref.m_top));
    w.write("m_top_sim", field_f32(sim.m_top));
    w.write("m_bot_ref", field_f32(ref.m_bot));
    w.write("m_bot_sim", field_f32(sim.m_bot));
    w.close();

    std::printf("rel_d=%.4f (tol %.2f)  rel_Q=%.4f (tol %.2f)\n",
                rel_d, tol_rel_d, rel_Q, tol_rel_Q);
    std::printf("[diagnostic] max|m_sim-m_ref|=%.3e RMS=%.3e\n",
                err_max, err_rms);
    std::printf("Saved %s\n", path.c_str());
    if (rel_d >= tol_rel_d || rel_Q >= tol_rel_Q) {
        std::printf("T=0 demag gate FAILED.\n");
        throw std::runtime_error("T=0 demag gate FAILED.");
    }
    std::printf("T=0 demag gate PASSED.\n");
    return 0;
}
