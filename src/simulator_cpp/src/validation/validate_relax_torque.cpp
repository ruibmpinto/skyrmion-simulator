// Relaxation-torque diagnostic / validator. Tests the claim that the
// elevated relax tau_max floor on a larger box is a max-norm artifact of
// a well-relaxed skyrmion (residual torque delocalized in the
// ferromagnetic background, bulk well-relaxed, size/energy converged,
// max-of-N growing with lattice size) rather than a genuinely
// under-relaxed skyrmion (torque concentrated at the domain wall, size /
// energy still drifting).
//
// For each box size it chunk-relaxes the Set-A SAF skyrmion under newell
// demag (calling relax() in fixed-step chunks, never early-stopping) and
// after each chunk records: tangential-torque percentiles (max, p99,
// median, rms), the LCC ellipse D1/D2, the total energy, and the max-
// torque site's distance from the skyrmion core. The final per-site
// torque field and m_z are dumped for the spatial map/histogram. One NPZ
// per box; a summary table is printed.
//
// Checks (see scripts/validate_relax_torque.py for the plots):
//   1. torque map/histogram        2. size/energy convergence vs step
//   3. box-size scaling of the floor   4. 256x256 positive control
#include "skyrmion/fields.hpp"
#include "skyrmion/energy.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/sweep/relax.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;
using skyrmion::stochastic::skyrmion_ellipse_lcc;
using skyrmion::stochastic::skyrmion_center_lcc_pbc;
using skyrmion::stochastic::skyrmion_diameter_lcc;

namespace {

int fft_threads_from_env() {
    if (const char* e = std::getenv("OMP_NUM_THREADS")) {
        if (*e && std::atoi(e) > 0) return std::atoi(e);
    }
    return 1;
}

npy::tensor<double> arr1d(const std::vector<double>& v) {
    npy::tensor<double> t({v.size()});
    if (!v.empty()) t.copy_from(v.data(), v.size());
    return t;
}

npy::tensor<double> arr2d(const std::vector<double>& v, int ny, int nx) {
    npy::tensor<double> t({static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)});
    if (!v.empty()) t.copy_from(v.data(), v.size());
    return t;
}

npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}

// Per-site tangential torque |m x (m x H)| (top layer). Returns the
// (ny*nx) field; the cross-product matches sweep/relax.cpp.
std::vector<double> tangential_torque_field(const Field3& m,
                                            const Field3& H) {
    const int ny = m.ny, nx = m.nx;
    std::vector<double> f(static_cast<std::size_t>(ny) * nx, 0.0);
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real mx = m(i, j, 0), my = m(i, j, 1), mz = m(i, j, 2);
            const Real Hx = H(i, j, 0), Hy = H(i, j, 1), Hz = H(i, j, 2);
            const Real ax = my * Hz - mz * Hy;
            const Real ay = mz * Hx - mx * Hz;
            const Real az = mx * Hy - my * Hx;
            const Real tx = my * az - mz * ay;
            const Real ty = mz * ax - mx * az;
            const Real tz = mx * ay - my * ax;
            f[static_cast<std::size_t>(i) * nx + j] =
                std::sqrt(tx * tx + ty * ty + tz * tz);
        }
    }
    return f;
}

struct TorqueStats { double max; double p99; double median; double rms; };

TorqueStats torque_stats(const std::vector<double>& f) {
    TorqueStats s{0.0, 0.0, 0.0, 0.0};
    if (f.empty()) return s;
    double ss = 0.0;
    for (double v : f) ss += v * v;
    s.rms = std::sqrt(ss / f.size());
    std::vector<double> g = f;
    std::sort(g.begin(), g.end());
    s.max = g.back();
    s.median = g[g.size() / 2];
    s.p99 = g[static_cast<std::size_t>(0.99 * (g.size() - 1))];
    return s;
}

}  // namespace

int main() {
    // ----- Run configuration -------------------------------------------------
    struct Box { int nx; int ny; };
    const std::vector<Box> boxes = {
        {256, 256}, {350, 350}, {350, 500}, {500, 500}};
    const double dt = 5.0e-14;
    const double alpha_relax = 1.0;
    const int chunk = 2000;          // steps between samples
    // Total = 250000 steps: the 256² size keeps drifting (211->186 nm)
    // and tau keeps falling (7e-5->3e-5) well past 40k, so a short run
    // does not capture the converged plateau.
    const int n_chunks = 125;
    const DemagKind demag_kind = DemagKind::Newell;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    const std::string out_dir =
        "output/stochastic_llgs/validation/relax_torque";
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    const int fft_threads = fft_threads_from_env();

    std::printf("relax-torque validator (newell, Set-A), %d chunks x %d "
                "steps, threads=%d\n", n_chunks, chunk, fft_threads);
    std::printf("%-9s %8s %10s %10s %10s %10s %8s %8s %12s\n",
                "box", "steps", "tau_max", "tau_p99", "tau_med",
                "tau_rms", "D1(nm)", "D2(nm)", "E(J)");

    for (const Box& b : boxes) {
        Params p = make_default_params();
        p.nx = b.nx; p.ny = b.ny; p.dt = dt;
        p.demag_kind = demag_kind;
        p.demag_accuracy = demag_accuracy;
        p.demag_tol_conv = demag_tol_conv;
        p.pulse = std::make_shared<ConstantPulse>(0.0);
        precompute(p);
        DemagState demag(p, fft_threads);

        SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R,
                                  p.skyrmion_dw);
        Field3 m_top = std::move(ic.m_top);
        Field3 m_bot = std::move(ic.m_bot);
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);

        std::vector<double> s_step, s_tmax, s_tp99, s_tmed, s_trms;
        std::vector<double> s_d1, s_d2, s_E, s_rmax;
        int step = 0;
        std::vector<double> tfield;  // last torque field (top)
        for (int c = 0; c < n_chunks; ++c) {
            // Relax `chunk` more steps; tol=0 so it never early-stops.
            RelaxResult r = relax(std::move(m_top), std::move(m_bot), p,
                                  &demag, chunk, alpha_relax,
                                  /*tol_torque=*/0.0, /*tol_dE=*/0.0,
                                  /*check_every=*/chunk, /*print_every=*/0,
                                  /*mask=*/nullptr);
            m_top = std::move(r.m_top);
            m_bot = std::move(r.m_bot);
            step += chunk;
            // Diagnostics on the current state.
            effective_field_demag(m_top, m_bot, p, demag, H_top, H_bot,
                                  nullptr);
            tfield = tangential_torque_field(m_top, H_top);
            const TorqueStats ts = torque_stats(tfield);
            const Ellipse e = skyrmion_ellipse_lcc(m_top, p.a, +1);
            const double E = total_energy(m_top, m_bot, p, demag);
            // Distance of the max-torque site from the skyrmion core.
            std::size_t kmax = 0;
            for (std::size_t k = 1; k < tfield.size(); ++k)
                if (tfield[k] > tfield[kmax]) kmax = k;
            const Center2D ctr = skyrmion_center_lcc_pbc(m_top, p.a, +1);
            const int iy = static_cast<int>(kmax / p.nx);
            const int ix = static_cast<int>(kmax % p.nx);
            const double dxn = (ix * p.a - ctr.cx);
            const double dyn = (iy * p.a - ctr.cy);
            const double rmax = std::sqrt(dxn * dxn + dyn * dyn);

            s_step.push_back(step);
            s_tmax.push_back(ts.max); s_tp99.push_back(ts.p99);
            s_tmed.push_back(ts.median); s_trms.push_back(ts.rms);
            s_d1.push_back(e.D1); s_d2.push_back(e.D2);
            s_E.push_back(E); s_rmax.push_back(rmax);
        }
        // Per-box NPZ: convergence series + final torque field + m_z.
        std::vector<double> mz(static_cast<std::size_t>(p.ny) * p.nx);
        for (int i = 0; i < p.ny; ++i)
            for (int j = 0; j < p.nx; ++j)
                mz[static_cast<std::size_t>(i) * p.nx + j] = m_top(i, j, 2);
        char fn[256];
        std::snprintf(fn, sizeof(fn), "%s/box_%dx%d.npz",
                      out_dir.c_str(), p.nx, p.ny);
        npy::npzfilewriter w(fn);
        w.write("nx", sc(p.nx)); w.write("ny", sc(p.ny));
        w.write("a", sc(p.a)); w.write("dt", sc(dt));
        w.write("steps", arr1d(s_step));
        w.write("tau_max", arr1d(s_tmax));
        w.write("tau_p99", arr1d(s_tp99));
        w.write("tau_median", arr1d(s_tmed));
        w.write("tau_rms", arr1d(s_trms));
        w.write("D1", arr1d(s_d1)); w.write("D2", arr1d(s_d2));
        w.write("E", arr1d(s_E));
        w.write("rmax_from_core", arr1d(s_rmax));
        w.write("torque_field", arr2d(tfield, p.ny, p.nx));
        w.write("mz", arr2d(mz, p.ny, p.nx));

        // Self-describing relaxed-field snapshot per box (full m_top/
        // m_bot + real observables), same layout as m_eq.npz so it can
        // be plotted/loaded with the existing snapshot tools.
        const Ellipse ef = skyrmion_ellipse_lcc(m_top, p.a, +1);
        const Center2D cf = skyrmion_center_lcc_pbc(m_top, p.a, +1);
        const double diamf = skyrmion_diameter_lcc(m_top, p.a, +1);
        const double qtop = topological_charge(m_top, p.a);
        const double qbot = topological_charge(m_bot, p.a);
        const double psif = dw_angle(m_top, p.a, +1, /*mz_thresh=*/0.5);
        SnapshotBuffer snap(p.ny, p.nx, 1);
        snap.append(m_top, m_bot, step, step * dt, 0,
                    qtop, qbot, cf.cx, cf.cy, diamf,
                    ef.D1, ef.D2, ef.theta, psif);
        char sfn[256];
        std::snprintf(sfn, sizeof(sfn), "%s/box_%dx%d_snapshot.npz",
                      out_dir.c_str(), p.nx, p.ny);
        Field3 pos = lattice_positions(p.nx, p.ny, p.a);
        snap.write(sfn, pos, pos, "{}");

        std::printf("%4dx%-4d %8d %10.2e %10.2e %10.2e %10.2e "
                    "%8.1f %8.1f %12.4e\n",
                    p.nx, p.ny, step, s_tmax.back(), s_tp99.back(),
                    s_tmed.back(), s_trms.back(),
                    s_d1.back() * 1e9, s_d2.back() * 1e9, s_E.back());
        std::fflush(stdout);
    }
    std::printf("wrote per-box NPZ to %s\n", out_dir.c_str());
    return 0;
}
