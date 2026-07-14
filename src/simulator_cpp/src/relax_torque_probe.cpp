#include "skyrmion/relax_torque_probe.hpp"

#include "skyrmion/energy.hpp"
#include "skyrmion/fields.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/sweep/relax.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <utility>
#include <vector>

namespace skyrmion {

using skyrmion::sweep::relax;
using skyrmion::sweep::RelaxResult;
using skyrmion::stochastic::skyrmion_ellipse_lcc;
using skyrmion::stochastic::skyrmion_center_lcc_pbc;
using skyrmion::stochastic::skyrmion_diameter_lcc;

namespace {

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

// Per-site tangential torque |m x (m x H)| (top layer).
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

// Keep only the cells inside the magnetic region (mask byte != 0).
std::vector<double> masked_values(const std::vector<double>& f,
                                  const std::uint8_t* mask) {
    std::vector<double> g;
    g.reserve(f.size());
    for (std::size_t k = 0; k < f.size(); ++k)
        if (mask[k]) g.push_back(f[k]);
    return g;
}

struct TorqueStats { double max; double p99; double median; double rms; };

// numpy 'linear' quantile: rank = q * (n - 1), interpolated.
double quantile_linear(const std::vector<double>& sorted, double q) {
    const double rank = q * (sorted.size() - 1);
    const std::size_t lo = static_cast<std::size_t>(rank);
    if (lo + 1 >= sorted.size()) return sorted.back();
    const double frac = rank - lo;
    return sorted[lo] * (1.0 - frac) + sorted[lo + 1] * frac;
}

TorqueStats torque_stats(const std::vector<double>& f) {
    TorqueStats s{0.0, 0.0, 0.0, 0.0};
    if (f.empty()) return s;
    double ss = 0.0;
    for (double v : f) ss += v * v;
    s.rms = std::sqrt(ss / f.size());
    std::vector<double> g = f;
    std::sort(g.begin(), g.end());
    s.max = g.back();
    // Matches np.median / np.percentile(..., 99) in the Python plotter.
    s.median = quantile_linear(g, 0.5);
    s.p99 = quantile_linear(g, 0.99);
    return s;
}

}  // namespace

void relax_torque_probe_header() {
    std::printf("%-9s %8s %10s %10s %10s %10s %8s %8s %12s\n",
                "box", "steps", "tau_max", "tau_p99", "tau_med",
                "tau_rms", "D1(nm)", "D2(nm)", "E(J)");
}

ProbeResult relax_torque_probe(Params& p, DemagState& demag,
                               const std::uint8_t* mask,
                               int n_chunks, int chunk, double alpha_relax,
                               const std::string& out_dir) {
    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
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
                              mask);
        m_top = std::move(r.m_top);
        m_bot = std::move(r.m_bot);
        step += chunk;
        // Diagnostics on the current state.
        effective_field_demag(m_top, m_bot, p, demag, H_top, H_bot, mask);
        tfield = tangential_torque_field(m_top, H_top);
        // Torque percentiles over the magnetic region only; the zeroed
        // vacuum cells would otherwise swamp the median.
        const TorqueStats ts = mask
            ? torque_stats(masked_values(tfield, mask))
            : torque_stats(tfield);
        const Ellipse e = skyrmion_ellipse_lcc(m_top, p.a, +1);
        const double E = total_energy(m_top, m_bot, p, demag, mask);
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
        // Per-chunk progress row, flushed so logs track the run live.
        std::printf("%4dx%-4d %8d %10.2e %10.2e %10.2e %10.2e "
                    "%8.1f %8.1f %12.4e\n",
                    p.nx, p.ny, step, ts.max, ts.p99, ts.median,
                    ts.rms, e.D1 * 1e9, e.D2 * 1e9, E);
        std::fflush(stdout);
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
    w.write("a", sc(p.a)); w.write("dt", sc(p.dt));
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

    // Self-describing relaxed-field snapshot (full m_top/m_bot + real
    // observables), same layout as m_eq.npz for the snapshot tools.
    const Ellipse ef = skyrmion_ellipse_lcc(m_top, p.a, +1);
    const Center2D cf = skyrmion_center_lcc_pbc(m_top, p.a, +1);
    const double diamf = skyrmion_diameter_lcc(m_top, p.a, +1);
    const double qtop = topological_charge(m_top, p.a);
    const double qbot = topological_charge(m_bot, p.a);
    const double psif = dw_angle(m_top, p.a, +1, /*mz_thresh=*/0.5);
    SnapshotBuffer snap(p.ny, p.nx, 1);
    snap.append(m_top, m_bot, step, step * p.dt, 0,
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

    return ProbeResult{s_d1.back() * 1e9, s_d2.back() * 1e9,
                       s_tmax.back(), s_E.back()};
}

}  // namespace skyrmion
