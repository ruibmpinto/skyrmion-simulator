#include "skyrmion/stochastic/trajectory.hpp"

#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/diagnostics.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"

#include <cmath>
#include <limits>
#include <memory>
#include <utility>

namespace skyrmion {
namespace stochastic {

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Total core weight w = sum(1 - polarity*m_z). Used to decide whether
// the layer still has a core (matches run_single's NaN guard).
double core_weight(const Field3& m, int polarity) {
    double w = 0.0;
    for (int i = 0; i < m.ny; ++i)
        for (int j = 0; j < m.nx; ++j)
            w += 1.0 - polarity * m(i, j, 2);
    return w;
}

// PBC centre with the run_single guard: NaN if the core has collapsed.
Center2D guarded_center(const Field3& m, Real a, int polarity) {
    if (core_weight(m, polarity) > 0.0) {
        return skyrmion_center_pbc(m, a, polarity);
    }
    return {kNaN, kNaN};
}

Center2D guarded_center_lcc(const Field3& m, Real a, int polarity) {
    if (core_weight(m, polarity) > 0.0) {
        return skyrmion_center_lcc_pbc(m, a, polarity);
    }
    return {kNaN, kNaN};
}

bool all_finite(const std::vector<double>& v, int end) {
    for (int i = 0; i < end; ++i) if (!std::isfinite(v[i])) return false;
    return true;
}

} // namespace

StochasticPayload run_trajectory(const StochasticConfig& cfg,
                                 SnapshotBuffer* snaps) {
    // Effective temperature via uniform Joule heating.
    const Real T_eff = T_of_j(cfg.j_current, cfg.T_sub, cfg.R_th);

    Params p = make_default_params();
    p.nx = cfg.nx; p.ny = cfg.ny; p.dt = cfg.dt;
    p.J_current = cfg.j_current;
    if (cfg.skyrmion_R > 0.0)  p.skyrmion_R = cfg.skyrmion_R;
    if (cfg.skyrmion_dw > 0.0) p.skyrmion_dw = cfg.skyrmion_dw;
    if (cfg.D > 0.0)           p.D = cfg.D;
    p.H_ext = {0.0, 0.0, cfg.H_z};
    if (cfg.use_demag) p.demag_kind = DemagKind::Slab;
    p.pulse = std::make_shared<ConstantPulse>(cfg.j_current);
    precompute(p);
    attach_thermal(p, T_eff, cfg.R_th, cfg.seed);

    std::unique_ptr<DemagState> demag;
    if (cfg.use_demag) demag.reset(new DemagState(p, 0));

    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    Field3 m_top = std::move(ic.m_top);
    Field3 m_bot = std::move(ic.m_bot);

    ThermalRng rng(static_cast<std::uint64_t>(cfg.seed));
    const Real sigma = p.sigma_noise;
    HeunStochasticStepper stepper(p, demag.get(), rng, sigma, cfg.tol_norm,
                                  /*mask=*/nullptr);
    const Real a = p.a;

    auto dump = [&](const Field3& mt, const Field3& mb, int64_t step,
                    Real t, int32_t phase) {
        if (!snaps) return;
        const Center2D c = guarded_center(mt, a, +1);
        snaps->append(mt, mb, step, t, phase,
                      topological_charge(mt, a), topological_charge(mb, a),
                      c.cx, c.cy, skyrmion_diameter(mt, a, +1),
                      kNaN, kNaN, kNaN, kNaN);  // D1/D2/theta/psi not computed
    };

    // -------- Phase 0: relaxation (J = 0, noise on) --------------------------
    auto pulse_save = p.pulse;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    p.H_DL = 0.0; p.H_FL = 0.0;
    if (snaps) dump(m_top, m_bot, 0, 0.0, 0);
    {
        Real t = 0.0;
        for (int s = 1; s <= cfg.n_relax; ++s) {
            stepper.step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
            if (snaps && cfg.snapshot_every > 0
                && s % cfg.snapshot_every == 0 && s != cfg.n_relax) {
                dump(m_top, m_bot, s, t, 0);
            }
        }
    }
    if (snaps && cfg.n_relax > 0) dump(m_top, m_bot, cfg.n_relax,
                                       cfg.n_relax * p.dt, 0);
    p.pulse = pulse_save;
    p.H_DL = p.DL_SOT * cfg.j_current;
    p.H_FL = p.FL_SOT * cfg.j_current;

    // -------- Phase 1: drive (sampling) --------------------------------------
    const int n_samples = (cfg.n_drive + cfg.sample_every - 1) / cfg.sample_every;
    StochasticPayload pl;
    auto& P = pl;
    P.t_sample.resize(n_samples);
    P.cx_wrapped.resize(n_samples);  P.cy_wrapped.resize(n_samples);
    P.cx_wrapped_bot.resize(n_samples); P.cy_wrapped_bot.resize(n_samples);
    P.cx_lcc.resize(n_samples); P.cy_lcc.resize(n_samples);
    P.cx_lcc_bot.resize(n_samples); P.cy_lcc_bot.resize(n_samples);
    P.Q.resize(n_samples); P.Q_bot.resize(n_samples);
    P.diameter.resize(n_samples); P.diameter_bot.resize(n_samples);
    P.diameter_lcc.resize(n_samples); P.diameter_lcc_bot.resize(n_samples);
    P.norm_drift_max.resize(n_samples);

    if (snaps) dump(m_top, m_bot, 0, 0.0, 1);
    int s_idx = 0;
    Real t = 0.0;
    for (int step = 1; step <= cfg.n_drive; ++step) {
        stepper.step(m_top, m_bot, t, p.dt, p);
        t += p.dt;
        if (step % cfg.sample_every == 0 && s_idx < n_samples) {
            P.t_sample[s_idx] = step * p.dt;
            const Center2D ct = guarded_center(m_top, a, +1);
            const Center2D cb = guarded_center(m_bot, a, -1);
            P.cx_wrapped[s_idx] = ct.cx; P.cy_wrapped[s_idx] = ct.cy;
            P.cx_wrapped_bot[s_idx] = cb.cx; P.cy_wrapped_bot[s_idx] = cb.cy;
            P.Q[s_idx] = topological_charge(m_top, a);
            P.Q_bot[s_idx] = topological_charge(m_bot, a);
            P.diameter[s_idx] = skyrmion_diameter(m_top, a, +1);
            P.diameter_bot[s_idx] = skyrmion_diameter(m_bot, a, -1);
            P.diameter_lcc[s_idx] = skyrmion_diameter_lcc(m_top, a, +1);
            P.diameter_lcc_bot[s_idx] = skyrmion_diameter_lcc(m_bot, a, -1);
            const Center2D cl = guarded_center_lcc(m_top, a, +1);
            const Center2D clb = guarded_center_lcc(m_bot, a, -1);
            P.cx_lcc[s_idx] = cl.cx; P.cy_lcc[s_idx] = cl.cy;
            P.cx_lcc_bot[s_idx] = clb.cx; P.cy_lcc_bot[s_idx] = clb.cy;
            P.norm_drift_max[s_idx] = stepper.last_norm_drift();
            ++s_idx;
        }
        if (snaps && cfg.snapshot_every > 0
            && step % cfg.snapshot_every == 0 && step != cfg.n_drive) {
            dump(m_top, m_bot, step, t, 1);
        }
    }
    if (snaps) dump(m_top, m_bot, cfg.n_drive, cfg.n_drive * p.dt, 1);

    // -------- Derived quantities ---------------------------------------------
    P.T_effective = T_eff;
    P.sigma_noise = sigma;
    P.flip_index = detect_annihilation(P.Q, cfg.q_threshold, cfg.k_consecutive);
    P.alive_at_end = (P.flip_index == -1);
    const int end = (P.flip_index == -1)
        ? n_samples : std::max(P.flip_index, 4);
    const Real L_x = cfg.nx * a;
    const Real L_y = cfg.ny * a;

    P.cx_unwrapped.assign(n_samples, kNaN);
    P.cy_unwrapped.assign(n_samples, kNaN);
    P.cx_unwrapped_bot.assign(n_samples, kNaN);
    P.cy_unwrapped_bot.assign(n_samples, kNaN);
    P.v_x = P.v_y = P.hall_deg = P.sigma_y = kNaN;
    P.velocity = kNaN;
    P.v_x_bot = P.v_y_bot = P.hall_deg_bot = kNaN;
    P.velocity_bot = kNaN;

    auto fit_layer = [&](const std::vector<double>& cx_w,
                         const std::vector<double>& cy_w,
                         std::vector<double>& cx_u, std::vector<double>& cy_u,
                         double& vx, double& vy, double& vmag, double& th,
                         double* sig_y) {
        if (end < 4 || !all_finite(cx_w, end)) return;
        std::vector<double> cxw(cx_w.begin(), cx_w.begin() + end);
        std::vector<double> cyw(cy_w.begin(), cy_w.begin() + end);
        Trajectory2D u = unwrap_trajectory(cxw, cyw, L_x, L_y);
        for (int i = 0; i < end; ++i) { cx_u[i] = u.cx[i]; cy_u[i] = u.cy[i]; }
        std::vector<double> tt(P.t_sample.begin(), P.t_sample.begin() + end);
        HallFit f = hall_angle(tt, u.cx, u.cy, 0.5);
        vx = f.v_x; vy = f.v_y; th = f.theta_deg;
        vmag = std::sqrt(vx * vx + vy * vy);
        if (sig_y && end >= 8) {
            const int h = end / 2;
            double mean = 0.0;
            for (int i = h; i < end; ++i) mean += u.cy[i];
            mean /= (end - h);
            double var = 0.0;
            for (int i = h; i < end; ++i) var += (u.cy[i] - mean) * (u.cy[i] - mean);
            *sig_y = std::sqrt(var / (end - h - 1));  // ddof = 1
        }
    };
    fit_layer(P.cx_wrapped, P.cy_wrapped, P.cx_unwrapped, P.cy_unwrapped,
              P.v_x, P.v_y, P.velocity, P.hall_deg, &P.sigma_y);
    fit_layer(P.cx_wrapped_bot, P.cy_wrapped_bot,
              P.cx_unwrapped_bot, P.cy_unwrapped_bot,
              P.v_x_bot, P.v_y_bot, P.velocity_bot, P.hall_deg_bot, nullptr);
    return pl;
}

} // namespace stochastic
} // namespace skyrmion
