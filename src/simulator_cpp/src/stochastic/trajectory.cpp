#include "skyrmion/stochastic/dissipation_partition.hpp"
#include "skyrmion/stochastic/trajectory.hpp"

#include "skyrmion/demag.hpp"
#include "skyrmion/fields.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/diagnostics.hpp"
#include "skyrmion/stochastic/equilibrate.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <exception>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

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
    // The drive profile is caller-supplied with no fallback: a missing
    // pulse is a configuration error, not an implied DC run.
    if (!cfg.drive_pulse) {
        throw std::runtime_error(
            "run_trajectory: cfg.drive_pulse is unset; pass an explicit "
            "pulse (ConstantPulse(j_current) for a DC drive).");
    }
    // Effective temperature via uniform Joule heating. T_sub == 0 is the
    // noise-free mode: skip T_of_j, which rejects non-positive T_sub.
    // Joule heating cannot be represented there, so a non-zero R_th is
    // an error rather than something to drop quietly.
    const bool noise_free = (cfg.T_sub == 0.0);
    if (noise_free && cfg.R_th != 0.0) {
        throw std::runtime_error(
            "run_trajectory: T_sub = 0 selects the noise-free mode, "
            "which cannot carry Joule heating; got R_th != 0.");
    }
    const Real T_eff = noise_free
        ? Real{0} : T_of_j(cfg.j_current, cfg.T_sub, cfg.R_th);

    Params p = make_default_params();
    p.nx = cfg.nx; p.ny = cfg.ny; p.dt = cfg.dt;
    if (cfg.skyrmion_R > 0.0)  p.skyrmion_R = cfg.skyrmion_R;
    if (cfg.skyrmion_dw > 0.0) p.skyrmion_dw = cfg.skyrmion_dw;
    if (cfg.D > 0.0)           p.D = cfg.D;
    if (cfg.K_top > 0.0)       p.K_top = cfg.K_top;
    if (cfg.a > 0.0)           p.a = cfg.a;
    p.H_ext = {0.0, 0.0, cfg.H_z};
    if (cfg.use_demag) {
        p.demag_kind = cfg.demag_kind;
        p.demag_accuracy = cfg.demag_accuracy;
        p.demag_tol_conv = cfg.demag_tol_conv;
    }
    p.pulse = cfg.drive_pulse;
    precompute(p);
    // attach_thermal accepts T = 0 as the deterministic limit and sets
    // sigma_noise = 0 there, so the noise-free mode needs no separate
    // path; only T_of_j above had to be bypassed.
    attach_thermal(p, T_eff, cfg.R_th, cfg.seed);

    std::unique_ptr<DemagState> demag;
    if (cfg.use_demag) demag.reset(new DemagState(p, cfg.fft_threads));

    // Initial condition: a caller-provided (pre-relaxed) field if given,
    // else a fresh SAF skyrmion seed.
    Field3 m_top, m_bot;
    if (cfg.m_init_top && cfg.m_init_bot) {
        m_top = *cfg.m_init_top;
        m_bot = *cfg.m_init_bot;
    } else {
        SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a,
                                  p.skyrmion_R, p.skyrmion_dw);
        m_top = std::move(ic.m_top);
        m_bot = std::move(ic.m_bot);
    }

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
    // Equilibrate to a thermal size plateau (cfg.equilibrate) or run a
    // fixed cfg.n_relax steps. The relaxed LCC ellipse is recorded as
    // the pre-drive finite-T equilibrium size. Only the initial and
    // relaxed frames are snapshotted in this phase.
    auto pulse_save = p.pulse;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    if (snaps) dump(m_top, m_bot, 0, 0.0, 0);
    int n_relax_used = 0;
    bool equil_converged = false;
    double d1_relaxed = kNaN, d2_relaxed = kNaN;
    {
        if (cfg.equilibrate) {
            const EquilResult er = equilibrate_to_plateau(
                m_top, m_bot, stepper, p, cfg.equil_check_every,
                cfg.equil_window, cfg.equil_tol, cfg.equil_k_consec,
                cfg.equil_max_steps, /*progress_every=*/0);
            n_relax_used = er.n_used;
            equil_converged = er.converged;
            d1_relaxed = er.d1_relaxed;
            d2_relaxed = er.d2_relaxed;
        } else {
            Real t = 0.0;
            for (int s = 1; s <= cfg.n_relax; ++s) {
                stepper.step(m_top, m_bot, t, p.dt, p);
                t += p.dt;
            }
            n_relax_used = cfg.n_relax;
            // Relaxed size = LCC ellipse of the (possibly pre-thermalized)
            // starting field; for stage-3 drives n_relax=0, so this
            // measures the loaded thermal state.
            try {
                const Ellipse e = skyrmion_ellipse_lcc(m_top, a, +1);
                d1_relaxed = e.D1; d2_relaxed = e.D2;
            } catch (const std::exception&) {}
        }
    }
    if (snaps) dump(m_top, m_bot, n_relax_used,
                    n_relax_used * p.dt, 0);
    p.pulse = pulse_save;

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
    P.D1_top.resize(n_samples); P.D2_top.resize(n_samples);
    P.theta_top.resize(n_samples);
    P.D1_bot.resize(n_samples); P.D2_bot.resize(n_samples);
    P.theta_bot.resize(n_samples);
    P.norm_drift_max.resize(n_samples);
    // Dissipation split: only sized (and only computed) on request, so
    // the default drive path is untouched.
    Field3 h_diss_top(cfg.ny, cfg.nx), h_diss_bot(cfg.ny, cfg.nx);
    Field3 dmdt_diss(cfg.ny, cfg.nx);
    const bool free_y_diss = (p.demag_kind == DemagKind::Racetrack);
    if (cfg.record_dissipation) {
        P.diss_trans.resize(n_samples); P.diss_def.resize(n_samples);
        P.diss_total.resize(n_samples);
        P.v_fit_x.resize(n_samples); P.v_fit_y.resize(n_samples);
        if (!demag) {
            throw std::runtime_error(
                "run_trajectory: record_dissipation needs the demag "
                "field path (use_demag).");
        }
    }

    if (snaps) dump(m_top, m_bot, 0, 0.0, 1);
    int s_idx = 0;
    Real t = 0.0;
    const auto t_drive_start = std::chrono::steady_clock::now();
    for (int step = 1; step <= cfg.n_drive; ++step) {
        stepper.step(m_top, m_bot, t, p.dt, p);
        t += p.dt;
        // Timestamped drive progress (gated by cfg.progress_every).
        if (cfg.progress_every > 0 && step % cfg.progress_every == 0) {
            const double wall = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - t_drive_start).count();
            std::printf(
                "    drive step %d/%d (%.0f ps), wall %.0fs, "
                "d=%.1f nm, Q=%.2f\n",
                step, cfg.n_drive, step * p.dt * 1e12, wall,
                skyrmion_diameter_lcc(m_top, a, +1) * 1e9,
                topological_charge(m_top, a));
            std::fflush(stdout);
        }
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
            // Elliptical axes of the LCC; NaN if the core has collapsed
            // (skyrmion_ellipse_lcc throws on < 3 sites).
            try {
                const Ellipse et = skyrmion_ellipse_lcc(m_top, a, +1);
                P.D1_top[s_idx] = et.D1; P.D2_top[s_idx] = et.D2;
                P.theta_top[s_idx] = et.theta;
            } catch (const std::exception&) {
                P.D1_top[s_idx] = kNaN; P.D2_top[s_idx] = kNaN;
                P.theta_top[s_idx] = kNaN;
            }
            try {
                const Ellipse eb = skyrmion_ellipse_lcc(m_bot, a, -1);
                P.D1_bot[s_idx] = eb.D1; P.D2_bot[s_idx] = eb.D2;
                P.theta_bot[s_idx] = eb.theta;
            } catch (const std::exception&) {
                P.D1_bot[s_idx] = kNaN; P.D2_bot[s_idx] = kNaN;
                P.theta_bot[s_idx] = kNaN;
            }
            const Center2D cl = guarded_center_lcc(m_top, a, +1);
            const Center2D clb = guarded_center_lcc(m_bot, a, -1);
            P.cx_lcc[s_idx] = cl.cx; P.cy_lcc[s_idx] = cl.cy;
            P.cx_lcc_bot[s_idx] = clb.cx; P.cy_lcc_bot[s_idx] = clb.cy;
            P.norm_drift_max[s_idx] = stepper.last_norm_drift();
            if (cfg.record_dissipation) {
                // Deterministic dm/dt of the top layer, from the same
                // two calls the stepper's predictor makes minus the
                // noise term. mask is null here (racetrack free-y is
                // carried by the demag kernel and the field assembly).
                effective_field_demag(m_top, m_bot, p, *demag,
                                      h_diss_top, h_diss_bot, nullptr);
                llgs_rhs(m_top, h_diss_top, p, t, dmdt_diss, nullptr);
                const DissipationSplit ds = dissipation_partition(
                    m_top, dmdt_diss, a, free_y_diss);
                P.diss_trans[s_idx] = ds.trans;
                P.diss_def[s_idx] = ds.def;
                P.diss_total[s_idx] = ds.total;
                P.v_fit_x[s_idx] = ds.v_x;
                P.v_fit_y[s_idx] = ds.v_y;
            }
            ++s_idx;
        }
        if (snaps && cfg.snapshot_every > 0
            && step % cfg.snapshot_every == 0 && step != cfg.n_drive) {
            dump(m_top, m_bot, step, t, 1);
        }
    }
    if (snaps) dump(m_top, m_bot, cfg.n_drive, cfg.n_drive * p.dt, 1);

    // -------- Derived quantities ---------------------------------------------
    P.T_sub = cfg.T_sub;
    P.j_current = cfg.j_current;
    P.D1_relaxed_top = d1_relaxed;
    P.D2_relaxed_top = d2_relaxed;
    P.n_relax_used = n_relax_used;
    P.equil_converged = equil_converged;
    P.T_effective = T_eff;
    P.sigma_noise = sigma;
    P.flip_index = detect_annihilation(P.Q, cfg.q_threshold, cfg.k_consecutive);
    P.alive_at_end = (P.flip_index == -1);
    // Final top-layer m_z snapshot for the field-classifier survival
    // criterion (applied at aggregation, not here).
    P.ny = cfg.ny;
    P.nx = cfg.nx;
    P.mz_final_top.resize(
        static_cast<std::size_t>(cfg.ny) * cfg.nx);
    for (int i = 0; i < cfg.ny; ++i) {
        for (int j = 0; j < cfg.nx; ++j) {
            P.mz_final_top[static_cast<std::size_t>(i) * cfg.nx + j] =
                static_cast<float>(m_top(i, j, 2));
        }
    }
    const int end = (P.flip_index == -1)
        ? n_samples : std::max(P.flip_index, 4);
    const Real L_x = cfg.nx * a;
    const Real L_y = cfg.ny * a;
    P.L_x = L_x;
    P.L_y = L_y;

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
        // Free-y racetrack: y is not periodic, never unwrap it.
        const bool periodic_y = (p.demag_kind != DemagKind::Racetrack);
        Trajectory2D u = unwrap_trajectory(cxw, cyw, L_x, L_y, periodic_y);
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
    // Fit the LCC (largest-connected-component core) tracker: unlike
    // the whole-lattice centroid it stays on the skyrmion once other
    // domains nucleate.
    fit_layer(P.cx_lcc, P.cy_lcc, P.cx_unwrapped, P.cy_unwrapped,
              P.v_x, P.v_y, P.velocity, P.hall_deg, &P.sigma_y);
    fit_layer(P.cx_lcc_bot, P.cy_lcc_bot,
              P.cx_unwrapped_bot, P.cy_unwrapped_bot,
              P.v_x_bot, P.v_y_bot, P.velocity_bot, P.hall_deg_bot, nullptr);
    return pl;
}

} // namespace stochastic
} // namespace skyrmion
