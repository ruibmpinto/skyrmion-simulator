/// \file
/// Shared helpers for the per-analysis sweep binaries: SLURM array
/// dispatch and a single-grid-point runner that covers the common
/// saf_skyrmion + (convergence | fixed-time) relax + drive pattern used
/// by the sweep analyses. Each binary builds its grid + pulse +
/// metadata and calls run_point.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/driver.hpp"
#include "skyrmion/sweep/relax.hpp"
#include "skyrmion/sweep/stepper.hpp"
#include "skyrmion/sweep/trace.hpp"
#include "skyrmion/sweep/trace_io.hpp"

#include <cmath>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace skyrmion {
namespace sweep {

/// Resolve which grid indices this process runs. If SLURM_ARRAY_TASK_ID
/// is set, run only that index (and validate range); otherwise run the
/// whole grid [0, n).
/// \param n Number of points in the binary's grid.
/// \return Either the single SLURM-selected index or every index in
///         [0, n), in order.
/// \throws std::runtime_error if SLURM_ARRAY_TASK_ID is set but falls
///         outside [0, n).
inline std::vector<int> resolve_indices(int n) {
    const char* env = std::getenv("SLURM_ARRAY_TASK_ID");
    if (env == nullptr) {
        std::vector<int> all(n);
        for (int i = 0; i < n; ++i) all[i] = i;
        return all;
    }
    const int idx = std::atoi(env);
    if (idx < 0 || idx >= n) {
        throw std::runtime_error(
            "resolve_indices: SLURM_ARRAY_TASK_ID out of range [0, "
            + std::to_string(n - 1) + "].");
    }
    return {idx};
}

/// Everything one grid point of a sweep needs.
struct PointConfig {
    Params* p = nullptr;               ///< built + precomputed by caller
    std::shared_ptr<Pulse> pulse;      ///< drive pulse
    bool use_demag = true;             ///< demag vs local-K_eff field model
    bool convergence_relax = true; ///< convergence-stop vs fixed-time relax
    int  n_drive = 0;                  ///< Drive steps
    int  sample_every = 1;             ///< Observable cadence, in steps
    int  n_relax_fixed = 0;   ///< fixed-time relax length (driver Phase 1)
    int  relax_max_steps = 200000;     ///< Step budget of the quench
    double relax_alpha = 1.0;          ///< Gilbert damping for the quench
    double relax_tol_torque = 1.0e-5;  ///< Torque gate (T); see relax()
    double relax_tol_dE = 1.0e-8;      ///< Relative-energy gate (demag)
    int  relax_check_every = 1000;     ///< Steps between quench checks
    double record_snapshot_at = -1.0;  ///< single trace snapshot; <0 none
    /// Field-snapshot stream (for animation).
    bool dump_snapshots = false;
    int  snapshot_every_drive = 1000;  ///< Drive frame cadence (steps)
    int  snapshot_every_relax = 0;     ///< Relax frame cadence (steps)
    int  max_snapshot_frames = 500;    ///< Snapshot buffer capacity
    std::string trace_path;            ///< Output trace .npz path
    std::string snapshot_path;         ///< Output snapshot .npz path
    Metadata metadata;                 ///< figure-specific; relax_* added here
    /// Progress-line cadence. print_every (steps) takes precedence when
    /// > 0; otherwise it is derived from print_dt (s) via dt, so every
    /// binary prints on a fixed simulated-time interval by default.
    double print_dt = 100.0e-12;       // 100 ps
    int print_every = 0;             ///< explicit step override; 0 => print_dt
};

/// Run one grid point: build the saf_skyrmion IC, relax (convergence or
/// fixed-time), drive, write the trace .npz and (optionally) a snapshot
/// .npz for animation.
///
/// On the convergence path the quench outcome is appended to
/// cfg.metadata (relax_mode, relax_converged, relax_n_steps,
/// relax_tau_max_final, relax_E_final) before the trace is written.
/// \param cfg Grid-point configuration; its Params are precomputed and
///        its metadata extended in place.
/// \throws std::runtime_error propagated from relax, run_trace or the
///         .npz writers.
inline void run_point(PointConfig& cfg) {
    Params& p = *cfg.p;
    precompute(p);

    // Effective progress cadence in steps: explicit print_every wins;
    // otherwise derive from the time interval print_dt and dt.
    int print_every = cfg.print_every;
    if (print_every <= 0 && cfg.print_dt > 0.0) {
        print_every = static_cast<int>(std::ceil(cfg.print_dt / p.dt));
        if (print_every < 1) print_every = 1;
    }

    std::unique_ptr<DemagState> demag;
    if (cfg.use_demag) {
        demag.reset(new DemagState(p, /*threads=*/0));
    }

    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    Field3 m_top = std::move(ic.m_top);
    Field3 m_bot = std::move(ic.m_bot);

    // Animation buffer is created before relaxation so the initial
    // (unrelaxed) and relaxed configurations are saved as phase-0 frames
    // ahead of the drive phase. Each frame stores the full lattice field
    // (m_top, m_bot at every site); positions are written once on write.
    std::unique_ptr<SnapshotBuffer> snaps;
    if (cfg.dump_snapshots) {
        snaps.reset(new SnapshotBuffer(p.ny, p.nx, cfg.max_snapshot_frames));
    }

    int n_relax_for_driver = 0;
    if (cfg.convergence_relax) {
        // Initial (unrelaxed) configuration: phase 0, step 0.
        if (snaps) {
            append_snapshot(*snaps, m_top, m_bot, 0, 0.0, 0, p);
        }
        RelaxResult rr = relax(m_top, m_bot, p, demag.get(),
                               cfg.relax_max_steps, cfg.relax_alpha,
                               cfg.relax_tol_torque, cfg.relax_tol_dE,
                               cfg.relax_check_every, print_every,
                               /*mask=*/nullptr);
        m_top = std::move(rr.m_top);
        m_bot = std::move(rr.m_bot);
        // Relaxed equilibrium: last phase-0 frame before the drive.
        if (snaps) {
            append_snapshot(*snaps, m_top, m_bot,
                            static_cast<int64_t>(rr.n_steps),
                            rr.n_steps * p.dt, 0, p);
        }
        cfg.metadata.add("relax_mode", std::string("convergence_stop"));
        cfg.metadata.add("relax_converged", rr.converged);
        cfg.metadata.add("relax_n_steps", static_cast<long long>(rr.n_steps));
        cfg.metadata.add("relax_tau_max_final", rr.tau_max);
        cfg.metadata.add("relax_E_final", rr.E_final);
    } else {
        n_relax_for_driver = cfg.n_relax_fixed;
        cfg.metadata.add("relax_mode", std::string("fixed_time"));
        // Fixed-time relax runs inside run_trace, which dumps the
        // initial + relaxed phase-0 endpoints itself.
    }

    std::unique_ptr<Stepper> stepper;
    if (cfg.use_demag)
        stepper.reset(new RK4DemagStepper(p, *demag, /*mask=*/nullptr));
    else
        stepper.reset(new RK4LocalKeffStepper(p, /*mask=*/nullptr));

    RunTraceArgs ta;
    ta.p = &p;
    ta.pulse = cfg.pulse;
    ta.n_relax = n_relax_for_driver;
    ta.n_drive = cfg.n_drive;
    ta.sample_every = cfg.sample_every;
    ta.step_drive = stepper.get();
    ta.step_relax = stepper.get();
    ta.m_top_init = m_top;
    ta.m_bot_init = m_bot;
    ta.record_snapshot_at = cfg.record_snapshot_at;
    ta.print_every = print_every;
    ta.snapshots = snaps.get();
    ta.snapshot_every_relax = cfg.snapshot_every_relax;
    ta.snapshot_every_drive = cfg.snapshot_every_drive;

    Trace trace = run_trace(ta);
    save_trace(cfg.trace_path, trace, cfg.metadata);

    if (cfg.dump_snapshots) {
        Field3 pos_top = lattice_positions(p.nx, p.ny, p.a);
        Field3 pos_bot = lattice_positions(p.nx, p.ny, p.a);
        for (int i = 0; i < p.ny; ++i)
            for (int j = 0; j < p.nx; ++j)
                pos_bot(i, j, 2) = -p.t_Co;
        snaps->write(cfg.snapshot_path, pos_top, pos_bot,
                     cfg.metadata.to_json());
    }
}

/// Gaussian sigma from FWHM.
/// \param fwhm Full width at half maximum, in the caller's unit
///        (seconds for the pulse widths used by the sweeps).
/// \return sigma = fwhm / (2 sqrt(2 ln 2)), in the same unit.
inline double fwhm_to_sigma(double fwhm) {
    return fwhm / (2.0 * std::sqrt(2.0 * std::log(2.0)));
}

/// "<stem>_snapshots.npz" beside the trace file.
/// \param trace_path Trace .npz path; a trailing ".npz" is stripped.
/// \return The snapshot archive path for that trace.
inline std::string snapshot_path_of(const std::string& trace_path) {
    const auto pos = trace_path.rfind(".npz");
    const std::string stem = (pos == std::string::npos)
        ? trace_path : trace_path.substr(0, pos);
    return stem + "_snapshots.npz";
}

/// Python D-tag: f'D{int(round(D*1e5)):03d}e-3' (e.g. 0.85e-3 ->
/// D085e-3).
/// \param D Interfacial DMI constant, in J/m^2.
/// \return The filename tag used by the Python sweep outputs.
inline std::string d_tag(double D) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "D%03de-3",
                  static_cast<int>(std::lround(D * 1e5)));
    return std::string(buf);
}

} // namespace sweep
} // namespace skyrmion
