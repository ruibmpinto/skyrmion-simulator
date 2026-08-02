// Pulse-shape driving study at finite temperature (racetrack, Set A).
//
// Drives a pre-equilibrated SAF skyrmion with one of six current-pulse
// shapes and records the standard per-trajectory payload, so the same
// aggregation and field classifier used by the racetrack campaign apply
// unchanged. The shapes are square, half-sine, the three triangular
// asymmetries (peak pinned to the leading edge, to the trailing edge, or
// at the midpoint) and Gaussian.
//
// Seeding, by temperature:
//   T_sub  > 0 : the campaign's cached thermal state for this
//                (T_sub, ens), m_thermal_T{T}_ens{ens}.npz.
//   T_sub == 0 : the campaign's T=0 relaxed state, m_eq_{nx}x{ny}.npz,
//                run noise-free. One member only -- a deterministic run
//                has no ensemble.
// Both live in the campaign directory (SEED_DIR); this study never
// writes there, its own output goes to OUT_ROOT.
//
// The simulated window is the pulse duration plus a settle interval, so
// the skyrmion is observed coming to rest after the drive ends. That is
// what makes rise/fall behaviour and the final displacement measurable,
// unlike the campaign's DC drive which never turns off.
//
// SLURM array dispatch is per TRAJECTORY over the flattened
// (shape, peak J, T_sub, ens) grid; SLURM_ARRAY_TASK_ID picks one.
// Selected by env var:
//   PS_CASE  index into the box/material case table (required)
//   PS_STAGE "pilot", "pilot_hi", "pilot_hi_all", "production" or
//            "width" (required)
//   SEED_DIR campaign directory holding m_eq / m_thermal (required)
//
// The two stages differ only in grid: the pilot sweeps peak J up INTO
// the labyrinth regime with a small ensemble, to locate the usable
// current per shape; production then runs a larger ensemble over the
// currents the pilot cleared.
#include "skyrmion/lattice.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/sweep_common.hpp"   // resolve_indices
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;
using namespace skyrmion::stochastic;

namespace {

// FFTW thread count for the demag transforms, from OMP_NUM_THREADS.
int fft_threads_from_env() {
    const char* e = std::getenv("OMP_NUM_THREADS");
    if (e && *e) {
        const int n = std::atoi(e);
        if (n > 0) return n;
    }
    return 1;
}

// Required environment variable; absence is an error, never a default.
std::string require_env(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || *v == '\0') {
        std::fprintf(stderr,
                     "error: %s must be set (no default).\n", name);
        std::exit(1);
    }
    return std::string(v);
}

// The six shapes compared by the study.
enum class Shape {
    Square, HalfSine, TriSharpRise, TriSharpFall, TriSymmetric, Gaussian
};

// Per-shape usable peak-current caps in units of 1e11 A/m^2, indexed
// [box case][temperature], temperatures in t_sub_list order
// (T = 0, 10, 50, 100 K). Caps are DEFINED AT FINITE T only: a T = 0
// point is one deterministic realization and cannot support the
// all-members-survived criterion, so the T = 0 slot carries the
// T = 10 K cap rather than a value of its own. A cap is the pilot's CONTIGUOUS window for
// that box: the largest peak current at which every realization stayed
// a skyrmion, with every lower current also surviving. Square at T=50
// on 350x500 is capped at 5 even though 8 survived, because 6 failed.
//
// Caps are per box and per temperature. A cap of 0 means the cell is
// NOT RUN, for one of two reasons that the driver distinguishes by
// whether the whole box row is zero:
//   - the whole row is zero: that box's pilot is unclassified, and
//     production refuses rather than borrowing another box's numbers;
//   - a single entry is zero: that (box, T) has no usable window at
//     all. 350x500 at T=100 is such a case. There the thermally
//     swollen skyrmion already spans 0.68 of the 700 nm track at the
//     LOWEST current in the ladder, so every cell is a spanning stripe
//     before the drive does anything. The box is too short at that
//     temperature; results there would measure the box, not the
//     physics. 700x500 at T=100 sits at 0.34 and is unaffected.
// T = 0 has no cap of its own (a single deterministic realization
// cannot support an all-members-survived criterion) and inherits the
// T = 10 K value.
struct ShapeSpec { Shape shape; const char* tag; double j_cap[3][4]; };

// Build the drive profile. `t_p` is the pulse duration; the pulse always
// starts at t = 0 because the drive phase does. `peak_j` is the peak
// amplitude for every shape, so shapes are compared at matched peak and
// duration; the delivered charge and action differ between shapes by
// construction and are recomputed from the profile during analysis.
std::shared_ptr<Pulse> make_pulse(Shape shape, double peak_j, double t_p,
                                  double gauss_fwhm) {
    switch (shape) {
    case Shape::Square:
        return std::make_shared<SquarePulse>(peak_j, 0.0, t_p);
    case Shape::HalfSine:
        return std::make_shared<HalfSinePulse>(peak_j, 0.0, t_p);
    case Shape::TriSharpRise:
        // Peak at the leading edge: instantaneous rise, linear fall.
        return std::make_shared<TrianglePulse>(peak_j, 0.0, 0.0, t_p);
    case Shape::TriSharpFall:
        // Peak at the trailing edge: linear rise, instantaneous fall.
        return std::make_shared<TrianglePulse>(peak_j, 0.0, t_p, t_p);
    case Shape::TriSymmetric:
        return std::make_shared<TrianglePulse>(peak_j, 0.0, 0.5*t_p, t_p);
    case Shape::Gaussian:
        // Centred in the window. Unlike the others the Gaussian has no
        // finite support, so a little current leaks into the settle
        // interval; the analysis integrates the profile over the whole
        // simulated window rather than assuming it stops at t_p.
        return std::make_shared<GaussianPulse>(
            peak_j, 0.5*t_p, gauss_fwhm);
    }
    // Unreachable for a valid enumerator; refuse rather than guess.
    throw std::runtime_error("make_pulse: unhandled shape.");
}

}  // namespace

int main() {
    // ----- Run configuration -------------------------------------------------
    // Box / material cases, hk36 Set A at the campaign DMI. Selected by
    // the PS_CASE env var; the plan runs case 0 in full first and uses
    // the longer tracks only to confirm box-independence.
    struct Case { double k_top; double dmi; int nx; int ny;
                  const char* tag; };
    const std::vector<Case> cases = {
        {1.3106e6, 0.72e-3, 350, 500, "hk36_D0p72_350x500"},
        {1.3106e6, 0.72e-3, 700, 500, "hk36_D0p72_700x500"},
        {1.3106e6, 0.72e-3, 1400, 500, "hk36_D0p72_1400x500"},
    };
    const std::string ps_case_env = require_env("PS_CASE");
    const int case_idx = std::atoi(ps_case_env.c_str());
    if (case_idx < 0 || case_idx >= static_cast<int>(cases.size())) {
        std::fprintf(stderr,
                     "error: PS_CASE=%d out of range [0, %zu].\n",
                     case_idx, cases.size() - 1);
        return 1;
    }
    const Case cs = cases[case_idx];
    const int nx = cs.nx, ny = cs.ny;
    // Campaign directory supplying the seeds; never written to here.
    const std::string seed_dir = require_env("SEED_DIR");

    // Rows: case 0 = 350x500 (measured), case 1 = 700x500, case 2 =
    // 1400x500. Cases 1 and 2 stay at 0 until their own pilots,
    // including the high-current extension, have been classified.
    const std::vector<ShapeSpec> shapes = {
        {Shape::Square,       "square",
         {{6, 6, 5, 0}, {6, 6, 5, 3}, {6, 6, 5, 3}}},
        {Shape::HalfSine,     "halfsine",
         {{8, 8, 6, 0}, {8, 8, 6, 4}, {8, 8, 6, 4}}},
        {Shape::TriSharpRise, "tri_sharprise",
         {{9, 9, 6, 0}, {9, 9, 6, 5}, {9, 9, 6, 5}}},
        {Shape::TriSharpFall, "tri_sharpfall",
         {{10, 10, 9, 0}, {10, 10, 9, 6}, {10, 10, 9, 6}}},
        {Shape::TriSymmetric, "tri_symmetric",
         {{9, 9, 6, 0}, {9, 9, 6, 5}, {9, 9, 6, 5}}},
        {Shape::Gaussian,     "gaussian",
         {{9, 9, 6, 0}, {9, 9, 6, 5}, {9, 9, 6, 5}}},
    };
    // Stage grid. Pilot: span peak J past the expected boundary so the
    // threshold is bracketed rather than assumed, with a small ensemble
    // (5 members resolve 0/5 vs 5/5 survival). Production: the currents
    // the stability maps clear, with a full ensemble.
    // Reference for the production list, docs/stability_box_size_hk36:
    // 2e11 stays compact at every T up to 100 K; 3e11 is the last
    // surviving current at 100 K; 4e11 is entirely labyrinth there.
    // Pilot: one uniform ladder for every cell, deliberately run past
    // the boundary so the threshold is bracketed. Production: the same
    // ladder truncated per (shape, T) at that cell's measured cap, so
    // no run is spent inside the labyrinth regime and every shape is
    // driven as hard as it actually tolerates.
    const std::string ps_stage = require_env("PS_STAGE");
    std::vector<double> peak_j_list;
    bool apply_caps = false;
    bool extend_only_unfailed = false;
    bool width_sweep = false;
    // Pulse durations for the width stage, in seconds. 500 ps is
    // absent: that slice is the completed production run.
    const std::vector<double> t_pulse_list = {
        100.0e-12, 200.0e-12, 300.0e-12, 400.0e-12,
        600.0e-12, 700.0e-12};
    // Ceiling of the low-current pilot ladder, in units of 1e11 A/m^2.
    // A cap equal to this means no failure was seen up to the ceiling,
    // so the true boundary lies above it.
    const double low_ladder_ceiling = 8.0;
    int n_ens = 0;
    if (ps_stage == "pilot") {
        peak_j_list = {2.0e11, 3.0e11, 4.0e11, 5.0e11, 6.0e11, 8.0e11};
        n_ens = 5;
    } else if (ps_stage == "pilot_hi") {
        // Extension above the first pilot's ceiling, to find the
        // current at which EVERY temperature fails. Disjoint from the
        // pilot ladder so the two stages compose into one table per
        // box without re-running anything. Restricted to the
        // (shape, T) cells that never failed on the low ladder --
        // a cell that already failed has its boundary bracketed and
        // gains nothing from more current.
        peak_j_list = {9.0e11, 1.0e12, 1.2e12, 1.5e12};
        extend_only_unfailed = true;
        n_ens = 5;
    } else if (ps_stage == "pilot_hi_all") {
        // Same high ladder as pilot_hi but with NO restriction and no
        // dependence on a measured cap table: every shape at every
        // temperature. Deliberate overhead, for a box whose low-ladder
        // classification is not yet available, so the high-current
        // search need not wait for it.
        peak_j_list = {9.0e11, 1.0e12, 1.2e12, 1.5e12};
        n_ens = 5;
    } else if (ps_stage == "width") {
        // Pulse-width sweep. Currents are not swept: each (shape, T)
        // runs at its cap and at half that cap, so the cap cells show
        // where widening the pulse breaks the state and the half-cap
        // cells stay alive across the whole width range, giving an
        // unbroken efficiency curve. The caps are the t_p = 500 ps
        // ones and are NOT re-measured, so cap cells at the widest
        // pulses are expected to fail -- a 700 ps pulse carries 1.4x
        // the charge of a 500 ps one at equal peak.
        apply_caps = true;
        width_sweep = true;
        n_ens = 25;
    } else if (ps_stage == "production") {
        peak_j_list = {0.5e11, 1.0e11, 2.0e11, 3.0e11, 4.0e11,
                       5.0e11, 6.0e11, 8.0e11};
        apply_caps = true;
        n_ens = 25;
    } else if (ps_stage == "production_ext") {
        // Extension of production above the 8e11 ladder ceiling, for the
        // shapes whose measured cap reaches 9-10e11. The two rungs are
        // disjoint from the production ladder, so this composes with the
        // completed production run without re-running anything; the cap
        // truncation keeps only cells where the rung is within the cap.
        peak_j_list = {9.0e11, 1.0e12};
        apply_caps = true;
        n_ens = 25;
    } else {
        std::fprintf(stderr,
                     "error: PS_STAGE=%s unknown; expected "
                     "\"pilot\", \"pilot_hi\", \"pilot_hi_all\", "
                     "\"production\", \"production_ext\" or "
                     "\"width\".\n",
                     ps_stage.c_str());
        return 1;
    }
    // T_sub = 0 is the deterministic baseline; the rest are thermal.
    const std::vector<double> t_sub_list = {0.0, 10.0, 50.0, 100.0};
    const double cell_a = make_default_params().a;
    const double dt = 5.0e-14;
    // Default pulse duration, used by every stage except "width",
    // which sweeps it. The window is always the duration plus an equal
    // unforced settle, and the Gaussian FWHM is half the duration, so
    // all six shapes are observed over the same span relative to their
    // own pulse at every width.
    const double t_pulse = 500.0e-12;
    const int sample_every = 200;           // 10 ps
    // Field-dump stream: one animation per CELL, so every
    // configuration is inspectable as a GIF. The designated member is
    // ens 0, which also covers the single deterministic member at
    // T = 0. The interval is derived per trajectory to give ~20 frames
    // whatever the window length.
    const long long seed_base = 202;
    const double r_th = 0.0;
    const double tol_norm = 5.0e-3;
    const double q_threshold = 0.5;
    const int k_consecutive = 10;
    // Racetrack: periodic x, free y, with free-y ghost cells for
    // exchange/DMI, matching the campaign this study seeds from.
    const DemagKind demag_kind = DemagKind::Racetrack;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    // The production extension writes into the production directory: its
    // cells are production cells above the 8e11 ladder, and they compose
    // with the completed production run rather than forming a stage of
    // their own.
    const std::string stage_dir =
        (ps_stage == "production_ext") ? "production" : ps_stage;
    const std::string out_dir =
        std::string("output/sweeps_driving_T/pulse_shape/") + stage_dir
        + "/" + cs.tag;
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    const int fft_threads = fft_threads_from_env();

    Field3 pos_top = lattice_positions(nx, ny, cell_a);
    Field3 pos_bot = lattice_positions(nx, ny, cell_a);

    // ----- Flat per-trajectory grid ------------------------------------------
    // A deterministic (T_sub = 0) cell has exactly one member; adding
    // ensemble copies of a noise-free run would duplicate work.
    struct Traj {
        int shape_idx; double peak_j; double T_sub; int cell_idx; int ens;
        // Pulse duration for this trajectory. Constant except in the
        // width stage, where it is the swept axis.
        double t_pulse;
    };
    std::vector<Traj> grid;
    int cell_idx = 0;
    for (std::size_t si = 0; si < shapes.size(); ++si) {
        for (std::size_t ti = 0; ti < t_sub_list.size(); ++ti) {
            const double T_sub = t_sub_list[ti];
            // Cap in A/m^2 for this (shape, T); the half-unit tolerance
            // absorbs the 1e11 scaling without a float-equality test.
            if (extend_only_unfailed) {
                const double cap_units = shapes[si].j_cap[case_idx][ti];
                if (cap_units <= 0.0) {
                    std::fprintf(stderr,
                        "error: pilot_hi requested for box %s but the "
                        "low-ladder cap for shape %s at T=%.0f K is "
                        "unmeasured; classify that box's pilot first.\n",
                        cs.tag, shapes[si].tag, t_sub_list[ti]);
                    return 1;
                }
                // Already failed below the ceiling: boundary known.
                if (cap_units < low_ladder_ceiling) continue;
            }
            double cap = std::numeric_limits<double>::infinity();
            if (apply_caps) {
                const double cap_units = shapes[si].j_cap[case_idx][ti];
                if (cap_units <= 0.0) {
                    // Whole row zero => the box is unclassified, which
                    // is an error. A lone zero => that temperature has
                    // no usable window and is deliberately skipped.
                    bool row_all_zero = true;
                    for (int tj = 0; tj < 4; ++tj) {
                        if (shapes[si].j_cap[case_idx][tj] > 0.0) {
                            row_all_zero = false;
                            break;
                        }
                    }
                    if (row_all_zero) {
                        std::fprintf(stderr,
                            "error: production requested for box %s but "
                            "the caps for shape %s are unmeasured. "
                            "Classify that box's pilot first; refusing "
                            "to reuse another box's caps.\n",
                            cs.tag, shapes[si].tag);
                        return 1;
                    }
                    std::printf(
                        "  skipping %s T=%.0f K: no usable window in "
                        "this box\n", shapes[si].tag, t_sub_list[ti]);
                    continue;
                }
                cap = cap_units*1.0e11 + 0.05e11;
            }
            if (width_sweep) {
                // Two currents only: the cap, and the largest ladder
                // rung at or below half the cap. Taking a rung rather
                // than cap/2 exactly keeps the value on the measured
                // grid and errs low, so the half-cap cell is certain to
                // survive at every width.
                const double cap_units =
                    shapes[si].j_cap[case_idx][ti];
                double half_units = 0.0;
                for (double rung : {0.5, 1.0, 2.0, 3.0, 4.0, 5.0,
                                    6.0, 8.0}) {
                    if (rung <= 0.5*cap_units) half_units = rung;
                }
                if (half_units <= 0.0) {
                    std::fprintf(stderr,
                        "error: no ladder rung at or below half the cap "
                        "for shape %s at T=%.0f K (cap %.3g).\n",
                        shapes[si].tag, T_sub, cap_units);
                    return 1;
                }
                const double j_pair[2] = {cap_units*1.0e11,
                                          half_units*1.0e11};
                for (double peak_j : j_pair) {
                    for (double t_p : t_pulse_list) {
                        const int members =
                            (T_sub == 0.0) ? 1 : n_ens;
                        for (int ens = 0; ens < members; ++ens) {
                            grid.push_back({static_cast<int>(si),
                                            peak_j, T_sub, cell_idx,
                                            ens, t_p});
                        }
                        ++cell_idx;
                    }
                }
                continue;
            }
            for (double peak_j : peak_j_list) {
                if (peak_j > cap) continue;
                const int members = (T_sub == 0.0) ? 1 : n_ens;
                for (int ens = 0; ens < members; ++ens) {
                    grid.push_back({static_cast<int>(si), peak_j, T_sub,
                                    cell_idx, ens, t_pulse});
                }
                ++cell_idx;
            }
        }
    }
    std::printf("pulse-shape grid [%s]: %zu trajectories over %d cells "
                "(box %dx%d, tag %s), %d anim streams\n",
                ps_stage.c_str(), grid.size(), cell_idx, nx, ny,
                cs.tag, cell_idx);

    for (int idx : resolve_indices(static_cast<int>(grid.size()))) {
        const Traj tr = grid[idx];
        const ShapeSpec sp = shapes[tr.shape_idx];
        const bool noise_free = (tr.T_sub == 0.0);
        // Window, Gaussian width and step count all follow this
        // trajectory's duration, so every shape is observed over the
        // same span relative to its own pulse and the Gaussian keeps a
        // fixed relative truncation at every width.
        const double t_p_i = tr.t_pulse;
        const double gauss_fwhm_i = 0.5*t_p_i;
        const int n_drive_i = static_cast<int>(
            std::llround(2.0*t_p_i/dt));
        const int snapshot_every_i =
            std::max(1, n_drive_i/20);

        // Seed: the T=0 relaxed state for the deterministic baseline,
        // else this member's cached thermal state. A missing seed is an
        // error; the study never silently falls back to a fresh IC.
        char seed_path[300];
        if (noise_free) {
            std::snprintf(seed_path, sizeof(seed_path),
                          "%s/m_eq_%dx%d.npz",
                          seed_dir.c_str(), nx, ny);
        } else {
            std::snprintf(seed_path, sizeof(seed_path),
                          "%s/m_thermal_T%05.1f_ens%03d.npz",
                          seed_dir.c_str(), tr.T_sub, tr.ens);
        }
        if (!std::filesystem::exists(seed_path)) {
            std::fprintf(stderr,
                "error: seed %s not found. For T_sub > 0 run the "
                "campaign's equilibrate stage; for T_sub = 0 its "
                "relax stage.\n", seed_path);
            return 1;
        }
        SAFPair seed = load_saf_npz(seed_path);

        StochasticConfig cfg;
        cfg.T_sub = tr.T_sub;
        cfg.R_th = r_th;
        // Peak amplitude; the profile itself carries the time
        // dependence.
        cfg.j_current = tr.peak_j;
        cfg.nx = nx; cfg.ny = ny; cfg.dt = dt;
        cfg.n_relax = 0;                 // pre-equilibrated seed
        cfg.n_drive = n_drive_i;
        cfg.sample_every = sample_every;
        // Seed stride keeps members and cells on disjoint noise
        // streams; the 2e9 offset separates this study from the
        // campaign's equilibrate (0) and drive (1e9) stages.
        cfg.seed = 2000000000LL + seed_base + 1000LL * tr.ens
                   + 1000000LL * tr.cell_idx;
        cfg.tol_norm = tol_norm;
        cfg.D = cs.dmi;
        cfg.K_top = cs.k_top;
        cfg.use_demag = true;
        cfg.demag_kind = demag_kind;
        cfg.demag_accuracy = demag_accuracy;
        cfg.demag_tol_conv = demag_tol_conv;
        cfg.q_threshold = q_threshold;
        cfg.k_consecutive = k_consecutive;
        cfg.equilibrate = false;         // seed is already equilibrated
        cfg.m_init_top = &seed.m_top;
        cfg.m_init_bot = &seed.m_bot;
        cfg.drive_pulse = make_pulse(sp.shape, tr.peak_j, t_p_i,
                                     gauss_fwhm_i);
        // Exactly one member per cell carries the field stream.
        const bool dump = (tr.ens == 0);
        cfg.dump_snapshots = dump;
        cfg.snapshot_every = dump ? snapshot_every_i : 0;
        cfg.max_snapshot_frames = n_drive_i/snapshot_every_i + 8;
        cfg.progress_every = 2000;
        cfg.fft_threads = fft_threads;
        // Translation/deformation split: the width study's headline
        // measurement, recorded for every member of that stage only.
        cfg.record_dissipation = width_sweep;

        std::ostringstream cfgjson;
        cfgjson.precision(17);
        cfgjson << "{\"shape\":\"" << sp.tag << "\""
                << ",\"T_sub\":" << cfg.T_sub
                << ",\"peak_j\":" << tr.peak_j
                << ",\"ens_idx\":" << tr.ens
                << ",\"t_pulse\":" << t_p_i
                << ",\"t_settle\":" << t_p_i
                << ",\"gauss_fwhm\":" << gauss_fwhm_i
                << ",\"nx\":" << nx << ",\"ny\":" << ny
                << ",\"dt\":" << dt
                << ",\"n_drive\":" << n_drive_i
                << ",\"noise_free\":" << (noise_free ? 1 : 0)
                << ",\"stage\":\"" << ps_stage << "\""
                << ",\"demag_kind\":\"racetrack\"}";

        std::unique_ptr<SnapshotBuffer> snaps;
        if (dump) {
            snaps.reset(new SnapshotBuffer(
                ny, nx, cfg.max_snapshot_frames));
        }
        StochasticPayload pl = run_trajectory(cfg, snaps.get());

        char fn[300];
        if (width_sweep) {
            std::snprintf(fn, sizeof(fn),
                          "%s/%s_tp%04.0f_T%05.1f_j%.2e_ens%03d.npz",
                          out_dir.c_str(), sp.tag, t_p_i*1e12,
                          tr.T_sub, tr.peak_j, tr.ens);
        } else {
            std::snprintf(fn, sizeof(fn),
                          "%s/%s_T%05.1f_j%.2e_ens%03d.npz",
                          out_dir.c_str(), sp.tag, tr.T_sub, tr.peak_j,
                          tr.ens);
        }
        save_trajectory(fn, pl, cfgjson.str());
        if (snaps) {
            char an[300];
            if (width_sweep) {
                std::snprintf(an, sizeof(an),
                              "%s/anim_%s_tp%04.0f_T%05.1f_j%.2e.npz",
                              out_dir.c_str(), sp.tag, t_p_i*1e12,
                              tr.T_sub, tr.peak_j);
            } else {
                std::snprintf(an, sizeof(an),
                              "%s/anim_%s_T%05.1f_j%.2e.npz",
                              out_dir.c_str(), sp.tag, tr.T_sub,
                              tr.peak_j);
            }
            snaps->write(an, pos_top, pos_bot, cfgjson.str());
        }
        std::printf(
            "  %-14s T_sub=%5.1f  peak_j=%.2e  ens=%03d  "
            "alive=%d  v=%.1f\n",
            sp.tag, tr.T_sub, tr.peak_j, tr.ens,
            pl.alive_at_end ? 1 : 0, pl.velocity);
        std::fflush(stdout);
    }
    return 0;
}
