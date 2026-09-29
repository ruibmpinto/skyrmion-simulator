// Skyrmion length-scale vs track-width (T_sub, j) scan -- STAGE 3 (drive).
// Production port of studies/saf_racetrack/experiments/scan_track_width.py.
//
// The thermal equilibrium at each temperature is current-independent, so
// it is computed once per (T_sub, ens) by stage 2 (equilibrate_track_width)
// and cached as m_thermal_T{T}_ens{ens}.npz. This stage loads the matching
// cached state for its trajectory and drives at the cell's current with
// newell demag -- no equilibration here (n_relax = 0). The else-branch of
// run_trajectory measures the loaded field's LCC ellipse as the pre-drive
// finite-T equilibrium size.
//
// SLURM array dispatch is per TRAJECTORY: 6 J x 6 T x 100 ens = 3600;
// SLURM_ARRAY_TASK_ID picks one. The ens == 0 member of each cell dumps
// the field-snapshot stream for animation.
#include "skyrmion/lattice.hpp"
#include "skyrmion/sweep/sweep_common.hpp"   // resolve_indices
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <sstream>
#include <memory>
#include <string>
#include <utility>
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
}  // namespace

int main() {
    // ----- Run configuration (box must match stages 1-2) ---------------------
    // Phase-B campaign cases; select one by the TW_CASE env var.
    // Must match equilibrate_track_width's case table (same order).
    struct Case { double k_top; double dmi; int nx; int ny; double a;
                  const char* tag; };
    const std::vector<Case> cases = {
        {1.294e6,  0.58e-3, 350, 500, 2.0e-9, "hk12p4_D0p58_350x500"},
        {1.294e6,  0.58e-3, 700, 500, 2.0e-9, "hk12p4_D0p58_700x500"},
        {1.3106e6, 0.72e-3, 350, 500, 2.0e-9, "hk36_D0p72_350x500"},
        {1.3106e6, 0.72e-3, 700, 500, 2.0e-9, "hk36_D0p72_700x500"},
        {1.3106e6, 0.72e-3, 1400, 500, 2.0e-9, "hk36_D0p72_1400x500"},
        {1.294e6,  0.58e-3, 1400, 500, 2.0e-9, "hk12p4_D0p58_1400x500"},
        {1.3106e6, 0.72e-3, 467, 333, 3.0e-9,
         "hk36_D0p72_467x333_a3nm"},
    };
    const char* tw_case_env = std::getenv("TW_CASE");
    if (tw_case_env == nullptr || *tw_case_env == '\0') {
        std::fprintf(stderr,
            "error: TW_CASE unset; expected 0..%zu.\n",
            cases.size() - 1);
        return 1;
    }
    const int case_idx = std::atoi(tw_case_env);
    if (case_idx < 0 || case_idx >= static_cast<int>(cases.size())) {
        std::fprintf(stderr,
            "error: TW_CASE=%d out of range [0, %zu].\n",
            case_idx, cases.size() - 1);
        return 1;
    }
    const Case tw = cases[case_idx];
    // Drive mode: dc (default) = constant 2 ns DC; stopgo = three 1 ns
    // square pulses at 0/2/4 ns (on/off/on/off/on = 5 ns), seeds reused
    // from the campaign, output written to a separate stopgo/ dir.
    const char* drive_env = std::getenv("TW_DRIVE");
    const std::string drive_mode =
        (drive_env && *drive_env) ? drive_env : "dc";
    if (drive_mode != "dc" && drive_mode != "stopgo") {
        std::fprintf(stderr,
            "error: TW_DRIVE=%s unknown (expected dc|stopgo).\n",
            drive_mode.c_str());
        return 1;
    }
    const bool stopgo = (drive_mode == "stopgo");
    // Transition-temperature study: DC drive at a single current over
    // fine intermediate temperatures, seeds reused from the campaign,
    // output to a separate ttrans/ dir.
    const char* tscan_env = std::getenv("TW_TSCAN");
    const bool tscan = (tscan_env != nullptr && *tscan_env != '\0');
    if (tscan && stopgo) {
        std::fprintf(stderr,
            "error: TW_TSCAN and TW_DRIVE=stopgo are exclusive.\n");
        return 1;
    }
    const int nx = tw.nx, ny = tw.ny;
    const double k_top = tw.k_top;
    // Lattice constant for the dump grids, from the case (the
    // trajectories run at this a via cfg.a below).
    const double cell_a = tw.a;
    const std::vector<double> j_list = tscan
        ? std::vector<double>{3.0e11}
        : std::vector<double>{0.5e11, 1.0e11, 2.0e11, 3.0e11, 4.0e11,
                              5.0e11};
    const std::vector<double> t_sub_list = tscan
        ? std::vector<double>{105.0, 110.0, 115.0, 120.0, 125.0}
        : std::vector<double>{10.0, 50.0, 100.0, 130.0, 160.0, 200.0};
    const double r_th = 0.0;
    const double dt = 5.0e-14;
    const int n_drive = stopgo ? 100000 : 40000;  // 5 ns : 2 ns
    const int sample_every = 200;    // 10 ps
    const int snapshot_every = 400;  // 20 ps anim frames
    const int n_ens = 100;
    const long long seed_base = 101;
    const double tol_norm = 5.0e-3;
    const double q_threshold = 0.5;
    const int k_consecutive = 10;
    // Racetrack: periodic x, free top/bottom (y) demag (Racetrack)
    // over the full box; track width = transverse box extent L_y = ny*a.
    // Exchange/DMI use free-y ghost cells (mask = nullptr), matching
    // the Python pipeline. D must match stages 1 & 2.
    const DemagKind demag_kind = DemagKind::Racetrack;
    const double dmi = tw.dmi;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    // Thermal seeds are always read from the campaign dir (stage 2).
    // In stopgo mode the driven trajectories are written to a separate
    // stopgo/ dir so they never overwrite the 2 ns DC campaign.
    const std::string campaign_dir =
        std::string("output/stochastic_llgs/scan_track_width/campaign/")
        + tw.tag;
    const std::string out_dir = stopgo
        ? std::string("output/stochastic_llgs/scan_track_width/stopgo/")
          + tw.tag
        : tscan ? (campaign_dir + "/ttrans")
                : campaign_dir;
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    const int fft_threads = fft_threads_from_env();

    Field3 pos_top = lattice_positions(nx, ny, cell_a);
    Field3 pos_bot = lattice_positions(nx, ny, cell_a);

    // ----- Flat per-trajectory grid: 24 cells x n_ens ------------------------
    struct Traj { double T_sub; double j; int cell_idx; int ens; };
    std::vector<Traj> grid;
    int cell_idx = 0;
    for (double T_sub : t_sub_list) {
        for (double j : j_list) {
            for (int ens = 0; ens < n_ens; ++ens)
                grid.push_back({T_sub, j, cell_idx, ens});
            ++cell_idx;
        }
    }

    for (int idx : resolve_indices(static_cast<int>(grid.size()))) {
        const Traj tr = grid[idx];
        // Load the shared per-(T, ens) thermal state (stage 2 output).
        char thp[200];
        std::snprintf(thp, sizeof(thp),
                      "%s/m_thermal_T%05.1f_ens%03d.npz",
                      campaign_dir.c_str(), tr.T_sub, tr.ens);
        if (!std::filesystem::exists(thp)) {
            std::fprintf(stderr,
                "error: %s not found. Run equilibrate_track_width "
                "first.\n", thp);
            return 1;
        }
        SAFPair thermal = load_saf_npz(thp);

        StochasticConfig cfg;
        cfg.T_sub = tr.T_sub;
        cfg.R_th = r_th;
        cfg.j_current = tr.j;
        if (stopgo) {
            // Three 1 ns square pulses at 0/2/4 ns: on/off/on/off/on
            // over 5 ns, ending while driving.
            std::vector<std::unique_ptr<Pulse>> squares;
            squares.push_back(std::make_unique<SquarePulse>(
                cfg.j_current, 0.0, 1.0e-9));
            squares.push_back(std::make_unique<SquarePulse>(
                cfg.j_current, 2.0e-9, 3.0e-9));
            squares.push_back(std::make_unique<SquarePulse>(
                cfg.j_current, 4.0e-9, 5.0e-9));
            cfg.drive_pulse = std::make_shared<SuperpositionPulse>(
                std::move(squares));
        } else {
            // DC drive, stated explicitly: there is no implied fallback.
            cfg.drive_pulse =
                std::make_shared<ConstantPulse>(cfg.j_current);
        }
        cfg.nx = nx; cfg.ny = ny; cfg.dt = dt;
        cfg.n_relax = 0;                 // pre-thermalized; no relax here
        cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        // Decorrelated seed stride per ensemble member and cell.
        // Stage offset 1e9 keeps the drive streams disjoint from the
        // equilibrate-stage seeds (cell_idx overlaps t_idx there).
        cfg.seed = 1000000000LL + seed_base + 1000LL * tr.ens
                   + 1000000LL * tr.cell_idx;
        cfg.tol_norm = tol_norm;
        cfg.D = dmi;
        cfg.K_top = k_top;
        cfg.a = tw.a;
        cfg.use_demag = true;
        cfg.demag_kind = demag_kind;
        cfg.demag_accuracy = demag_accuracy;
        cfg.demag_tol_conv = demag_tol_conv;
        cfg.q_threshold = q_threshold;
        cfg.k_consecutive = k_consecutive;
        cfg.equilibrate = false;         // seed from cached thermal state
        cfg.m_init_top = &thermal.m_top;
        cfg.m_init_bot = &thermal.m_bot;
        // Only the first ensemble member dumps the field stream.
        const bool dump = (tr.ens == 0);
        cfg.dump_snapshots = dump;
        cfg.snapshot_every = dump ? snapshot_every : 0;
        cfg.max_snapshot_frames = n_drive / snapshot_every + 8;
        // Timestamped drive progress every 2000 steps (100 ps).
        cfg.progress_every = 2000;
        cfg.fft_threads = fft_threads;

        std::ostringstream cfgjson;
        cfgjson.precision(17);
        cfgjson << "{\"T_sub\":" << cfg.T_sub
                << ",\"j_current\":" << cfg.j_current
                << ",\"ens_idx\":" << tr.ens
                << ",\"nx\":" << nx << ",\"ny\":" << ny
                << ",\"dt\":" << dt
                << ",\"n_drive\":" << n_drive
                << ",\"demag_kind\":\"racetrack\"}";

        std::unique_ptr<SnapshotBuffer> snaps;
        if (dump) {
            snaps.reset(new SnapshotBuffer(
                ny, nx, cfg.max_snapshot_frames));
        }
        StochasticPayload pl = run_trajectory(cfg, snaps.get());

        char fn[200];
        std::snprintf(fn, sizeof(fn),
                      "%s/T%05.1f_j%.2e_ens%03d.npz",
                      out_dir.c_str(), tr.T_sub, tr.j, tr.ens);
        save_trajectory(fn, pl, cfgjson.str());
        if (snaps) {
            char an[200];
            std::snprintf(an, sizeof(an),
                          "%s/anim_T%05.1f_j%.2e.npz",
                          out_dir.c_str(), tr.T_sub, tr.j);
            snaps->write(an, pos_top, pos_bot, cfgjson.str());
        }
        std::printf(
            "  T_sub=%5.1f  j=%.2e  ens=%03d  T_eff=%5.1f  "
            "alive=%d  v=%.1f\n",
            tr.T_sub, tr.j, tr.ens, pl.T_effective,
            pl.alive_at_end ? 1 : 0, pl.velocity);
        std::fflush(stdout);
    }
    return 0;
}
