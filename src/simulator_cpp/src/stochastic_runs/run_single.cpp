// Single-trajectory stochastic-LLGS production runner. Port of
// studies/saf_racetrack/scripts/.../production/run_single.py::main. Run config = named
// variables at the top of main(); no argparse. Optional snapshot dump
// for animating the thermal trajectory.
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"

#include <cstdio>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>

using namespace skyrmion;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration --------------------------------------------------
    StochasticConfig cfg;
    cfg.T_sub = 300.0;
    cfg.R_th = 0.0;
    cfg.j_current = 4.0e11;
    // DC drive, stated explicitly: there is no implied fallback.
    cfg.drive_pulse = std::make_shared<ConstantPulse>(cfg.j_current);
    cfg.nx = 256; cfg.ny = 256;
    cfg.dt = 5.0e-14;
    cfg.n_relax = 10000;
    cfg.n_drive = 20000;
    cfg.sample_every = 100;
    cfg.seed = 17;
    cfg.tol_norm = 5.0e-3;
    cfg.use_demag = false;
    cfg.q_threshold = 0.5;
    cfg.k_consecutive = 10;
    cfg.dump_snapshots = false;      // set true to animate the trajectory
    cfg.snapshot_every = 200;        // ~10 ps cadence
    cfg.max_snapshot_frames = 500;
    const std::string out_dir = "output/stochastic_llgs";
    const std::string out_npz = "run_single.npz";
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    const std::string trace_path = out_dir + "/" + out_npz;

    std::ostringstream cfgjson;
    cfgjson.precision(17);
    cfgjson << "{\"T_sub\":" << cfg.T_sub << ",\"R_th\":" << cfg.R_th
            << ",\"j_current\":" << cfg.j_current
            << ",\"nx\":" << cfg.nx << ",\"ny\":" << cfg.ny
            << ",\"dt\":" << cfg.dt << ",\"n_relax\":" << cfg.n_relax
            << ",\"n_drive\":" << cfg.n_drive
            << ",\"sample_every\":" << cfg.sample_every
            << ",\"seed\":" << cfg.seed << ",\"tol_norm\":" << cfg.tol_norm
            << ",\"use_demag\":" << (cfg.use_demag ? "true" : "false")
            << ",\"q_threshold\":" << cfg.q_threshold
            << ",\"k_consecutive\":" << cfg.k_consecutive << "}";

    std::unique_ptr<SnapshotBuffer> snaps;
    if (cfg.dump_snapshots) {
        snaps.reset(new SnapshotBuffer(cfg.ny, cfg.nx, cfg.max_snapshot_frames));
    }

    std::printf("run_single: stochastic SAF skyrmion trajectory\n");
    std::printf("  T_sub=%.1f K  j=%.2e A/m^2  %dx%d  demag=%s\n",
                cfg.T_sub, cfg.j_current, cfg.nx, cfg.ny,
                cfg.use_demag ? "true" : "false");

    StochasticPayload pl = run_trajectory(cfg, snaps.get());
    save_trajectory(trace_path, pl, cfgjson.str());

    if (snaps) {
        Field3 pos_top = lattice_positions(cfg.nx, cfg.ny, 2.0e-9);
        Field3 pos_bot = lattice_positions(cfg.nx, cfg.ny, 2.0e-9);
        snaps->write(out_dir + "/" + "run_single_snapshots.npz",
                     pos_top, pos_bot, cfgjson.str());
    }

    std::printf("  T_eff=%.1f K  sigma=%.3e  alive=%d  flip_index=%d\n",
                pl.T_effective, pl.sigma_noise, pl.alive_at_end ? 1 : 0,
                pl.flip_index);
    std::printf("  v=%.1f m/s (vx=%.1f, vy=%.1f)  theta_H=%.2f deg\n",
                pl.velocity, pl.v_x, pl.v_y, pl.hall_deg);
    std::printf("  wrote %s\n", trace_path.c_str());
    return 0;
}
