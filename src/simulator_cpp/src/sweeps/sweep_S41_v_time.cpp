// Pham et al. (2024) Figure S41: single DC trajectory.
// J = 1e11 A/m^2 square pulse for 2 ns + 500 ps tail. Port of
// studies/saf_racetrack/scripts/reproduction_S41_S49/sweeps/sweep_S41_v_time.py. Run config = variables at top of main().
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>

using namespace skyrmion;
using namespace skyrmion::sweep;

int main() {
    // ----- Run configuration -------------------------------------------------
    const bool use_full_demag = true;
    const DemagKind demag_kind = DemagKind::Newell;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    const double J0 = 1.0e11;
    const double t_pulse = 2.0e-9;
    const double D = 0.85e-3;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double drive_time = t_pulse + 500.0e-12;
    const double sample_dt = 5.0e-12;
    const double relax_time = 500.0e-12;
    const bool dump_snapshots = true;            // showcase trajectory
    const double snapshot_dt = 25.0e-12;         // was 50.0e-12
    const double print_dt = 100.0e-12;           // progress-line interval
    const std::string out_dir = "output/sweeps_S41_S49/S41";
    // -------------------------------------------------------------------------
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));
    const int print_every = static_cast<int>(std::ceil(print_dt / dt));

    Params p = make_default_params();
    p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
    if (use_full_demag) {
        p.demag_kind = demag_kind;
        p.demag_accuracy = demag_accuracy;
        p.demag_tol_conv = demag_tol_conv;
    }
    precompute(p);

    const std::string tag = use_full_demag
        ? (demag_kind == DemagKind::Newell ? "newell" : "slab") : "keff";
    const std::string trace_path =
        out_dir + "/run_" + tag + "_" + d_tag(D) + ".npz";

    PointConfig cfg;
    cfg.p = &p;
    cfg.pulse = std::make_shared<SquarePulse>(J0, 0.0, t_pulse);
    cfg.use_demag = use_full_demag;
    cfg.convergence_relax = use_full_demag;
    cfg.n_drive = n_drive;
    cfg.sample_every = sample_every;
    cfg.n_relax_fixed = n_relax_fixed;
    cfg.dump_snapshots = dump_snapshots;
    cfg.snapshot_every_drive = snapshot_every;
    cfg.print_every = print_every;
    cfg.trace_path = trace_path;
    cfg.snapshot_path = snapshot_path_of(trace_path);
    cfg.metadata.add("figure", std::string("S41"));
    cfg.metadata.add("demag", std::string(use_full_demag ? "full_fft" : "local_keff"));
    cfg.metadata.add("J0", J0);
    cfg.metadata.add("t_pulse", t_pulse);
    cfg.metadata.add("D", D);
    cfg.metadata.add("nx", nx);
    cfg.metadata.add("ny", ny);
    cfg.metadata.add("dt", dt);
    cfg.metadata.add("n_drive", n_drive);
    cfg.metadata.add("sample_every", sample_every);

    std::printf("S41: J0=%.2e, t_pulse=%.1f ns, D=%.2f mJ/m^2, demag=%s\n",
                J0, t_pulse * 1e9, D * 1e3, tag.c_str());
    run_point(cfg);
    std::printf("  wrote %s\n", trace_path.c_str());
    return 0;
}
