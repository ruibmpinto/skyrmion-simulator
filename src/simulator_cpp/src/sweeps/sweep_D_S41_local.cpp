// S41 D-sweep, local-K_eff field model (no FFT demag). Port of
// scripts/sweep_D_S41_local.py. Same protocol as sweep_D_S41 with the
// field model pinned to K_eff.
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;

int main() {
    // ----- Run configuration -------------------------------------------------
    const std::vector<double> D_values = {
        0.62e-3, 0.72e-3, 0.80e-3, 0.85e-3, 0.90e-3, 0.95e-3, 1.00e-3};
    const double J0 = 1.0e11;
    const double t_pulse = 2.0e-9;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double drive_time = t_pulse + 500.0e-12;
    const double sample_dt = 5.0e-12;
    const bool dump_snapshots = false;
    const double snapshot_dt = 25.0e-12;  // was 50.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S41_D_sweep";
    // -------------------------------------------------------------------------
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    for (int idx : resolve_indices(static_cast<int>(D_values.size()))) {
        const double D = D_values[idx];
        Params p = make_default_params();
        p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
        precompute(p);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/D_%.3fmJm2_keff.npz",
                      out_dir.c_str(), D * 1e3);

        PointConfig cfg;
        cfg.p = &p;
        cfg.pulse = std::make_shared<SquarePulse>(J0, 0.0, t_pulse);
        cfg.use_demag = false;
        cfg.convergence_relax = true;
        cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.dump_snapshots = dump_snapshots;
        cfg.snapshot_every_drive = snapshot_every;
        cfg.trace_path = fn;
        cfg.snapshot_path = snapshot_path_of(fn);
        cfg.metadata.add("figure", std::string("S41_D_sweep"));
        cfg.metadata.add("demag", std::string("local_keff"));
        cfg.metadata.add("field_kind", std::string("keff"));
        cfg.metadata.add("J0", J0);
        cfg.metadata.add("t_pulse", t_pulse);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", sample_every);

        std::printf("D-sweep (local): D=%.3f mJ/m^2 -> %s\n", D * 1e3, fn);
        run_point(cfg);
    }
    return 0;
}
