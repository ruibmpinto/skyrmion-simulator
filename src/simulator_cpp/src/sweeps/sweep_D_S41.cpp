// S41 trajectory swept over the DMI constant D (local-K_eff field
// model by default). Port of scripts/sweep_D_S41.py. SLURM array
// dispatch picks one D; full serial loop when unset.
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;

int main() {
    // ----- Run configuration -------------------------------------------------
    // 'keff' uses the local-K_eff path (no FFT demag). Set use_demag to
    // true + a DemagKind to switch to the explicit-demag field model.
    const bool use_demag = false;
    const DemagKind demag_kind = DemagKind::Newell;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
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
    const char* tag = use_demag
        ? (demag_kind == DemagKind::Newell ? "newell" : "slab") : "keff";

    for (int idx : resolve_indices(static_cast<int>(D_values.size()))) {
        const double D = D_values[idx];
        Params p = make_default_params();
        p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
        if (use_demag) {
            p.demag_kind = demag_kind;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
        }
        precompute(p);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/D_%.3fmJm2_%s.npz",
                      out_dir.c_str(), D * 1e3, tag);

        PointConfig cfg;
        cfg.p = &p;
        cfg.pulse = std::make_shared<SquarePulse>(J0, 0.0, t_pulse);
        cfg.use_demag = use_demag;
        cfg.convergence_relax = true;     // convergence-stop for both paths
        cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.dump_snapshots = dump_snapshots;
        cfg.snapshot_every_drive = snapshot_every;
        cfg.trace_path = fn;
        cfg.snapshot_path = snapshot_path_of(fn);
        cfg.metadata.add("figure", std::string("S41_D_sweep"));
        cfg.metadata.add("demag", std::string(use_demag ? "full_fft" : "local_keff"));
        cfg.metadata.add("field_kind", std::string(tag));
        cfg.metadata.add("J0", J0);
        cfg.metadata.add("t_pulse", t_pulse);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", sample_every);

        std::printf("D-sweep: D=%.3f mJ/m^2 (%s) -> %s\n",
                    D * 1e3, tag, fn);
        run_point(cfg);
    }
    return 0;
}
