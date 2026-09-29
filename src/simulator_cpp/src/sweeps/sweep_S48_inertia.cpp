// Pham et al. (2024) Figure S48: DC square pulse, H_RKKY sweep,
// Set B material constants. Port of studies/saf_racetrack/scripts/sweep_S48_inertia.py.
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;

int main() {
    // ----- Run configuration -------------------------------------------------
    const bool use_full_demag = true;
    const DemagKind demag_kind = DemagKind::Newell;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    const std::vector<double> H_RKKY_values = {
        0.100, 0.150, 0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500};
    const double J0 = 1.0e11;
    const double t_pulse = 2.0e-9;
    const double alpha = 0.216;     // Set B
    const double gamma = 175.9e9;   // Set B
    const double D = 0.62e-3;       // Set B
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double relax_time = 500.0e-12;
    const double sample_dt = 1.0e-12;
    const bool dump_snapshots = false;
    const double snapshot_dt = 25.0e-12;  // was 50.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S48";
    // -------------------------------------------------------------------------
    const double drive_time = t_pulse + 1.0e-9;
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    for (int idx : resolve_indices(static_cast<int>(H_RKKY_values.size()))) {
        const double H_RKKY = H_RKKY_values[idx];
        Params p = make_default_params();
        p.D = D; p.alpha = alpha; p.gamma_ = gamma;
        p.H_RKKY = H_RKKY;
        p.nx = nx; p.ny = ny; p.dt = dt;
        if (use_full_demag) {
            p.demag_kind = demag_kind;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
        }
        precompute(p);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/HRKKY_%.0fmT_%s.npz",
                      out_dir.c_str(), H_RKKY * 1000.0, d_tag(D).c_str());

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
        cfg.trace_path = fn;
        cfg.snapshot_path = snapshot_path_of(fn);
        cfg.metadata.add("figure", std::string("S48"));
        cfg.metadata.add("demag", std::string(use_full_demag ? "full_fft" : "local_keff"));
        cfg.metadata.add("J0", J0);
        cfg.metadata.add("t_pulse", t_pulse);
        cfg.metadata.add("H_RKKY", H_RKKY);
        cfg.metadata.add("alpha", alpha);
        cfg.metadata.add("gamma", gamma);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", sample_every);

        std::printf("S48: H_RKKY=%.0f mT -> %s\n", H_RKKY * 1000.0, fn);
        run_point(cfg);
    }
    return 0;
}
