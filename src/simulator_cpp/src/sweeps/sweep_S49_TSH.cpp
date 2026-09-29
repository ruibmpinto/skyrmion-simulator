// Pham et al. (2024) Figure S49: numerical LLGS verification of the
// topological spin Hall torque on a (R/Delta) x lambda_sq grid, Set B
// constants. Port of the LLGS part of studies/saf_racetrack/scripts/sweep_S49_TSH.py; the
// analytic Thiele curves stay in Python.
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
    const double Delta = 24.5e-9;
    const std::vector<double> R_over_Delta = {
        1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0};
    const std::vector<double> lambda_sq_values = {0.0, 3.0e-18, 50.0e-18};
    const double J0 = 8.0e11;
    const double t_pulse = 2.0e-9;
    const double alpha = 0.216;     // Set B
    const double gamma = 175.9e9;   // Set B
    const double D = 0.62e-3;       // Set B
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double relax_time = 500.0e-12;
    const double sample_dt = 5.0e-12;
    const bool dump_snapshots = false;
    const double snapshot_dt = 25.0e-12;  // was 50.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S49";
    // -------------------------------------------------------------------------
    const double drive_time = t_pulse + 500.0e-12;
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    struct Point { double rd; double lam; };
    std::vector<Point> grid;
    for (double rd : R_over_Delta)
        for (double lam : lambda_sq_values)
            grid.push_back({rd, lam});

    for (int idx : resolve_indices(static_cast<int>(grid.size()))) {
        const double rd = grid[idx].rd;
        const double lam = grid[idx].lam;
        const double R = rd * Delta;

        Params p = make_default_params();
        p.D = D; p.alpha = alpha; p.gamma_ = gamma;
        p.nx = nx; p.ny = ny; p.dt = dt;
        p.lambda_sq = lam;
        p.skyrmion_R = R; p.skyrmion_dw = Delta;
        if (use_full_demag) {
            p.demag_kind = demag_kind;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
        }
        precompute(p);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/llgs_R%.1f_lam%.0fnm2_%s.npz",
                      out_dir.c_str(), rd, lam * 1e18, d_tag(D).c_str());

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
        cfg.metadata.add("figure", std::string("S49_llgs"));
        cfg.metadata.add("demag", std::string(use_full_demag ? "full_fft" : "local_keff"));
        cfg.metadata.add("J0", J0);
        cfg.metadata.add("t_pulse", t_pulse);
        cfg.metadata.add("R", R);
        cfg.metadata.add("Delta", Delta);
        cfg.metadata.add("R_over_Delta", rd);
        cfg.metadata.add("lambda_sq_m2", lam);
        cfg.metadata.add("lambda_sq_nm2", lam * 1e18);
        cfg.metadata.add("alpha", alpha);
        cfg.metadata.add("gamma", gamma);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", sample_every);

        std::printf("S49: R/Delta=%.1f, lambda_sq=%.0f nm^2 -> %s\n",
                    rd, lam * 1e18, fn);
        run_point(cfg);
    }
    return 0;
}
