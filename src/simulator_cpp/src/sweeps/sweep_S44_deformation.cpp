// Pham et al. (2024) Figure S44: skyrmion deformation under Gaussian
// drive. Panel A (J=8.9e11, fine 2 ps sampling) + panels BC (J sweep,
// 5 ps sampling). Port of scripts/sweep_S44_deformation.py.
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
    const double FWHM = 500.0e-12;
    const double tail_sigmas = 3.0;
    const double J_panel_A = 8.9e11;
    const std::vector<double> J_values_BC = {
        1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e11,
        6.0e11, 7.0e11, 8.0e11, 8.9e11};
    const double D = 0.85e-3;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double relax_time = 500.0e-12;
    const bool dump_snapshots = true;            // deformation visualization
    const double snapshot_dt = 12.5e-12;         // was 25.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S44";
    // -------------------------------------------------------------------------
    const double sigma = fwhm_to_sigma(FWHM);
    const double t_center = tail_sigmas * sigma;
    const double drive_time = 2.0 * tail_sigmas * sigma + 200.0e-12;
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int sample_every_A = static_cast<int>(std::ceil(2.0e-12 / dt));
    const int sample_every_BC = static_cast<int>(std::ceil(5.0e-12 / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    struct Point { bool is_A; double J0; int sample_every; };
    std::vector<Point> grid;
    grid.push_back({true, J_panel_A, sample_every_A});
    for (double J0 : J_values_BC) grid.push_back({false, J0, sample_every_BC});

    for (int idx : resolve_indices(static_cast<int>(grid.size()))) {
        const Point pt = grid[idx];
        Params p = make_default_params();
        p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
        if (use_full_demag) {
            p.demag_kind = demag_kind;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
        }
        precompute(p);

        char fn[160];
        if (pt.is_A)
            std::snprintf(fn, sizeof(fn), "%s/panelA_%s.npz",
                          out_dir.c_str(), d_tag(D).c_str());
        else
            std::snprintf(fn, sizeof(fn), "%s/J_%.2e_%s.npz",
                          out_dir.c_str(), pt.J0, d_tag(D).c_str());

        PointConfig cfg;
        cfg.p = &p;
        cfg.pulse = std::make_shared<GaussianPulse>(pt.J0, t_center, FWHM);
        cfg.use_demag = use_full_demag;
        cfg.convergence_relax = use_full_demag;
        cfg.n_drive = n_drive;
        cfg.sample_every = pt.sample_every;
        cfg.n_relax_fixed = n_relax_fixed;
        cfg.dump_snapshots = dump_snapshots;
        cfg.snapshot_every_drive = snapshot_every;
        cfg.trace_path = fn;
        cfg.snapshot_path = snapshot_path_of(fn);
        cfg.metadata.add("figure", std::string("S44"));
        cfg.metadata.add("panel", std::string(pt.is_A ? "A" : "BC"));
        cfg.metadata.add("demag", std::string(use_full_demag ? "full_fft" : "local_keff"));
        cfg.metadata.add("J0", pt.J0);
        cfg.metadata.add("FWHM", FWHM);
        cfg.metadata.add("sigma", sigma);
        cfg.metadata.add("t_center", t_center);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", pt.sample_every);

        std::printf("S44: panel %s, J0=%.2e -> %s\n",
                    pt.is_A ? "A" : "BC", pt.J0, fn);
        run_point(cfg);
    }
    return 0;
}
