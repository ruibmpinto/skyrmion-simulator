// Pham et al. (2024) Figure S46b: Gaussian pulse, FWHM sweep at fixed
// H_RKKY. Port of scripts/sweep_S46b_FWHM.py.
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
    const std::vector<double> FWHM_values = {
        100.0e-12, 150.0e-12, 200.0e-12, 250.0e-12,
        300.0e-12, 350.0e-12, 400.0e-12, 500.0e-12};
    const double J0 = 8.9e11;
    const double H_RKKY = 0.205;
    const double tail_sigmas = 3.0;
    const double D = 0.85e-3;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double relax_time = 500.0e-12;
    const double sample_dt = 5.0e-12;
    const bool dump_snapshots = false;
    const double snapshot_dt = 25.0e-12;  // was 50.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S46b";
    // -------------------------------------------------------------------------
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    for (int idx : resolve_indices(static_cast<int>(FWHM_values.size()))) {
        const double FWHM = FWHM_values[idx];
        const double sigma = fwhm_to_sigma(FWHM);
        const double t_center = tail_sigmas * sigma;
        const double drive_time = 2.0 * tail_sigmas * sigma + 200.0e-12;
        const int n_drive = static_cast<int>(std::ceil(drive_time / dt));

        Params p = make_default_params();
        p.D = D; p.nx = nx; p.ny = ny; p.dt = dt;
        p.H_RKKY = H_RKKY;
        if (use_full_demag) {
            p.demag_kind = demag_kind;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
        }
        precompute(p);

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/FWHM_%.0fps_%s.npz",
                      out_dir.c_str(), FWHM * 1e12, d_tag(D).c_str());

        PointConfig cfg;
        cfg.p = &p;
        cfg.pulse = std::make_shared<GaussianPulse>(J0, t_center, FWHM);
        cfg.use_demag = use_full_demag;
        cfg.convergence_relax = use_full_demag;
        cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.n_relax_fixed = n_relax_fixed;
        cfg.dump_snapshots = dump_snapshots;
        cfg.snapshot_every_drive = snapshot_every;
        cfg.trace_path = fn;
        cfg.snapshot_path = snapshot_path_of(fn);
        cfg.metadata.add("figure", std::string("S46b"));
        cfg.metadata.add("demag", std::string(use_full_demag ? "full_fft" : "local_keff"));
        cfg.metadata.add("J0", J0);
        cfg.metadata.add("H_RKKY", H_RKKY);
        cfg.metadata.add("FWHM", FWHM);
        cfg.metadata.add("sigma", sigma);
        cfg.metadata.add("t_center", t_center);
        cfg.metadata.add("D", D);
        cfg.metadata.add("nx", nx);
        cfg.metadata.add("ny", ny);
        cfg.metadata.add("dt", dt);
        cfg.metadata.add("n_drive", n_drive);
        cfg.metadata.add("sample_every", sample_every);

        std::printf("S46b: FWHM=%.0f ps -> %s\n", FWHM * 1e12, fn);
        run_point(cfg);
    }
    return 0;
}
