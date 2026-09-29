// Pham et al. (2024) Figure S46a: Gaussian pulse, H_RKKY sweep.
// Port of studies/saf_racetrack/scripts/sweep_S46a_HRKKY.py.
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
        0.200, 0.250, 0.300, 0.350, 0.400, 0.450, 0.500, 0.550,
        0.600, 0.650, 0.700, 0.750, 0.800, 0.850, 0.900, 0.950};
    const double J0 = 8.9e11;
    const double FWHM = 500.0e-12;
    const double tail_sigmas = 3.0;
    const double D = 0.85e-3;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const double relax_time = 500.0e-12;
    const double sample_dt = 5.0e-12;
    const bool dump_snapshots = false;
    const double snapshot_dt = 25.0e-12;  // was 50.0e-12
    const std::string out_dir = "output/sweeps_S41_S49/S46a";
    // -------------------------------------------------------------------------
    const double sigma = fwhm_to_sigma(FWHM);
    const double t_center = tail_sigmas * sigma;
    const double drive_time = 2.0 * tail_sigmas * sigma + 200.0e-12;
    const int n_drive = static_cast<int>(std::ceil(drive_time / dt));
    const int sample_every = static_cast<int>(std::ceil(sample_dt / dt));
    const int n_relax_fixed = static_cast<int>(std::ceil(relax_time / dt));
    const int snapshot_every = static_cast<int>(std::ceil(snapshot_dt / dt));

    for (int idx : resolve_indices(static_cast<int>(H_RKKY_values.size()))) {
        const double H_RKKY = H_RKKY_values[idx];
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
        std::snprintf(fn, sizeof(fn), "%s/HRKKY_%.0fmT_%s.npz",
                      out_dir.c_str(), H_RKKY * 1000.0, d_tag(D).c_str());

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
        cfg.metadata.add("figure", std::string("S46a"));
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

        std::printf("S46a: H_RKKY=%.0f mT -> %s\n", H_RKKY * 1000.0, fn);
        run_point(cfg);
    }
    return 0;
}
