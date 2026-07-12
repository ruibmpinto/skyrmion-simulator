// Coarse DMI sweep for a stable, larger racetrack skyrmion. For each D
// in the paper's band and two box sizes (256^2, 350^2) it relaxes the
// Set-A SAF skyrmion under the racetrack boundary (periodic x, free y
// for BOTH demag and exchange/DMI; DemagKind::Racetrack, mask = nullptr)
// via the shared relax_torque_probe, producing the same per-box outputs
// as validate_relax_torque (convergence-series NPZ + snapshot + table
// row), under a per-D subdirectory. The equilibrium size grows as
// D -> D_c; box-independence (D1 matching across the two boxes) marks a
// genuine finite equilibrium, while D1 >> D2 growing with box marks the
// stripe (D > D_c).
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/relax_torque_probe.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

int fft_threads_from_env() {
    if (const char* e = std::getenv("OMP_NUM_THREADS")) {
        if (*e && std::atoi(e) > 0) return std::atoi(e);
    }
    return 1;
}

}  // namespace

int main() {
    // ----- Run configuration -------------------------------------------------
    struct Box { int nx; int ny; };
    const std::vector<Box> boxes = {{256, 256}, {350, 350}};
    const std::vector<double> d_list = {
        0.47e-3, 0.52e-3, 0.57e-3, 0.62e-3, 0.66e-3};
    const double dt = 5.0e-14;
    const double alpha_relax = 1.0;
    const int chunk = 2000;          // steps between samples
    const int n_chunks = 125;        // 250000 steps, no early stop
    // Seed at the periodic equilibrium diameter (radius 103 nm = 206 nm);
    // relaxation finds each D's equilibrium from there.
    const double skyrmion_radius = 103.0e-9;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    const std::string root =
        "output/stochastic_llgs/validation/dmi_sweep_racetrack";
    // -------------------------------------------------------------------------
    const int fft_threads = fft_threads_from_env();

    std::printf("DMI sweep, racetrack BC (free-y demag + free-y "
                "exchange/DMI), %d chunks x %d steps, threads=%d\n",
                n_chunks, chunk, fft_threads);
    relax_torque_probe_header();

    for (double dmi : d_list) {
        const std::string out_dir = root + "/" + dmi_dir_tag(dmi);
        std::filesystem::create_directories(out_dir);
        ProbeResult pr[2];
        for (int bi = 0; bi < static_cast<int>(boxes.size()); ++bi) {
            const Box b = boxes[bi];
            Params p = make_default_params();
            p.nx = b.nx; p.ny = b.ny; p.dt = dt;
            p.D = dmi;
            p.skyrmion_R = skyrmion_radius;
            p.demag_kind = DemagKind::Racetrack;
            p.demag_accuracy = demag_accuracy;
            p.demag_tol_conv = demag_tol_conv;
            p.pulse = std::make_shared<ConstantPulse>(0.0);
            precompute(p);
            DemagState demag(p, fft_threads);
            pr[bi] = relax_torque_probe(p, demag, /*mask=*/nullptr,
                                        n_chunks, chunk, alpha_relax,
                                        out_dir);
        }
        // Box-independence: relative difference of D1 across the boxes.
        const double rel = (pr[1].D1 > 0.0)
            ? std::abs(pr[0].D1 - pr[1].D1) / pr[1].D1 : 0.0;
        std::printf("   -> D=%.2f  D1(256)=%.1f D1(350)=%.1f nm  "
                    "|dD1|/D1=%.3f  (%s)\n",
                    dmi * 1e3, pr[0].D1, pr[1].D1, rel,
                    rel < 0.05 ? "stable/compact" : "box-dependent");
        std::fflush(stdout);
    }
    std::printf("wrote per-(D,box) NPZ under %s\n", root.c_str());
    return 0;
}
