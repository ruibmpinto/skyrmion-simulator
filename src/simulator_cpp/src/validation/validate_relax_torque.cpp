// Relaxation-torque diagnostic / validator. Tests the claim that the
// elevated relax tau_max floor on a larger box is a max-norm artifact of
// a well-relaxed skyrmion (residual torque delocalized in the
// ferromagnetic background, bulk well-relaxed, size/energy converged,
// max-of-N growing with lattice size) rather than a genuinely
// under-relaxed skyrmion (torque concentrated at the domain wall, size /
// energy still drifting).
//
// For each box size it chunk-relaxes the Set-A SAF skyrmion under newell
// demag (calling relax() in fixed-step chunks, never early-stopping) and
// after each chunk records: tangential-torque percentiles (max, p99,
// median, rms), the LCC ellipse D1/D2, the total energy, and the max-
// torque site's distance from the skyrmion core. The final per-site
// torque field and m_z are dumped for the spatial map/histogram. One NPZ
// per box; a summary table is printed.
//
// Checks (see scripts/validate_relax_torque.py for the plots):
//   1. torque map/histogram        2. size/energy convergence vs step
//   3. box-size scaling of the floor   4. 256x256 positive control
#include "skyrmion/lattice.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/relax_torque_probe.hpp"

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

int main(int argc, char** argv) {
    // ----- Run configuration -------------------------------------------------
    struct Box { int nx; int ny; };
    const std::vector<Box> boxes = {
        {350, 500}, {256, 256}, {350, 350}, {500, 500}};
    const double dt = 5.0e-14;
    const double alpha_relax = 1.0;
    const int chunk = 2000;          // steps between samples
    // Total = 250000 steps: the 256² size keeps drifting (211->186 nm)
    // and tau keeps falling (7e-5->3e-5) well past 40k, so a short run
    // does not capture the converged plateau.
    const int n_chunks = 125;
    // DMI chosen from the coarse racetrack sweep to hit the 180-200 nm
    // target: 0.52 -> 142 nm, 0.57 -> 205 nm (box-independent, compact),
    // so 0.545 interpolates to ~185 nm. Still below the effective D_c
    // (~0.5-0.8), so the skyrmion stays a stable finite-radius state.
    const double dmi = 0.545e-3;
    // Seed at the periodic equilibrium: profile radius R = 103 nm gives
    // diameter 2R = 206 nm, so any change is the response to the
    // racetrack confinement, not to a mismatched initial size.
    const double skyrmion_radius = 103.0e-9;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    // Racetrack width (physical, along the free y direction), used only
    // for racetrack. The magnetic strip spans the full periodic x
    // and a centred band of this height in y; vacuum margins above and
    // below carry the Rohart-Thiaville free edge (xia = D/2A). Fixed
    // across boxes so the confinement-set size is separated from box
    // size; the margin grows with the box. Must leave >=1 vacuum row,
    // i.e. fit inside the smallest box (256 cells x 2 nm = 512 nm).
    const double track_width_m = 400e-9;
    // Boundary condition is an explicit required flag so periodic and
    // free-BC runs never overwrite each other (each writes its own
    // subdirectory). No silent default.
    if (argc != 2) {
        std::fprintf(stderr,
            "usage: %s <newell|newell_freebc|racetrack|masked_band>\n"
            "  newell        periodic (circular-convolution) demag\n"
            "  newell_freebc isolated (zero-padded 2N) demag\n"
            "  racetrack     periodic x, free top/bottom (y) for BOTH\n"
            "                demag and exchange/DMI over the FULL box\n"
            "                (track width = box, no mask)\n"
            "  masked_band   racetrack demag + a centred width-W magnetic\n"
            "                band (free edge at the band, vacuum margins)\n",
            argv[0]);
        return 1;
    }
    const std::string bc = argv[1];
    DemagKind demag_kind;
    if (bc == "newell") demag_kind = DemagKind::Newell;
    else if (bc == "newell_freebc") demag_kind = DemagKind::NewellFreeBC;
    else if (bc == "racetrack") demag_kind = DemagKind::Racetrack;
    else if (bc == "masked_band") demag_kind = DemagKind::Racetrack;
    else {
        std::fprintf(stderr,
            "validate_relax_torque: unknown boundary %s; expected "
            "'newell', 'newell_freebc', 'racetrack', or 'masked_band'.\n",
            bc.c_str());
        return 1;
    }
    // masked_band = racetrack demag + a centred width-W band mask
    // (free edge at the band). racetrack = full-box free-y, no mask.
    const bool use_band_mask = (bc == "masked_band");
    // Tag the output by DMI (mJ/m^2) so runs at different D never
    // overwrite each other, then by boundary condition.
    const std::string out_dir =
        "output/stochastic_llgs/validation/relax_torque/"
        + dmi_dir_tag(dmi) + "/" + bc;
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    const int fft_threads = fft_threads_from_env();

    std::printf("relax-torque validator (%s, Set-A), %d chunks x %d "
                "steps, threads=%d\n", bc.c_str(), n_chunks, chunk,
                fft_threads);
    relax_torque_probe_header();

    for (const Box& b : boxes) {
        Params p = make_default_params();
        p.nx = b.nx; p.ny = b.ny; p.dt = dt;
        p.D = dmi;
        p.skyrmion_R = skyrmion_radius;
        p.demag_kind = demag_kind;
        p.demag_accuracy = demag_accuracy;
        p.demag_tol_conv = demag_tol_conv;
        p.pulse = std::make_shared<ConstantPulse>(0.0);
        precompute(p);
        DemagState demag(p, fft_threads);

        // masked_band: a centred magnetic band of width track_width_m in
        // y (full periodic x), vacuum margins above/below carrying the
        // free edge. racetrack and the periodic/isolated demags relax
        // the whole box (mask = nullptr); racetrack's free-y exchange/DMI
        // is then applied at the box edge by effective_field_demag.
        std::vector<std::uint8_t> mask;
        const std::uint8_t* mask_ptr = nullptr;
        if (use_band_mask) {
            mask = track_mask(p.nx, p.ny, p.a, track_width_m);
            mask_ptr = mask.data();
        }
        relax_torque_probe(p, demag, mask_ptr, n_chunks, chunk,
                           alpha_relax, out_dir);
    }
    std::printf("wrote per-box NPZ to %s\n", out_dir.c_str());
    return 0;
}
