// One-off equilibrium relaxation for the track-width scan. The T=0,
// J=0 newell-demag equilibrium is identical for every (T_sub, j) cell,
// so it is relaxed ONCE here and written to m_eq.npz; scan_track_width
// loads that field and seeds every trajectory from it. Run this single
// job (many cores) before dispatching the drive array.
#include "skyrmion/sweep/sweep_common.hpp"   // relax, make_default_params
#include "skyrmion/observables.hpp"          // topological_charge
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

using namespace skyrmion;
using namespace skyrmion::sweep;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration (must match scan_track_width box) ---------------
    const int nx = 350, ny = 500;
    const double dt = 5.0e-14;
    // Racetrack: periodic x, free top/bottom (y) demag (Racetrack)
    // over the full box; the track width is the transverse box extent
    // L_y = ny*a. Exchange/DMI use free-y ghost cells (mask = nullptr),
    // matching the Python pipeline.
    const DemagKind demag_kind = DemagKind::Racetrack;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    // DMI below D_c, chosen from the coarse sweep to relax to the
    // ~185 nm target track skyrmion; seed radius 103 nm (diameter
    // 206 nm). Must match equilibrate & scan stages.
    const double dmi = 0.545e-3;
    const double skyrmion_radius = 103.0e-9;
    const int relax_max_steps = 200000;
    const double relax_alpha = 1.0;
    const double relax_tol_torque = 1.0e-5;
    const double relax_tol_dE = 1.0e-8;
    const int relax_check_every = 1000;
    const std::string out_dir = "output/stochastic_llgs/scan_track_width";
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    // Box-tagged so the 350x500 and 500x350 equilibria never collide.
    const std::string m_eq_path = out_dir + "/m_eq_"
        + std::to_string(nx) + "x" + std::to_string(ny) + ".npz";

    Params p = make_default_params();
    p.nx = nx; p.ny = ny; p.dt = dt;
    p.D = dmi;
    p.skyrmion_R = skyrmion_radius;
    p.demag_kind = demag_kind;
    p.demag_accuracy = demag_accuracy;
    p.demag_tol_conv = demag_tol_conv;
    precompute(p);
    int fft_threads = 1;
    if (const char* e = std::getenv("OMP_NUM_THREADS")) {
        if (*e && std::atoi(e) > 0) fft_threads = std::atoi(e);
    }
    DemagState demag(p, fft_threads);
    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    RelaxResult eq = relax(ic.m_top, ic.m_bot, p, &demag,
                           relax_max_steps, relax_alpha,
                           relax_tol_torque, relax_tol_dE,
                           relax_check_every, /*print_every=*/10000,
                           /*mask=*/nullptr);
    const Ellipse e = skyrmion_ellipse_lcc(eq.m_top, p.a, +1);
    std::printf("relax_track_width: converged=%d n=%d tau_max=%.2e T  "
                "D1=%.1f D2=%.1f nm\n",
                eq.converged ? 1 : 0, eq.n_steps, eq.tau_max,
                e.D1 * 1e9, e.D2 * 1e9);

    // Relaxed observables, stored in the snapshot so m_eq.npz is
    // self-describing (the field is the load target; these are echoes).
    const Real q_top = topological_charge(eq.m_top, p.a);
    const Real q_bot = topological_charge(eq.m_bot, p.a);
    const Center2D ctr = skyrmion_center_lcc_pbc(eq.m_top, p.a, +1);
    const Real diam = skyrmion_diameter_lcc(eq.m_top, p.a, +1);
    const Real psi = dw_angle(eq.m_top, p.a, +1, /*mz_thresh=*/0.5);

    // Write m_eq.npz via a one-frame SnapshotBuffer (keys m_top, m_bot
    // as (1, ny, nx, 3); scan_track_width strips the frame axis).
    SnapshotBuffer buf(p.ny, p.nx, 1);
    buf.append(eq.m_top, eq.m_bot, eq.n_steps, eq.n_steps * dt, 0,
               q_top, q_bot, ctr.cx, ctr.cy, diam,
               e.D1, e.D2, e.theta, psi);
    Field3 pos = lattice_positions(p.nx, p.ny, p.a);
    buf.write(m_eq_path, pos, pos, "{}");
    std::printf("wrote %s\n", m_eq_path.c_str());
    return 0;
}
