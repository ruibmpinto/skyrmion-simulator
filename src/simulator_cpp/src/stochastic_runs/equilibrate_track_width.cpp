// Skyrmion track-width campaign -- STAGE 2 (thermal equilibration).
// The J=0 thermal equilibrium depends only on temperature, not current,
// so it is computed ONCE per (T_sub, ens) here and cached as
// m_thermal_T{T}_ens{ens}.npz; the drive stage (scan_track_width) loads
// the matching state and reuses it across all 6 current values, saving
// ~6x the equilibration compute.
//
// Each task loads the shared T=0 equilibrium m_eq.npz (relax_track_width),
// thermally equilibrates at its temperature until the LCC size plateaus,
// and writes the equilibrated field. SLURM array dispatch is per
// (T_sub, ens): 6 T x 100 ens = 600 tasks.
#include "skyrmion/lattice.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/sweep_common.hpp"   // resolve_indices
#include "skyrmion/stochastic/equilibrate.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"   // load_saf_npz
#include "skyrmion/io_npz.hpp"
#include "skyrmion/demag.hpp"

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::sweep;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration (box must match stages 1 & 3) -------------------
    const int nx = 350, ny = 500;
    // Lattice constant for the dump grids, read from the same
    // Params default the trajectories run with.
    const double cell_a = make_default_params().a;
    const std::vector<double> t_sub_list = {
        10.0, 50.0, 100.0, 130.0, 160.0, 200.0};
    const int n_ens = 100;
    const double r_th = 0.0;
    const double dt = 5.0e-14;
    const long long seed_base = 101;
    const double tol_norm = 5.0e-3;
    // Racetrack: periodic x, free top/bottom (y) demag (Racetrack)
    // over the full box; the track width is the transverse box extent
    // L_y = ny*a. Exchange/DMI use free-y ghost cells (mask = nullptr),
    // matching the Python pipeline. D must match stages 1 & 3.
    const DemagKind demag_kind = DemagKind::Racetrack;
    const double dmi = 0.545e-3;
    const double demag_accuracy = 4.0;
    const double demag_tol_conv = 0.02;
    // Equilibrate until the LCC size plateaus.
    const int equil_check_every = 200;     // 10 ps between size checks
    const int equil_window = 10;           // checks per window (100 ps)
    const double equil_tol = 0.02;         // relative size tolerance
    const int equil_k_consec = 3;          // consecutive stable windows
    const int equil_max_steps = 300000;    // 15 ns cap
    const std::string out_dir = "output/stochastic_llgs/scan_track_width";
    // Box-tagged so the 350x500 and 500x350 equilibria never collide.
    const std::string m_eq_path = out_dir + "/m_eq_"
        + std::to_string(nx) + "x" + std::to_string(ny) + ".npz";
    // -------------------------------------------------------------------------
    std::filesystem::create_directories(out_dir);
    int fft_threads = 1;
    if (const char* e = std::getenv("OMP_NUM_THREADS")) {
        if (*e && std::atoi(e) > 0) fft_threads = std::atoi(e);
    }
    if (!std::filesystem::exists(m_eq_path)) {
        std::fprintf(stderr,
            "error: %s not found. Run relax_track_width first.\n",
            m_eq_path.c_str());
        return 1;
    }
    const SAFPair m_eq = load_saf_npz(m_eq_path);

    Field3 pos = lattice_positions(nx, ny, cell_a);

    // ----- Grid: one entry per (T_sub, ens) ----------------------------------
    struct Cell { double T_sub; int t_idx; int ens; };
    std::vector<Cell> grid;
    for (int ti = 0; ti < static_cast<int>(t_sub_list.size()); ++ti)
        for (int ens = 0; ens < n_ens; ++ens)
            grid.push_back({t_sub_list[ti], ti, ens});

    for (int idx : resolve_indices(static_cast<int>(grid.size()))) {
        const Cell c = grid[idx];
        Params p = make_default_params();
        p.nx = nx; p.ny = ny; p.dt = dt;
        p.D = dmi;
        p.demag_kind = demag_kind;
        p.demag_accuracy = demag_accuracy;
        p.demag_tol_conv = demag_tol_conv;
        // Equilibration is at J = 0: zero the current-driven fields.
        p.pulse = std::make_shared<ConstantPulse>(0.0);
        precompute(p);
        p.H_DL = 0.0; p.H_FL = 0.0;
        // Thermal noise at this temperature (no Joule heating, J = 0).
        const long long seed = seed_base + 1000LL * c.ens
                               + 1000000LL * c.t_idx;
        attach_thermal(p, c.T_sub, r_th, seed);

        std::unique_ptr<DemagState> demag(new DemagState(p, fft_threads));
        ThermalRng rng(static_cast<std::uint64_t>(seed));
        HeunStochasticStepper stepper(p, demag.get(), rng, p.sigma_noise,
                                      tol_norm, /*mask=*/nullptr);

        Field3 m_top = m_eq.m_top;
        Field3 m_bot = m_eq.m_bot;
        // Progress line every 10 checks (every 2000 steps = 100 ps).
        const EquilResult er = equilibrate_to_plateau(
            m_top, m_bot, stepper, p, equil_check_every, equil_window,
            equil_tol, equil_k_consec, equil_max_steps,
            /*progress_every=*/10);

        char fn[200];
        std::snprintf(fn, sizeof(fn),
                      "%s/m_thermal_T%05.1f_ens%03d.npz",
                      out_dir.c_str(), c.T_sub, c.ens);
        SnapshotBuffer buf(ny, nx, 1);
        buf.append(m_top, m_bot, er.n_used, er.n_used * dt, 0,
                   0.0, 0.0, 0.0, 0.0, 0.0,
                   er.d1_relaxed, er.d2_relaxed, 0.0, 0.0);
        buf.write(fn, pos, pos, "{}");
        std::printf(
            "  T_sub=%5.1f  ens=%03d  conv=%d  n=%d  "
            "D1=%.1f  D2=%.1f nm  -> %s\n",
            c.T_sub, c.ens, er.converged ? 1 : 0, er.n_used,
            er.d1_relaxed * 1e9, er.d2_relaxed * 1e9, fn);
        std::fflush(stdout);
    }
    return 0;
}
