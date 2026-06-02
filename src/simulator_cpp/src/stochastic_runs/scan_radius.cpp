// Stochastic (D, H_z) scan (skyrmion-size dependence). Port of
// scripts/.../production/scan_radius.py. Demag on; each (D, H_z, ens)
// grid point is one driven trajectory.
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"
#include "skyrmion/sweep/sweep_common.hpp"

#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration --------------------------------------------------
    struct Cell { double D; double H_z; };
    const std::vector<Cell> dh_cells = {
        {0.4e-3,  0.10}, {0.5e-3,  0.05}, {0.62e-3, 0.00}, {0.8e-3, -0.05}};
    const double t_sub = 300.0;
    const double j_current = 4.0e11;
    const double r_th = 0.0;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const int n_relax = 10000, n_drive = 40000, sample_every = 200;
    const int n_ens = 30;
    const long long seed_base = 311;
    const double tol_norm = 5.0e-3;
    const bool use_demag = true;
    const double q_threshold = 0.5;
    const int k_consecutive = 10;
    const std::string out_dir = "output/stochastic_llgs/scan_radius";
    // -------------------------------------------------------------------------
    struct Pt { double D; double H_z; int ens; int cell; };
    std::vector<Pt> grid;
    for (int cell = 0; cell < static_cast<int>(dh_cells.size()); ++cell) {
        for (int e = 0; e < n_ens; ++e)
            grid.push_back({dh_cells[cell].D, dh_cells[cell].H_z, e, cell});
    }

    for (int idx : sweep::resolve_indices(static_cast<int>(grid.size()))) {
        const Pt pt = grid[idx];
        StochasticConfig cfg;
        cfg.T_sub = t_sub; cfg.j_current = j_current; cfg.R_th = r_th;
        cfg.nx = nx; cfg.ny = ny; cfg.dt = dt;
        cfg.n_relax = n_relax; cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.seed = seed_base + 1000LL * pt.ens + 1000000LL * pt.cell;
        cfg.tol_norm = tol_norm; cfg.use_demag = use_demag;
        cfg.q_threshold = q_threshold; cfg.k_consecutive = k_consecutive;
        cfg.D = pt.D; cfg.H_z = pt.H_z;

        StochasticPayload pl = run_trajectory(cfg, nullptr);
        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/D%.3f_Hz%+.3f_ens%03d.npz",
                      out_dir.c_str(), pt.D * 1e3, pt.H_z, pt.ens);
        save_trajectory(fn, pl, "{}",
                        {{"D", pt.D}, {"H_z", pt.H_z},
                         {"ens_idx", static_cast<double>(pt.ens)}});
        std::printf("  D=%.3e H_z=%+.3f ens=%03d alive=%d -> %s\n",
                    pt.D, pt.H_z, pt.ens, pl.alive_at_end ? 1 : 0, fn);
    }
    return 0;
}
