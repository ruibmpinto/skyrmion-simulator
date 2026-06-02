// Stochastic thermal-annihilation (Arrhenius) scan. Port of
// scripts/.../production/scan_arrhenius.py. j = 0 (no drive/heating);
// records t_flip = flip_index * sample_every * dt per trajectory.
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"
#include "skyrmion/sweep/sweep_common.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration --------------------------------------------------
    const std::vector<double> t_sub_list = {
        10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0};
    const double r_th = 0.0;
    const double j_current = 0.0;       // no drive => T_eff = T_sub
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const int n_relax = 4000, n_drive = 200000, sample_every = 100;
    const int n_ens = 200;
    const long long seed_base = 211;
    const double tol_norm = 5.0e-3;
    const bool use_demag = false;
    const double q_threshold = 0.5;
    const int k_consecutive = 10;
    const std::string out_dir = "output/stochastic_llgs/scan_arrhenius";
    // -------------------------------------------------------------------------
    struct Pt { double T_sub; int ens; int cell; };
    std::vector<Pt> grid;
    int cell = 0;
    for (double T : t_sub_list) {
        for (int e = 0; e < n_ens; ++e) grid.push_back({T, e, cell});
        ++cell;
    }

    for (int idx : sweep::resolve_indices(static_cast<int>(grid.size()))) {
        const Pt pt = grid[idx];
        StochasticConfig cfg;
        cfg.T_sub = pt.T_sub; cfg.j_current = j_current; cfg.R_th = r_th;
        cfg.nx = nx; cfg.ny = ny; cfg.dt = dt;
        cfg.n_relax = n_relax; cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.seed = seed_base + 1000LL * pt.ens + 1000000LL * pt.cell;
        cfg.tol_norm = tol_norm; cfg.use_demag = use_demag;
        cfg.q_threshold = q_threshold; cfg.k_consecutive = k_consecutive;

        StochasticPayload pl = run_trajectory(cfg, nullptr);
        const double t_flip = (pl.flip_index < 0)
            ? std::numeric_limits<double>::quiet_NaN()
            : pl.flip_index * sample_every * dt;
        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/T%05.1f_ens%03d.npz",
                      out_dir.c_str(), pt.T_sub, pt.ens);
        save_trajectory(fn, pl, "{}",
                        {{"T_sub", pt.T_sub}, {"t_flip", t_flip},
                         {"ens_idx", static_cast<double>(pt.ens)}});
        std::printf("  T=%.1f ens=%03d alive=%d -> %s\n",
                    pt.T_sub, pt.ens, pl.alive_at_end ? 1 : 0, fn);
    }
    return 0;
}
