// Stochastic (T_sub x j) scan. Port of
// studies/saf_racetrack/scripts/.../production/scan_tj.py. One trajectory per
// (T_sub, j, ensemble) grid point; SLURM array (or full serial loop)
// selects which point(s) run. Seed = seed_base + 1000*ens + 1e6*cell.
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"
#include "skyrmion/sweep/sweep_common.hpp"   // resolve_indices

#include <cstdio>
#include <memory>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::stochastic;

int main() {
    // ----- Run configuration --------------------------------------------------
    const std::vector<double> t_sub_list = {
        10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0};
    const std::vector<double> j_list = {
        1.0e11, 2.0e11, 4.0e11, 8.0e11, 1.6e12};
    const double r_th = 0.0;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const int n_relax = 10000, n_drive = 40000, sample_every = 200;
    const int n_ens = 30;
    const long long seed_base = 101;
    const double tol_norm = 5.0e-3;
    const bool use_demag = false;
    const double q_threshold = 0.5;
    const int k_consecutive = 10;
    const std::string out_dir = "output/stochastic_llgs/scan_tj";
    // -------------------------------------------------------------------------
    struct Pt { double T_sub; double j; int ens; int cell; };
    std::vector<Pt> grid;
    int cell = 0;
    for (double T : t_sub_list)
        for (double j : j_list) {
            for (int e = 0; e < n_ens; ++e) grid.push_back({T, j, e, cell});
            ++cell;
        }

    for (int idx : sweep::resolve_indices(static_cast<int>(grid.size()))) {
        const Pt pt = grid[idx];
        StochasticConfig cfg;
        cfg.T_sub = pt.T_sub; cfg.j_current = pt.j; cfg.R_th = r_th;
        cfg.drive_pulse =
            std::make_shared<ConstantPulse>(cfg.j_current);
        cfg.nx = nx; cfg.ny = ny; cfg.dt = dt;
        cfg.n_relax = n_relax; cfg.n_drive = n_drive;
        cfg.sample_every = sample_every;
        cfg.seed = seed_base + 1000LL * pt.ens + 1000000LL * pt.cell;
        cfg.tol_norm = tol_norm; cfg.use_demag = use_demag;
        cfg.q_threshold = q_threshold; cfg.k_consecutive = k_consecutive;

        StochasticPayload pl = run_trajectory(cfg, nullptr);
        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/T%05.1f_j%.2e_ens%03d.npz",
                      out_dir.c_str(), pt.T_sub, pt.j, pt.ens);
        save_trajectory(fn, pl, "{}",
                        {{"T_sub", pt.T_sub}, {"j_current", pt.j},
                         {"ens_idx", static_cast<double>(pt.ens)}});
        std::printf("  T_sub=%.1f j=%.2e ens=%03d alive=%d -> %s\n",
                    pt.T_sub, pt.j, pt.ens, pl.alive_at_end ? 1 : 0, fn);
    }
    return 0;
}
