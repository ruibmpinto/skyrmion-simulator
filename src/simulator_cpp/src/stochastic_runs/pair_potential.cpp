// Forced-pair inter-skyrmion potential scan. Port of
// studies/saf_racetrack/scripts/.../production/pair_potential.py. Two skyrmions are placed a
// distance r_init apart and the pair separation r(t) is recorded under
// J = 0 + thermal noise + demag. Bespoke IC and observable, so this is
// a standalone loop (not run_trajectory).
#include "skyrmion/demag.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/pair_ic.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"
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
    const std::vector<double> r_init_list = {180e-9, 240e-9, 320e-9, 400e-9};
    const double t_sub = 300.0;
    const double r_th = 0.0;
    const int nx = 256, ny = 256;
    const double dt = 5.0e-14;
    const int n_relax = 4000, n_drive = 20000, sample_every = 100;
    const int n_ens = 30;
    const long long seed_base = 411;
    const double tol_norm = 5.0e-3;
    const bool use_demag = true;
    const std::string out_dir = "output/stochastic_llgs/pair_potential";
    // -------------------------------------------------------------------------
    struct Pt { double r_init; int ens; int cell; };
    std::vector<Pt> grid;
    for (int cell = 0; cell < static_cast<int>(r_init_list.size()); ++cell)
        for (int e = 0; e < n_ens; ++e)
            grid.push_back({r_init_list[cell], e, cell});

    for (int idx : sweep::resolve_indices(static_cast<int>(grid.size()))) {
        const Pt pt = grid[idx];
        const long long seed = seed_base + 1000LL * pt.ens + 1000000LL * pt.cell;

        Params p = make_default_params();
        p.nx = nx; p.ny = ny; p.dt = dt;
        if (use_demag) p.demag_kind = DemagKind::Slab;
        p.pulse = std::make_shared<ConstantPulse>(0.0);
        precompute(p);
        attach_thermal(p, t_sub, r_th, seed);   // j=0, no Joule heating

        std::unique_ptr<DemagState> demag;
        if (use_demag) demag.reset(new DemagState(p, 0));

        const Real a = p.a;
        const Real L_x = nx * a, L_y = ny * a;
        const Real cy0 = L_y / 2.0;
        SAFPair ic = two_skyrmion_pair_ic(
            nx, ny, a, p.skyrmion_R, p.skyrmion_dw, +1,
            L_x / 2.0 - 0.5 * pt.r_init, cy0,
            L_x / 2.0 + 0.5 * pt.r_init, cy0);
        Field3 m_top = std::move(ic.m_top);
        Field3 m_bot = std::move(ic.m_bot);

        ThermalRng rng(static_cast<std::uint64_t>(seed));
        HeunStochasticStepper stepper(p, demag.get(), rng, p.sigma_noise,
                                      tol_norm, /*mask=*/nullptr);

        // Relax (J = 0).
        Real t = 0.0;
        for (int s = 0; s < n_relax; ++s) { stepper.step(m_top, m_bot, t, dt, p); t += dt; }

        // Drive window (J = 0): record pair separation r(t).
        std::vector<double> t_sample, r_pair, Q;
        bool alive_both = true;
        t = 0.0;
        for (int step = 1; step <= n_drive; ++step) {
            stepper.step(m_top, m_bot, t, dt, p);
            t += dt;
            if (step % sample_every == 0) {
                t_sample.push_back(step * dt);
                r_pair.push_back(pair_separation(m_top, a, alive_both));
                Q.push_back(topological_charge(m_top, a));
            }
        }

        char fn[160];
        std::snprintf(fn, sizeof(fn), "%s/r%06.1fnm_ens%03d.npz",
                      out_dir.c_str(), pt.r_init * 1e9, pt.ens);
        save_pair_potential(fn, t_sample, r_pair, Q, alive_both,
                            pt.r_init, pt.ens, t_sub, p.sigma_noise);
        std::printf("  r_init=%.1f nm ens=%03d alive_both=%d -> %s\n",
                    pt.r_init * 1e9, pt.ens, alive_both ? 1 : 0, fn);
    }
    return 0;
}
