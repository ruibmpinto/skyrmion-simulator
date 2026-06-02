// Structural smoke test for the stochastic production scans. Production
// configs (256x256, 1e4-2e5 Heun steps, FFT demag) are far too slow for
// CI, so this exercises the two shared code paths at a tiny config and
// asserts the .npz key/scalar contract that the Python post-processing
// relies on:
//   * run_trajectory + save_trajectory  -> scan_tj / scan_arrhenius /
//     scan_radius (the radius D/H_z overrides are exercised too);
//   * two_skyrmion_pair_ic + pair_separation + save_pair_potential ->
//     pair_potential.
// Scalars must round-trip as 0-d arrays (numpy Python scalars); a (1,)
// array would make int()/float() raise under numpy 2.x.
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/pair_ic.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"
#include "skyrmion/stochastic/trajectory.hpp"
#include "skyrmion/stochastic/trajectory_io.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace skyrmion;
using namespace skyrmion::stochastic;
using test_common::TestRef;
using test_common::TestRunner;

namespace {

// True iff `path` contains every key, each readable as the expected
// dtype (TestRef throws on a missing key or a dtype mismatch).
bool has_doubles(TestRef& ref, const std::vector<std::string>& arrays,
                 const std::vector<std::string>& scalars) {
    for (const auto& k : arrays) {
        const auto v = ref.vec<double>(k);
        (void)v;
    }
    for (const auto& k : scalars) {
        const double s = ref.scalar<double>(k);
        (void)s;
    }
    return true;
}

} // namespace

int main() {
    TestRunner r;
    const std::filesystem::path dir =
        std::filesystem::temp_directory_path() / "skyrmion_smoke";
    std::filesystem::create_directories(dir);

    // ---- Path 1: run_trajectory + save_trajectory (scan_*) --------------
    StochasticConfig cfg;
    cfg.T_sub = 30.0;
    cfg.R_th = 0.0;
    cfg.j_current = 1.0e11;
    cfg.nx = 24;
    cfg.ny = 24;
    cfg.dt = 5.0e-14;
    cfg.n_relax = 4;
    cfg.n_drive = 8;
    cfg.sample_every = 2;
    cfg.seed = 17;
    cfg.tol_norm = 1.0e-1;
    cfg.use_demag = false;
    cfg.q_threshold = 0.5;
    cfg.k_consecutive = 3;
    cfg.skyrmion_R = 8.0e-9;
    cfg.skyrmion_dw = 3.0e-9;
    cfg.D = 1.0e-3;            // exercise scan_radius D override
    cfg.H_z = 0.05;            // exercise scan_radius H_z override

    r.expect_no_throw("run_trajectory_tiny", [&] {
        StochasticPayload pl = run_trajectory(cfg, nullptr);
        const std::string fn = (dir / "traj.npz").string();
        save_trajectory(fn, pl, "{\"smoke\":true}",
                        {{"T_sub", cfg.T_sub},
                         {"j_current", cfg.j_current},
                         {"ens_idx", 0.0}});
        TestRef ref(fn);
        has_doubles(ref,
                    {"t_sample", "Q", "diameter", "cx_unwrapped",
                     "norm_drift_max", "diameter_lcc"},
                    {"T_effective", "sigma_noise", "v_x", "hall_deg",
                     "T_sub", "j_current", "ens_idx"});
        // int64 0-d scalars.
        (void)ref.scalar<std::int64_t>("alive_at_end");
        (void)ref.scalar<std::int64_t>("flip_index");
        if (!(pl.sigma_noise > 0.0)) {
            throw std::runtime_error("sigma_noise must be > 0 at T > 0");
        }
    });

    // ---- Path 2: pair IC + pair_separation + save_pair_potential --------
    r.expect_no_throw("pair_potential_tiny", [&] {
        const int nx = 48, ny = 48;
        Params p = make_default_params();
        p.nx = nx;
        p.ny = ny;
        p.J_current = 0.0;
        p.skyrmion_R = 8.0e-9;
        p.skyrmion_dw = 3.0e-9;
        p.pulse = std::make_shared<ConstantPulse>(0.0);
        precompute(p);
        attach_thermal(p, 30.0, 0.0, 411);
        const Real a = p.a;
        const Real Lx = nx * a, Ly = ny * a;
        const Real r_init = 40.0e-9;
        SAFPair ic = two_skyrmion_pair_ic(
            nx, ny, a, p.skyrmion_R, p.skyrmion_dw, +1,
            Lx / 2.0 - 0.5 * r_init, Ly / 2.0,
            Lx / 2.0 + 0.5 * r_init, Ly / 2.0);
        Field3 m_top = std::move(ic.m_top);
        Field3 m_bot = std::move(ic.m_bot);

        // On the freshly placed pair the measured separation must match
        // the placement to within a couple of cells.
        bool alive = true;
        const Real r_meas = pair_separation(m_top, a, alive);
        if (!alive || !(std::abs(r_meas - r_init) < 3.0 * a)) {
            throw std::runtime_error("pair_separation off the placement");
        }

        ThermalRng rng(static_cast<std::uint64_t>(p.seed));
        HeunStochasticStepper stepper(p, nullptr, rng, p.sigma_noise,
                                      5.0e-2);
        std::vector<double> t_sample, r_pair, Q;
        Real t = 0.0;
        for (int s = 1; s <= 4; ++s) {
            stepper.step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
            if (s % 2 == 0) {
                t_sample.push_back(s * p.dt);
                r_pair.push_back(pair_separation(m_top, a, alive));
                Q.push_back(topological_charge(m_top, a));
            }
        }
        const std::string fn = (dir / "pair.npz").string();
        save_pair_potential(fn, t_sample, r_pair, Q, alive, r_init, 0,
                            30.0, p.sigma_noise);
        TestRef ref(fn);
        has_doubles(ref, {"t_sample", "r_pair", "Q"},
                    {"alive_both", "r_init", "ens_idx", "T_sub",
                     "sigma_noise"});
    });

    if (std::getenv("SKYRMION_SMOKE_KEEP") == nullptr) {
        std::filesystem::remove_all(dir);
    }
    return r.report("test_stochastic_scans");
}
