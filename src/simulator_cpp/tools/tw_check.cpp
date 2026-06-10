// Validation: relax the Set-A SAF skyrmion at 256x256 with newell demag
// via the same relax()/ellipse pipeline the track-width scan uses, print
// the LCC ellipse axes (expect D1 ~ D2 ~ 186.5 nm), and dump the relaxed
// field for a Python cross-check of skyrmion_ellipse_lcc.
#include "skyrmion/sweep/sweep_common.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/lcc.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"
#include "skyrmion/io_npz.hpp"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <memory>

using namespace skyrmion;
using namespace skyrmion::sweep;
using namespace skyrmion::stochastic;

int main() {
    int fft_threads = 1;
    if (const char* e = std::getenv("OMP_NUM_THREADS")) {
        if (*e && std::atoi(e) > 0) fft_threads = std::atoi(e);
    }
    Params p = make_default_params();
    p.nx = 256; p.ny = 256;
    p.demag_kind = DemagKind::Newell;
    p.demag_accuracy = 4.0;
    p.demag_tol_conv = 0.02;
    precompute(p);
    DemagState demag(p, fft_threads);
    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    const auto t0 = std::chrono::steady_clock::now();
    RelaxResult eq = relax(ic.m_top, ic.m_bot, p, &demag,
                           /*max_steps=*/4000, /*alpha=*/1.0,
                           /*tol_torque=*/1.0e-5, /*tol_dE=*/1.0e-8,
                           /*check_every=*/1000, /*print_every=*/0,
                           /*mask=*/nullptr);
    const double wall = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();
    const Ellipse e = skyrmion_ellipse_lcc(eq.m_top, p.a, +1);
    const Real d = skyrmion_diameter_lcc(eq.m_top, p.a, +1);
    std::printf("cpp relax 256 newell: threads=%d n=%d wall=%.1fs "
                "(%.1f ms/step) d_lcc=%.1f D1=%.1f D2=%.1f nm\n",
                fft_threads, eq.n_steps, wall,
                1000.0 * wall / eq.n_steps, d * 1e9,
                e.D1 * 1e9, e.D2 * 1e9);
    SnapshotBuffer buf(p.ny, p.nx, 1);
    buf.append(eq.m_top, eq.m_bot, 0, 0.0, 0,
               0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    Field3 pos = lattice_positions(p.nx, p.ny, p.a);
    buf.write("output/stochastic_llgs/m_eq_cpp.npz", pos, pos, "{}");
    std::printf("wrote output/stochastic_llgs/m_eq_cpp.npz\n");

    // ----- Stochastic-stepper benchmark (no demag, T>0) ----------------------
    // Isolates the per-step Heun stepper + thermal-noise path (the loops
    // parallelized here); no demag FFT so the stepper/noise dominate.
    {
        Params ps = make_default_params();
        ps.nx = 256; ps.ny = 256; ps.dt = 5.0e-14;
        ps.demag_kind = DemagKind::None;
        ps.pulse = std::make_shared<ConstantPulse>(2.0e11);
        precompute(ps);
        attach_thermal(ps, /*T=*/100.0, /*R_th=*/0.0, /*seed=*/7);
        SAFPair s = saf_skyrmion(ps.nx, ps.ny, ps.a, ps.skyrmion_R,
                                 ps.skyrmion_dw);
        Field3 mt = s.m_top, mb = s.m_bot;
        ThermalRng rng(7);
        HeunStochasticStepper stepper(ps, /*demag=*/nullptr, rng,
                                      ps.sigma_noise, /*tol_norm=*/5.0e-3,
                                      /*mask=*/nullptr);
        const int nstep = 4000;
        const auto ts = std::chrono::steady_clock::now();
        Real tt = 0.0;
        for (int s2 = 0; s2 < nstep; ++s2) {
            stepper.step(mt, mb, tt, ps.dt, ps);
            tt += ps.dt;
        }
        const double wstep = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - ts).count();
        std::printf("cpp stepper 256 nodemag: threads=%d n=%d "
                    "wall=%.1fs (%.2f ms/step)\n",
                    fft_threads, nstep, wstep,
                    1000.0 * wstep / nstep);
    }
    return 0;
}
