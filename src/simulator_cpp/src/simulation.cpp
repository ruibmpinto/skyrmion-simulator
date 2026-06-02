#include "skyrmion/simulation.hpp"

#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/lattice.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/pulses.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>

namespace skyrmion {

namespace {

void record_frame(SnapshotBuffer& buf,
                  const Field3& m_top, const Field3& m_bot,
                  int64_t step, Real t, int32_t phase, Real a) {
    // No error silencing: observables raise loudly if the texture has
    // degenerated (e.g. annihilated skyrmion), matching the Python
    // observe_state contract.
    const Real Q_top = topological_charge(m_top, a);
    const Real Q_bot = topological_charge(m_bot, a);
    const Center2D c = skyrmion_center(m_top, a, +1);
    const Real diameter_top = skyrmion_diameter(m_top, a, +1);
    const Ellipse e = skyrmion_ellipse(m_top, a, +1);
    const Real psi = dw_angle(m_top, a, +1, 0.5);
    buf.append(m_top, m_bot, step, t, phase,
               Q_top, Q_bot, c.cx, c.cy, diameter_top,
               e.D1, e.D2, e.theta, psi);
    // Live progress: one line per recorded frame (initial / cadence /
    // final), in both phases. Mirrors the Python main.py step prints.
    std::printf("  [%s] step %6lld  t=%8.1f ps  "
                "Q_top=%+.4f  Q_bot=%+.4f\n",
                phase == 0 ? "relax" : "drive",
                static_cast<long long>(step), t * 1e12, Q_top, Q_bot);
    std::fflush(stdout);
}

} // namespace

void run(Params& p) {
    precompute(p);
    std::filesystem::create_directories(p.output_dir);

    // Initial conditions: SAF skyrmion pair.
    SAFPair pair = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw);
    Field3 m_top = std::move(pair.m_top);
    Field3 m_bot = std::move(pair.m_bot);

    // Lattice positions (z=0 for top, z=-t_Co for bottom — Ovito convention).
    Field3 pos_top = lattice_positions(p.nx, p.ny, p.a);
    Field3 pos_bot = lattice_positions(p.nx, p.ny, p.a);
    for (int i = 0; i < p.ny; ++i)
        for (int j = 0; j < p.nx; ++j)
            pos_bot(i, j, 2) = -p.t_Co;

    SnapshotBuffer buf(p.ny, p.nx, p.max_dump_frames);

    std::printf("SAF Skyrmion Simulator (C++)\n");
    std::printf("  Lattice  : %d x %d (a = %.2e m)\n", p.nx, p.ny, p.a);
    std::printf("  Relax    : %d steps (J=0)\n", p.n_relax);
    std::printf("  Drive    : %d steps, dt = %.2e s\n", p.n_steps, p.dt);
    std::printf("  Dump every %d (relax), %d (drive)\n",
                p.dump_every_relax, p.dump_every_drive);
    if (p.J_current != 0.0) {
        std::printf("  Current  : J = %.2e A/m^2\n", p.J_current);
        std::printf("  H_DL     : %.4e T\n", p.H_DL);
        std::printf("  H_FL     : %.4e T\n", p.H_FL);
    }
    std::printf("  Output   : %s/%s\n", p.output_dir.c_str(),
                p.snapshot_file.c_str());
    std::printf("%s\n", std::string(60, '-').c_str());

    const auto t_start = std::chrono::steady_clock::now();

    // ------------------------------------------------------------------------
    // Phase 0: relaxation (J = 0). Pulse temporarily swapped to zero.
    // ------------------------------------------------------------------------
    Real H_DL_save = p.H_DL, H_FL_save = p.H_FL;
    auto pulse_save = p.pulse;
    p.H_DL = 0.0;
    p.H_FL = 0.0;
    p.pulse = std::make_shared<ConstantPulse>(0.0);

    {
        std::unique_ptr<DemagState> demag;
        if (p.demag_kind != DemagKind::None) {
            demag = std::make_unique<DemagState>(p, /*threads*/ 0);
        }
        // Always dump the initial (unrelaxed) frame as phase 0, step 0.
        if (p.dump_snapshots) {
            record_frame(buf, m_top, m_bot, 0, 0.0, 0, p.a);
        }

        if (demag) {
            RHSDemag rhs(p, *demag);
            Real t = 0.0;
            for (int s = 1; s <= p.n_relax; ++s) {
                rk4_step(rhs, m_top, m_bot, t, p.dt, p);
                t += p.dt;
                if (p.dump_snapshots && (s % p.dump_every_relax == 0)
                    && s != p.n_relax) {
                    record_frame(buf, m_top, m_bot, s, t, 0, p.a);
                }
            }
        } else {
            RHSLocalKeff rhs(p);
            Real t = 0.0;
            for (int s = 1; s <= p.n_relax; ++s) {
                rk4_step(rhs, m_top, m_bot, t, p.dt, p);
                t += p.dt;
                if (p.dump_snapshots && (s % p.dump_every_relax == 0)
                    && s != p.n_relax) {
                    record_frame(buf, m_top, m_bot, s, t, 0, p.a);
                }
            }
        }
        // Final relaxed frame (last frame of phase 0).
        if (p.dump_snapshots) {
            record_frame(buf, m_top, m_bot,
                         p.n_relax, p.n_relax * p.dt, 0, p.a);
        }
    }

    std::printf("Relaxation done.\n%s\n",
                std::string(60, '-').c_str());

    // Restore drive.
    p.H_DL = H_DL_save;
    p.H_FL = H_FL_save;
    p.pulse = pulse_save;

    // ------------------------------------------------------------------------
    // Phase 1: current-driven dynamics. Pulse time resets to 0.
    // ------------------------------------------------------------------------
    {
        std::unique_ptr<DemagState> demag;
        if (p.demag_kind != DemagKind::None) {
            demag = std::make_unique<DemagState>(p, /*threads*/ 0);
        }
        // First frame of phase 1 (= relaxed state at t=0 of drive).
        if (p.dump_snapshots) {
            record_frame(buf, m_top, m_bot, 0, 0.0, 1, p.a);
        }

        if (demag) {
            RHSDemag rhs(p, *demag);
            Real t = 0.0;
            for (int s = 1; s <= p.n_steps; ++s) {
                rk4_step(rhs, m_top, m_bot, t, p.dt, p);
                t += p.dt;
                if (p.dump_snapshots && (s % p.dump_every_drive == 0)
                    && s != p.n_steps) {
                    record_frame(buf, m_top, m_bot, s, t, 1, p.a);
                }
            }
        } else {
            RHSLocalKeff rhs(p);
            Real t = 0.0;
            for (int s = 1; s <= p.n_steps; ++s) {
                rk4_step(rhs, m_top, m_bot, t, p.dt, p);
                t += p.dt;
                if (p.dump_snapshots && (s % p.dump_every_drive == 0)
                    && s != p.n_steps) {
                    record_frame(buf, m_top, m_bot, s, t, 1, p.a);
                }
            }
        }
        if (p.dump_snapshots) {
            record_frame(buf, m_top, m_bot,
                         p.n_steps, p.n_steps * p.dt, 1, p.a);
        }
    }

    const auto t_end = std::chrono::steady_clock::now();
    const double wall =
        std::chrono::duration<double>(t_end - t_start).count();
    std::printf("Done. Wall: %.1f s. Frames dumped: %d.\n",
                wall, buf.n_frames());

    if (p.dump_snapshots) {
        const std::string path =
            std::filesystem::path(p.output_dir) / p.snapshot_file;
        buf.write(path, pos_top, pos_bot, params_to_json(p));
        std::printf("Snapshots written: %s\n", path.c_str());
    }
}

} // namespace skyrmion
