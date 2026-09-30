#include "skyrmion/sweep/driver.hpp"

#include "skyrmion/sweep/observations.hpp"

#include <cstdio>
#include <stdexcept>
#include <utility>

namespace skyrmion {
namespace sweep {

Trace run_trace(RunTraceArgs& args) {
    Params& p = *args.p;
    if (args.n_relax < 0)
        throw std::runtime_error("run_trace: n_relax must be >= 0.");
    if (args.n_drive <= 0)
        throw std::runtime_error("run_trace: n_drive must be > 0.");
    if (args.sample_every <= 0)
        throw std::runtime_error("run_trace: sample_every must be > 0.");
    if (!args.step_drive || !args.step_relax)
        throw std::runtime_error("run_trace: step_drive/step_relax required.");

    // The final state at n_drive * dt is sampled unconditionally, so
    // the full drive window is reachable by the snapshot.
    const double drive_window = args.n_drive * p.dt;
    if (args.record_snapshot_at >= 0.0
        && args.record_snapshot_at > drive_window) {
        throw std::runtime_error(
            "run_trace: record_snapshot_at outside drive window.");
    }

    Field3 m_top = std::move(args.m_top_init);
    Field3 m_bot = std::move(args.m_bot_init);
    const bool dump = (args.snapshots != nullptr);

    // ----- Phase 1: relaxation (J = 0) ---------------------------------------
    auto pulse_save = p.pulse;
    p.pulse = std::make_shared<ConstantPulse>(0.0);

    if (dump && args.n_relax > 0) {
        append_snapshot(*args.snapshots, m_top, m_bot, 0, 0.0, 0, p);
    }
    {
        Real t = 0.0;
        for (int step = 1; step <= args.n_relax; ++step) {
            args.step_relax->step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
            if (dump && args.snapshot_every_relax > 0
                && step % args.snapshot_every_relax == 0
                && step != args.n_relax) {
                append_snapshot(*args.snapshots, m_top, m_bot, step,
                              t, 0, p);
            }
            if (args.print_every > 0 && step % args.print_every == 0) {
                std::printf("    relax step %d/%d (%.1f ps)\n",
                            step, args.n_relax, step * p.dt * 1e12);
            }
        }
        if (dump && args.n_relax > 0) {
            append_snapshot(*args.snapshots, m_top, m_bot, args.n_relax,
                          args.n_relax * p.dt, 0, p);
        }
    }
    p.pulse = pulse_save;

    // ----- Phase 2: drive ----------------------------------------------------
    p.pulse = args.pulse;
    Trace trace;
    try {
        Real t = 0.0;
        for (int step = 0; step < args.n_drive; ++step) {
            if (step % args.sample_every == 0) {
                const Observations o = observe_state(m_top, m_bot, p);
                trace.push(t, o);
                if (args.record_snapshot_at >= 0.0 && !trace.has_snapshot
                    && t >= args.record_snapshot_at) {
                    trace.snapshot_m_top = m_top;
                    trace.snapshot_m_bot = m_bot;
                    trace.snapshot_t = t;
                    trace.has_snapshot = true;
                }
            }
            if (dump) {
                const bool at_cadence =
                    (args.snapshot_every_drive > 0
                     && step % args.snapshot_every_drive == 0);
                if (step == 0 || at_cadence) {
                    append_snapshot(*args.snapshots, m_top, m_bot, step,
                                  t, 1, p);
                }
            }
            args.step_drive->step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
            if (args.print_every > 0
                && (step + 1) % args.print_every == 0) {
                std::printf("    drive step %d/%d (%.1f ps)\n",
                            step + 1, args.n_drive, (step + 1) * p.dt * 1e12);
            }
        }
        // Final state at t = n_drive * dt: sampled unconditionally so
        // the trace endpoint is the state the drive ends on.
        const Observations o = observe_state(m_top, m_bot, p);
        trace.push(t, o);
        if (args.record_snapshot_at >= 0.0 && !trace.has_snapshot
            && t >= args.record_snapshot_at) {
            trace.snapshot_m_top = m_top;
            trace.snapshot_m_bot = m_bot;
            trace.snapshot_t = t;
            trace.has_snapshot = true;
        }
        if (dump) {
            append_snapshot(*args.snapshots, m_top, m_bot, args.n_drive,
                          args.n_drive * p.dt, 1, p);
        }
    } catch (...) {
        p.pulse = pulse_save;
        throw;
    }
    p.pulse = pulse_save;
    return trace;
}

} // namespace sweep
} // namespace skyrmion
