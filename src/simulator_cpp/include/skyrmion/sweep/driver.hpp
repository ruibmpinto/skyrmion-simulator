/// \file
/// Single-trajectory orchestrator. Port of
/// src/sweeps/driver.py::run_one with an added field-snapshot stream
/// (optional) so the animation viewer can replay the run.
///
/// Phase 1 (relax): J=0 for `n_relax` steps (skipped when 0, e.g. when
/// the caller pre-relaxed via sweep::relax). Phase 2 (drive): the
/// supplied pulse for `n_drive` steps, sampling observables every
/// `sample_every` steps. Field snapshots (when `snapshots != nullptr`)
/// are dumped at independent relax/drive cadences with phase tags
/// 0 (relax) and 1 (drive), matching the simulator's SnapshotBuffer.
#pragma once

#include "skyrmion/io_npz.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/observations.hpp"
#include "skyrmion/sweep/stepper.hpp"
#include "skyrmion/sweep/trace.hpp"
#include "skyrmion/types.hpp"

#include <cstdint>
#include <memory>

namespace skyrmion {
namespace sweep {

/// Append one field snapshot to the animation buffer. Stores the full
/// lattice fields (m_top, m_bot at every site) plus the top-layer
/// scalar observables the SnapshotBuffer schema requires. Shared by
/// run_trace (drive frames) and run_point (the initial / relaxed
/// phase-0 frames) so every frame carries one consistent payload.
/// Top-layer polarity +1.
/// \param buf Destination frame buffer.
/// \param m_top Top-layer magnetization.
/// \param m_bot Bottom-layer magnetization.
/// \param step LLGS step index of the frame.
/// \param t Physical time of the frame, in seconds.
/// \param phase_id 0 for the relaxation phase (J = 0), 1 for drive.
/// \param p Run configuration, used to measure the observables.
inline void append_snapshot(SnapshotBuffer& buf, const Field3& m_top,
                            const Field3& m_bot, int64_t step, Real t,
                            int32_t phase_id, const Params& p) {
    const Observations o = observe_state(m_top, m_bot, p);
    buf.append(m_top, m_bot, step, t, phase_id,
               o.Q_top, o.Q_bot, o.cx_top, o.cy_top, o.d_top,
               o.D1_top, o.D2_top, o.theta_top, o.psi_top);
}

/// Inputs for one run_trace call.
///
/// run_trace moves the initial fields out of this struct and swaps
/// `p->pulse` / `p->H_DL` / `p->H_FL` per phase, restoring them on exit.
struct RunTraceArgs {
    /// Run configuration; required. Mutated and restored by run_trace.
    Params* p = nullptr;
    std::shared_ptr<Pulse> pulse;  ///< Drive-phase current pulse
    int n_relax = 0;      ///< Phase-1 steps at J = 0; 0 skips the phase
    int n_drive = 0;      ///< Phase-2 (drive) steps; must be > 0
    int sample_every = 1; ///< Observable sampling cadence, in steps
    Stepper* step_drive = nullptr;  ///< Integrator for the drive phase
    Stepper* step_relax = nullptr;  ///< Integrator for the relax phase
    Field3 m_top_init;    ///< Initial top-layer magnetization (moved)
    Field3 m_bot_init;    ///< Initial bottom-layer magnetization (moved)
    /// Drive time, in seconds, at which the single in-trace snapshot
    /// of (m_top, m_bot) is recorded.
    double record_snapshot_at = -1.0;   // < 0 => no single snapshot
    int print_every = 0;  ///< Progress-line cadence (steps); 0 silences

    /// Field-snapshot stream for animation. nullptr disables it.
    SnapshotBuffer* snapshots = nullptr;
    int snapshot_every_relax = 0;      ///< steps; 0 => only endpoints
    int snapshot_every_drive = 0;      ///< steps; 0 => only endpoints
};

/// Run one trajectory: the optional J = 0 relaxation phase followed by
/// the driven phase, sampling observables into the returned trace.
/// \param args Run inputs; the initial fields are moved out of it.
/// \return The drive-phase observable trace, with the single snapshot
///         attached when record_snapshot_at was reached.
/// \throws std::runtime_error if n_relax < 0, n_drive <= 0,
///         sample_every <= 0, either stepper is null, or
///         record_snapshot_at lies beyond the drive window
///         (n_drive * dt).
Trace run_trace(RunTraceArgs& args);

} // namespace sweep
} // namespace skyrmion
