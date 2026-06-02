// Single-trajectory orchestrator. Port of
// src/sweeps/driver.py::run_one with an added field-snapshot stream
// (optional) so the animation viewer can replay the run.
//
// Phase 1 (relax): J=0 for `n_relax` steps (skipped when 0, e.g. when
// the caller pre-relaxed via sweep::relax). Phase 2 (drive): the
// supplied pulse for `n_drive` steps, sampling observables every
// `sample_every` steps. Field snapshots (when `snapshots != nullptr`)
// are dumped at independent relax/drive cadences with phase tags
// 0 (relax) and 1 (drive), matching the simulator's SnapshotBuffer.
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

// Append one field snapshot to the animation buffer. Stores the full
// lattice fields (m_top, m_bot at every site) plus the top-layer scalar
// observables the SnapshotBuffer schema requires. Shared by run_trace
// (drive frames) and run_point (the initial / relaxed phase-0 frames)
// so every frame carries one consistent payload. Top-layer polarity +1.
inline void append_snapshot(SnapshotBuffer& buf, const Field3& m_top,
                            const Field3& m_bot, int64_t step, Real t,
                            int32_t phase_id, const Params& p) {
    const Observations o = observe_state(m_top, m_bot, p);
    buf.append(m_top, m_bot, step, t, phase_id,
               o.Q_top, o.Q_bot, o.cx_top, o.cy_top, o.d_top,
               o.D1_top, o.D2_top, o.theta_top, o.psi_top);
}

struct RunTraceArgs {
    Params* p = nullptr;
    std::shared_ptr<Pulse> pulse;
    int n_relax = 0;
    int n_drive = 0;
    int sample_every = 1;
    Stepper* step_drive = nullptr;
    Stepper* step_relax = nullptr;
    Field3 m_top_init;
    Field3 m_bot_init;
    double record_snapshot_at = -1.0;   // < 0 => no single snapshot
    int print_every = 0;

    // Field-snapshot stream for animation. nullptr disables it.
    SnapshotBuffer* snapshots = nullptr;
    int snapshot_every_relax = 0;        // steps; 0 => only endpoints
    int snapshot_every_drive = 0;        // steps; 0 => only endpoints
};

Trace run_trace(RunTraceArgs& args);

} // namespace sweep
} // namespace skyrmion
