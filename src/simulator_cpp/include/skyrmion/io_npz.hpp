// Snapshot writer. Accumulates m_top, m_bot, observables, and frame
// metadata into preallocated buffers; writes a single .npz on close().
#pragma once

#include "skyrmion/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace skyrmion {

class SnapshotBuffer {
public:
    // Capacity check: more than max_frames appends raises std::runtime_error.
    SnapshotBuffer(int ny, int nx, int max_frames);

    // Append one frame. Observable scalars are stored alongside the field
    // arrays so a single .npz round-trips both for the matplotlib viewer.
    void append(const Field3& m_top, const Field3& m_bot,
                int64_t step, Real time_s, int32_t phase_id,
                Real Q_top, Real Q_bot,
                Real cx_top, Real cy_top, Real diameter_top,
                Real D1_top, Real D2_top, Real theta_top,
                Real psi_top);

    // Write the .npz archive. `pos_top` / `pos_bot` are static arrays
    // saved once; `params_json` is the serialized parameter struct.
    void write(const std::string& path,
               const Field3& pos_top, const Field3& pos_bot,
               const std::string& params_json) const;

    int n_frames() const { return n_frames_; }

private:
    int ny_, nx_, max_frames_, n_frames_;
    std::vector<Real> m_top_, m_bot_;     // size max_frames*ny*nx*3
    std::vector<int64_t> step_;
    std::vector<Real> time_s_;
    std::vector<int32_t> phase_id_;
    std::vector<Real> Q_top_, Q_bot_;
    std::vector<Real> cx_top_, cy_top_;
    std::vector<Real> diameter_top_;
    std::vector<Real> D1_top_, D2_top_, theta_top_;
    std::vector<Real> psi_top_;
};

} // namespace skyrmion
