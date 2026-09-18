/// \file
/// Snapshot writer. Accumulates m_top, m_bot, observables, and frame
/// metadata into preallocated buffers; writes a single .npz on close().
#pragma once

#include "skyrmion/types.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace skyrmion {

/// Fixed-capacity accumulator for simulation snapshots.
///
/// All frame storage is allocated up front from (ny, nx, max_frames),
/// so appending never reallocates during a run. The archive written by
/// write() is the contract consumed by the Python plotting and
/// animation scripts; its keys are tabulated in the C++ README.
class SnapshotBuffer {
public:
    /// Allocate buffers for at most max_frames frames of an ny-by-nx
    /// two-layer lattice.
    /// \param ny Lattice rows.
    /// \param nx Lattice columns.
    /// \param max_frames Frame capacity; appending beyond it raises
    ///        std::runtime_error.
    SnapshotBuffer(int ny, int nx, int max_frames);

    /// Append one frame.
    ///
    /// Observable scalars are stored alongside the field arrays so a
    /// single .npz round-trips both for the matplotlib viewer.
    /// \param m_top Top-layer magnetization.
    /// \param m_bot Bottom-layer magnetization.
    /// \param step LLGS step index of the frame.
    /// \param time_s Physical time of the frame, in seconds.
    /// \param phase_id 0 for the relaxation phase (J = 0), 1 for drive.
    /// \param Q_top Top-layer topological charge.
    /// \param Q_bot Bottom-layer topological charge.
    /// \param cx_top Top-layer skyrmion center x, in metres.
    /// \param cy_top Top-layer skyrmion center y, in metres.
    /// \param diameter_top Top-layer skyrmion diameter, in metres.
    /// \param D1_top Major-axis diameter from second moments, metres.
    /// \param D2_top Minor-axis diameter from second moments, metres.
    /// \param theta_top Ellipse major-axis angle, in radians.
    /// \param psi_top Right-domain-wall in-plane angle, in radians.
    /// \throws std::runtime_error if the capacity is exceeded.
    void append(const Field3& m_top, const Field3& m_bot,
                int64_t step, Real time_s, int32_t phase_id,
                Real Q_top, Real Q_bot,
                Real cx_top, Real cy_top, Real diameter_top,
                Real D1_top, Real D2_top, Real theta_top,
                Real psi_top);

    /// Write the .npz archive holding every appended frame.
    /// \param path Destination archive path.
    /// \param pos_top Top-layer site positions; static, saved once.
    /// \param pos_bot Bottom-layer site positions; static, saved once.
    /// \param params_json Serialized Params struct, stored as bytes
    ///        under the params_json key.
    void write(const std::string& path,
               const Field3& pos_top, const Field3& pos_bot,
               const std::string& params_json) const;

    /// Number of frames appended so far.
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
