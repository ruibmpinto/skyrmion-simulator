#include "skyrmion/io_npz.hpp"

#include <npy/npy.h>

#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace skyrmion {

SnapshotBuffer::SnapshotBuffer(int ny, int nx, int max_frames)
    : ny_(ny), nx_(nx), max_frames_(max_frames), n_frames_(0) {
    if (ny <= 0 || nx <= 0 || max_frames <= 0) {
        throw std::runtime_error("SnapshotBuffer: invalid sizes.");
    }
    const std::size_t per_frame = static_cast<std::size_t>(ny) * nx * 3;
    m_top_.assign(per_frame * max_frames, 0.0);
    m_bot_.assign(per_frame * max_frames, 0.0);
    step_.reserve(max_frames);
    time_s_.reserve(max_frames);
    phase_id_.reserve(max_frames);
    Q_top_.reserve(max_frames);
    Q_bot_.reserve(max_frames);
    cx_top_.reserve(max_frames);
    cy_top_.reserve(max_frames);
    diameter_top_.reserve(max_frames);
    D1_top_.reserve(max_frames);
    D2_top_.reserve(max_frames);
    theta_top_.reserve(max_frames);
    psi_top_.reserve(max_frames);
}

void SnapshotBuffer::append(const Field3& m_top, const Field3& m_bot,
                            int64_t step, Real time_s, int32_t phase_id,
                            Real Q_top, Real Q_bot,
                            Real cx_top, Real cy_top, Real diameter_top,
                            Real D1_top, Real D2_top, Real theta_top,
                            Real psi_top) {
    if (n_frames_ >= max_frames_) {
        throw std::runtime_error(
            "SnapshotBuffer: max_dump_frames exceeded; raise Params.max_dump_frames.");
    }
    if (m_top.ny != ny_ || m_top.nx != nx_ || m_bot.ny != ny_ || m_bot.nx != nx_) {
        throw std::runtime_error("SnapshotBuffer: shape mismatch.");
    }
    const std::size_t per_frame = static_cast<std::size_t>(ny_) * nx_ * 3;
    std::memcpy(m_top_.data() + n_frames_ * per_frame,
                m_top.data.data(), per_frame * sizeof(Real));
    std::memcpy(m_bot_.data() + n_frames_ * per_frame,
                m_bot.data.data(), per_frame * sizeof(Real));
    step_.push_back(step);
    time_s_.push_back(time_s);
    phase_id_.push_back(phase_id);
    Q_top_.push_back(Q_top);
    Q_bot_.push_back(Q_bot);
    cx_top_.push_back(cx_top);
    cy_top_.push_back(cy_top);
    diameter_top_.push_back(diameter_top);
    D1_top_.push_back(D1_top);
    D2_top_.push_back(D2_top);
    theta_top_.push_back(theta_top);
    psi_top_.push_back(psi_top);
    ++n_frames_;
}

namespace {

template <typename T>
npy::tensor<T> make_tensor_1d(const std::vector<T>& src) {
    npy::tensor<T> t({src.size()});
    t.copy_from(src.data(), src.size());
    return t;
}

template <typename T>
npy::tensor<T> make_tensor_4d(const T* data, int n_frames, int ny, int nx) {
    npy::tensor<T> t({static_cast<std::size_t>(n_frames),
                      static_cast<std::size_t>(ny),
                      static_cast<std::size_t>(nx),
                      static_cast<std::size_t>(3)});
    t.copy_from(data, static_cast<std::size_t>(n_frames) * ny * nx * 3);
    return t;
}

template <typename T>
npy::tensor<T> make_tensor_3d(const std::vector<T>& src, int ny, int nx) {
    npy::tensor<T> t({static_cast<std::size_t>(ny),
                      static_cast<std::size_t>(nx),
                      static_cast<std::size_t>(3)});
    t.copy_from(src.data(), src.size());
    return t;
}

} // namespace

void SnapshotBuffer::write(const std::string& path,
                           const Field3& pos_top, const Field3& pos_bot,
                           const std::string& params_json) const {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    npy::npzfilewriter w(path);
    w.write("m_top",      make_tensor_4d<Real>(m_top_.data(), n_frames_, ny_, nx_));
    w.write("m_bot",      make_tensor_4d<Real>(m_bot_.data(), n_frames_, ny_, nx_));
    w.write("pos_top",    make_tensor_3d<Real>(pos_top.data, ny_, nx_));
    w.write("pos_bot",    make_tensor_3d<Real>(pos_bot.data, ny_, nx_));
    w.write("step",       make_tensor_1d<int64_t>(step_));
    w.write("time_s",     make_tensor_1d<Real>(time_s_));
    w.write("phase_id",   make_tensor_1d<int32_t>(phase_id_));
    w.write("Q_top",      make_tensor_1d<Real>(Q_top_));
    w.write("Q_bot",      make_tensor_1d<Real>(Q_bot_));
    w.write("cx_top",     make_tensor_1d<Real>(cx_top_));
    w.write("cy_top",     make_tensor_1d<Real>(cy_top_));
    w.write("diameter_top", make_tensor_1d<Real>(diameter_top_));
    w.write("D1_top",     make_tensor_1d<Real>(D1_top_));
    w.write("D2_top",     make_tensor_1d<Real>(D2_top_));
    w.write("theta_top",  make_tensor_1d<Real>(theta_top_));
    w.write("psi_top",    make_tensor_1d<Real>(psi_top_));
    // Parameter snapshot as raw bytes; reload with bytes(arr.tobytes()).
    std::vector<uint8_t> json_bytes(params_json.begin(), params_json.end());
    w.write("params_json", make_tensor_1d<uint8_t>(json_bytes));
    w.close();
}

} // namespace skyrmion
