#include "skyrmion/sweep/trace_io.hpp"

#include <npy/npy.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <sstream>
#include <stdexcept>

namespace skyrmion {
namespace sweep {

namespace {

std::string json_double(double v) {
    if (std::isnan(v)) return "NaN";
    if (std::isinf(v)) return v > 0 ? "Infinity" : "-Infinity";
    std::ostringstream s;
    s.precision(17);
    s << v;
    return s.str();
}

std::string json_string(const std::string& v) {
    std::ostringstream s;
    s << '"';
    for (char c : v) {
        if (c == '"' || c == '\\') s << '\\' << c;
        else s << c;
    }
    s << '"';
    return s.str();
}

npy::tensor<double> tensor_1d(const std::vector<double>& v) {
    npy::tensor<double> t({v.size()});
    if (!v.empty()) t.copy_from(v.data(), v.size());
    return t;
}

npy::tensor<double> tensor_3d(const Field3& f) {
    npy::tensor<double> t({static_cast<std::size_t>(f.ny),
                           static_cast<std::size_t>(f.nx), 3});
    t.copy_from(f.data.data(), f.data.size());
    return t;
}

} // namespace

void Metadata::add(const std::string& key, double value) {
    items_.emplace_back(key, json_double(value));
}
void Metadata::add(const std::string& key, long long value) {
    items_.emplace_back(key, std::to_string(value));
}
void Metadata::add(const std::string& key, int value) {
    items_.emplace_back(key, std::to_string(value));
}
void Metadata::add(const std::string& key, bool value) {
    items_.emplace_back(key, value ? "true" : "false");
}
void Metadata::add(const std::string& key, const std::string& value) {
    items_.emplace_back(key, json_string(value));
}
void Metadata::merge(const Metadata& other) {
    for (const auto& kv : other.items_) items_.push_back(kv);
}
std::string Metadata::to_json() const {
    std::ostringstream s;
    s << '{';
    for (std::size_t i = 0; i < items_.size(); ++i) {
        if (i) s << ',';
        s << json_string(items_[i].first) << ':' << items_[i].second;
    }
    s << '}';
    return s.str();
}

void save_trace(const std::string& path, const Trace& trace,
                const Metadata& metadata) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    npy::npzfilewriter w(path);
    w.write("t",         tensor_1d(trace.t));
    w.write("cx_top",    tensor_1d(trace.cx_top));
    w.write("cy_top",    tensor_1d(trace.cy_top));
    w.write("cx_bot",    tensor_1d(trace.cx_bot));
    w.write("cy_bot",    tensor_1d(trace.cy_bot));
    w.write("d_top",     tensor_1d(trace.d_top));
    w.write("d_bot",     tensor_1d(trace.d_bot));
    w.write("D1_top",    tensor_1d(trace.D1_top));
    w.write("D2_top",    tensor_1d(trace.D2_top));
    w.write("theta_top", tensor_1d(trace.theta_top));
    w.write("D1_bot",    tensor_1d(trace.D1_bot));
    w.write("D2_bot",    tensor_1d(trace.D2_bot));
    w.write("theta_bot", tensor_1d(trace.theta_bot));
    w.write("psi_top",   tensor_1d(trace.psi_top));
    w.write("psi_bot",   tensor_1d(trace.psi_bot));
    w.write("Q_top",     tensor_1d(trace.Q_top));
    w.write("Q_bot",     tensor_1d(trace.Q_bot));
    if (trace.has_snapshot) {
        w.write("snapshot_m_top", tensor_3d(trace.snapshot_m_top));
        w.write("snapshot_m_bot", tensor_3d(trace.snapshot_m_bot));
        std::vector<double> st = {trace.snapshot_t};
        w.write("snapshot_t", tensor_1d(st));
    }
    const std::string json = metadata.to_json();
    std::vector<std::uint8_t> bytes(json.begin(), json.end());
    npy::tensor<std::uint8_t> meta({bytes.size()});
    if (!bytes.empty()) meta.copy_from(bytes.data(), bytes.size());
    w.write("_metadata", meta);
    w.close();
}

} // namespace sweep
} // namespace skyrmion
