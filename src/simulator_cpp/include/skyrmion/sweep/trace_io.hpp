// NPZ persistence for sweep traces. The trace .npz schema matches
// src/sweeps/io.py::save_trace (per-frame observable arrays + a
// `_metadata` blob), except metadata is stored as a 1-D uint8 array of
// UTF-8 JSON bytes (libnpy cannot emit numpy 0-D unicode arrays). The
// patched Python load_trace decodes both representations.
#pragma once

#include "skyrmion/sweep/trace.hpp"

#include <string>
#include <utility>
#include <vector>

namespace skyrmion {
namespace sweep {

// Minimal ordered JSON object builder for run metadata.
class Metadata {
public:
    void add(const std::string& key, double value);
    void add(const std::string& key, long long value);
    void add(const std::string& key, int value);
    void add(const std::string& key, bool value);
    void add(const std::string& key, const std::string& value);
    void merge(const Metadata& other);
    std::string to_json() const;
private:
    std::vector<std::pair<std::string, std::string>> items_;  // key -> JSON
};

// Write a trace + metadata to a single .npz (the per-frame observables).
// Snapshot keys are written only when trace.has_snapshot is true.
void save_trace(const std::string& path, const Trace& trace,
                const Metadata& metadata);

} // namespace sweep
} // namespace skyrmion
