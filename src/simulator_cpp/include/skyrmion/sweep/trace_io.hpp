/// \file
/// NPZ persistence for sweep traces. The trace .npz schema matches
/// src/sweeps/io.py::save_trace (per-frame observable arrays + a
/// `_metadata` blob), except metadata is stored as a 1-D uint8 array of
/// UTF-8 JSON bytes (libnpy cannot emit numpy 0-D unicode arrays). The
/// patched Python load_trace decodes both representations.
#pragma once

#include "skyrmion/sweep/trace.hpp"

#include <string>
#include <utility>
#include <vector>

namespace skyrmion {
namespace sweep {

/// Minimal ordered JSON object builder for run metadata.
///
/// Keys are appended in insertion order and never deduplicated, so the
/// emitted object preserves the order the caller added entries in.
class Metadata {
public:
    /// Append a floating-point entry. NaN and infinities are emitted as
    /// the bare tokens NaN / Infinity / -Infinity, which Python's json
    /// module accepts.
    /// \param key Entry name.
    /// \param value Entry value, written with 17 significant digits.
    void add(const std::string& key, double value);
    /// Append a 64-bit integer entry.
    /// \param key Entry name.
    /// \param value Entry value.
    void add(const std::string& key, long long value);
    /// Append an integer entry.
    /// \param key Entry name.
    /// \param value Entry value.
    void add(const std::string& key, int value);
    /// Append a boolean entry, emitted as true / false.
    /// \param key Entry name.
    /// \param value Entry value.
    void add(const std::string& key, bool value);
    /// Append a string entry; quotes and backslashes are escaped.
    /// \param key Entry name.
    /// \param value Entry value.
    void add(const std::string& key, const std::string& value);
    /// Append every entry of another Metadata, in its own order.
    /// \param other Source metadata; left unchanged.
    void merge(const Metadata& other);
    /// Serialize the accumulated entries.
    /// \return A single JSON object holding the entries in order.
    std::string to_json() const;
private:
    std::vector<std::pair<std::string, std::string>> items_;  // key -> JSON
};

/// Write a trace + metadata to a single .npz (the per-frame
/// observables). Snapshot keys are written only when
/// trace.has_snapshot is true.
///
/// Parent directories of `path` are created if missing; the metadata
/// JSON is stored under the `_metadata` key as UTF-8 bytes.
/// \param path Destination .npz path.
/// \param trace Observable history to serialize.
/// \param metadata Run metadata for the `_metadata` key.
void save_trace(const std::string& path, const Trace& trace,
                const Metadata& metadata);

} // namespace sweep
} // namespace skyrmion
