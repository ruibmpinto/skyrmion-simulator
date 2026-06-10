// NPZ persistence for stochastic-LLGS trajectories. Mirrors
// src/stochastic_llgs/io.py::save_trajectory: payload arrays/scalars
// plus a `meta_` header (schema version, timestamp, config echo).
#pragma once

#include "skyrmion/initial_conditions.hpp"   // SAFPair
#include "skyrmion/stochastic/trajectory.hpp"

#include <string>
#include <utility>
#include <vector>

namespace skyrmion {
namespace stochastic {

// Load a SAF field pair (keys "m_top"/"m_bot") from an .npz written by
// SnapshotBuffer (shape (1, ny, nx, 3)) or a plain (ny, nx, 3) array.
// Used to pick up the relaxed m_eq / equilibrated m_thermal fields.
SAFPair load_saf_npz(const std::string& path);

// Write the full payload + metadata to a single .npz. `config_json` is
// the run configuration echoed under meta_config_repr (UTF-8 bytes).
// `extras` are figure-specific scalars (e.g. D, H_z, ens_idx, t_flip)
// written as 0-d arrays alongside the payload.
void save_trajectory(const std::string& path, const StochasticPayload& pl,
                     const std::string& config_json,
                     const std::vector<std::pair<std::string, double>>& extras
                         = {});

// Persist the forced-pair payload (pair_potential scan). Arrays are
// written 1-d; scalars 0-d (Python-scalar load). Mirrors the inline
// writer in pair_potential.py.
void save_pair_potential(const std::string& path,
                         const std::vector<double>& t_sample,
                         const std::vector<double>& r_pair,
                         const std::vector<double>& Q,
                         bool alive_both, double r_init, int ens_idx,
                         double T_sub, double sigma_noise);

} // namespace stochastic
} // namespace skyrmion
