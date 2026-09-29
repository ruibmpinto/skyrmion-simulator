/// \file
/// NPZ persistence for stochastic-LLGS trajectories. Mirrors
/// src/skyrmion_simulator/stochastic_llgs/io.py::save_trajectory: payload arrays/scalars
/// plus a `meta_` header (schema version, timestamp, config echo).
#pragma once

#include "skyrmion/initial_conditions.hpp"   // SAFPair
#include "skyrmion/stochastic/trajectory.hpp"

#include <string>
#include <utility>
#include <vector>

namespace skyrmion {
namespace stochastic {

/// Load a SAF field pair (keys "m_top"/"m_bot") from an .npz written
/// by SnapshotBuffer (shape (1, ny, nx, 3)) or a plain (ny, nx, 3)
/// array. Used to pick up the relaxed m_eq / equilibrated m_thermal
/// fields.
/// \param path Path to the .npz file.
/// \return The top/bottom field pair. With a 4-d array only the first
///         frame's worth of values is taken.
/// \throws std::runtime_error if either key has a rank other than 3
///         or 4, or holds too few values for its (ny, nx, 3) shape.
SAFPair load_saf_npz(const std::string& path);

/// Write the full payload + metadata to a single .npz. `config_json`
/// is the run configuration echoed under meta_config_repr (UTF-8
/// bytes). `extras` are figure-specific scalars (e.g. D, H_z,
/// ens_idx, t_flip) written as 0-d arrays alongside the payload.
///
/// Per-sample arrays are written 1-d and scalars 0-d, so numpy loads
/// the latter as Python scalars. The dissipation-split keys appear
/// only when the run recorded them, leaving the default key set
/// unchanged. Parent directories are created as needed.
/// \param path Destination .npz path.
/// \param pl Trajectory payload to serialise.
/// \param config_json Run configuration, stored verbatim as UTF-8
///        bytes under meta_config_repr.
/// \param extras Extra named scalars written as 0-d arrays.
/// \throws std::runtime_error if pl.mz_final_top does not hold
///         exactly pl.ny * pl.nx values.
void save_trajectory(const std::string& path, const StochasticPayload& pl,
                     const std::string& config_json,
                     const std::vector<std::pair<std::string, double>>& extras
                         = {});

/// Persist the forced-pair payload (pair_potential scan). Arrays are
/// written 1-d; scalars 0-d (Python-scalar load). Mirrors the inline
/// writer in pair_potential.py.
/// \param path Destination .npz path; parent directories are created.
/// \param t_sample Sample times (s).
/// \param r_pair Pair separation per sample (m).
/// \param Q Topological charge per sample (dimensionless).
/// \param alive_both False once either skyrmion lost its core;
///        written as 1.0 / 0.0.
/// \param r_init Initial pair separation (m).
/// \param ens_idx Ensemble-member index of this trajectory.
/// \param T_sub Substrate temperature (K).
/// \param sigma_noise Thermal-field amplitude (T*sqrt(s)).
void save_pair_potential(const std::string& path,
                         const std::vector<double>& t_sample,
                         const std::vector<double>& r_pair,
                         const std::vector<double>& Q,
                         bool alive_both, double r_init, int ens_idx,
                         double T_sub, double sigma_noise);

} // namespace stochastic
} // namespace skyrmion
