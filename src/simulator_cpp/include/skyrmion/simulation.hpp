/// \file
/// Top-level driver: relax (phase 0) + current-driven (phase 1).
#pragma once

#include "skyrmion/parameters.hpp"

namespace skyrmion {

/// Run a full two-phase simulation and write its snapshot archive.
///
/// Phase 0 relaxes the seeded texture at zero current; phase 1 applies
/// the configured drive. Both phases append frames to a single
/// SnapshotBuffer, written once at the end of the run.
/// \param p Run configuration. Taken by non-const reference because
///        phase 0 temporarily zeroes the drive terms (H_DL, H_FL and
///        the pulse) and restores them before phase 1.
void run(Params& p);

} // namespace skyrmion
