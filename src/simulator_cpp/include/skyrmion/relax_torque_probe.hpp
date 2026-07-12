// Shared relaxation-torque probe used by validate_relax_torque and
// sweep_dmi_racetrack. Chunk-relaxes a Set-A SAF skyrmion (configured in
// the caller's Params) under `demag`, never early-stopping, recording the
// tangential-torque percentiles, the LCC ellipse D1/D2, the energy, and
// the max-torque site distance after each chunk. Writes one per-box NPZ
// (convergence series + final torque field + m_z) and a self-describing
// snapshot, and prints one summary table row. `mask` (or nullptr) selects
// the free-boundary geometry exactly as effective_field_demag.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"

#include <cstdint>
#include <string>

namespace skyrmion {

struct ProbeResult {
    double D1;
    double D2;
    double tau_max;
    double E;
};

// Print the shared column header (call once before a probe loop).
void relax_torque_probe_header();

// Relax and diagnose one box; write <out_dir>/box_<nx>x<ny>{,_snapshot}.npz
// and print the row. Returns the final-state observables.
ProbeResult relax_torque_probe(Params& p, DemagState& demag,
                               const std::uint8_t* mask,
                               int n_chunks, int chunk, double alpha_relax,
                               const std::string& out_dir);

}  // namespace skyrmion
