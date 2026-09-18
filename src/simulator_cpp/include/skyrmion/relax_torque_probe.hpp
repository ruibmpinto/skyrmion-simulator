/// \file
/// Shared relaxation-torque probe used by validate_relax_torque and
/// sweep_dmi_racetrack. Chunk-relaxes a Set-A SAF skyrmion (configured in
/// the caller's Params) under `demag`, never early-stopping, recording the
/// tangential-torque percentiles, the LCC ellipse D1/D2, the energy, and
/// the max-torque site distance after each chunk. Writes one per-box NPZ
/// (convergence series + final torque field + m_z) and a self-describing
/// snapshot, and prints one summary table row. `mask` (or nullptr) selects
/// the free-boundary geometry exactly as effective_field_demag.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"

#include <cstdint>
#include <string>

namespace skyrmion {

/// Final-state observables of one probed box.
struct ProbeResult {
    double D1;       ///< LCC ellipse major-axis diameter (nm).
    double D2;       ///< LCC ellipse minor-axis diameter (nm).
    double tau_max;  ///< Max |m x (m x H)| over the region (Tesla).
    double E;        ///< Total energy of the relaxed state (J).
};

/// Print the shared column header (call once before a probe loop).
void relax_torque_probe_header();

/// Relax and diagnose one box; write <out_dir>/box_<nx>x<ny>{,_snapshot}.npz
/// and print the row. Returns the final-state observables.
/// The seed state is saf_skyrmion() at p.skyrmion_R / p.skyrmion_dw, and
/// relaxation runs `n_chunks` blocks of `chunk` steps with both
/// convergence tolerances set to zero, so it never stops early.
/// \param p Parameter set; passed by reference because the relaxation
///        temporarily overrides the damping and the current.
/// \param demag Demag state used for the field and the energy.
/// \param mask Free-boundary region flags, or nullptr for periodic;
///        torque percentiles are taken over the masked cells only.
/// \param n_chunks Number of relaxation blocks to run.
/// \param chunk Number of RK4 steps per block.
/// \param alpha_relax Gilbert damping override for the quench.
/// \param out_dir Existing directory receiving the two .npz files.
/// \return The final D1, D2 (nm), max torque (T) and energy (J).
ProbeResult relax_torque_probe(Params& p, DemagState& demag,
                               const std::uint8_t* mask,
                               int n_chunks, int chunk, double alpha_relax,
                               const std::string& out_dir);

}  // namespace skyrmion
