// Single-trajectory stochastic-LLGS runner. Port of
// src/stochastic_llgs/production/run_single.py::trajectory_worker.
//
// Builds a SAF skyrmion, relaxes under J = 0 (with thermal noise),
// drives it at the configured current with uniform Joule heating
// (T = T_sub + R_th j^2), and records the full per-trajectory payload.
// An optional SnapshotBuffer captures the field history (relax phase 0,
// drive phase 1) for animation.
#pragma once

#include "skyrmion/demag.hpp"   // DemagKind
#include "skyrmion/io_npz.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/types.hpp"

#include <memory>
#include <vector>

namespace skyrmion {
namespace stochastic {

struct StochasticConfig {
    // K. Strictly positive selects the thermal (Heun) dynamics via
    // T_of_j. Exactly 0 selects the noise-free mode: no Joule heating,
    // sigma_noise = 0, and the Heun stepper then reproduces the
    // deterministic path bit for bit -- used for the T = 0 baseline.
    Real T_sub = 300.0;
    Real R_th = 0.0;           // K m^4 / A^2
    Real j_current = 4.0e11;   // A/m^2
    int  nx = 256, ny = 256;
    Real dt = 5.0e-14;
    int  n_relax = 10000;
    int  n_drive = 20000;
    int  sample_every = 100;
    long long seed = 17;
    Real tol_norm = 5.0e-3;
    bool use_demag = false;
    // Demag formulation when use_demag (Set-A needs Newell).
    DemagKind demag_kind = DemagKind::Newell;
    Real demag_accuracy = 4.0;   // Newell quadrature density
    Real demag_tol_conv = 0.02;  // Newell convergence gate
    Real q_threshold = 0.5;
    int  k_consecutive = 10;
    Real skyrmion_R = 0.0;     // 0 => keep default_params value
    Real skyrmion_dw = 0.0;    // 0 => keep default
    Real D = 0.0;              // DMI override (0 => keep default)
    Real K_top = 0.0;          // top anisotropy override (0 => default)
    Real a = 0.0;              // lattice constant override (0 => default)
    Real H_z = 0.0;            // external field z-component (Tesla)
    // Pre-relaxed starting field; both non-null => seed from these
    // (skip saf_skyrmion), both null => fresh SAF skyrmion seed.
    const Field3* m_init_top = nullptr;
    const Field3* m_init_bot = nullptr;
    // Drive-phase current profile. REQUIRED -- run_trajectory throws if
    // it is unset; there is no implied DC fallback. For a DC drive pass
    // ConstantPulse(j_current) explicitly. With a shaped pulse,
    // j_current carries the PEAK amplitude and still sets the
    // Joule-heating term through T_of_j, which is exact for R_th = 0
    // as used by the racetrack campaign.
    std::shared_ptr<Pulse> drive_pulse;
    // Thermal equilibration of the relax phase. If equilibrate, run the
    // J=0 noisy dynamics until the LCC size plateaus (ignoring n_relax);
    // else run a fixed n_relax steps.
    bool equilibrate = false;
    int  equil_check_every = 200;
    int  equil_window = 10;
    Real equil_tol = 0.02;
    int  equil_k_consec = 3;
    int  equil_max_steps = 300000;
    // Field-snapshot stream (animation). Disabled if dump_snapshots false.
    bool dump_snapshots = false;
    int  snapshot_every = 0;
    int  max_snapshot_frames = 500;
    // Timestamped drive-phase progress every this-many steps (0 silent).
    int  progress_every = 0;
    // FFTW thread count for the demag transforms (0 => single-threaded).
    int  fft_threads = 0;
    // Record the per-sample translation/deformation split of the
    // deterministic dm/dt. Off by default (two extra field evaluations
    // per sample); enabled for the pulse-width study.
    bool record_dissipation = false;
};

struct StochasticPayload {
    std::vector<double> t_sample;
    std::vector<double> cx_wrapped, cy_wrapped, cx_unwrapped, cy_unwrapped;
    std::vector<double> Q, diameter;
    std::vector<double> cx_wrapped_bot, cy_wrapped_bot;
    std::vector<double> cx_unwrapped_bot, cy_unwrapped_bot;
    std::vector<double> Q_bot, diameter_bot;
    std::vector<double> cx_lcc, cy_lcc, cx_lcc_bot, cy_lcc_bot;
    std::vector<double> diameter_lcc, diameter_lcc_bot;
    // Elliptical axes (LCC second moments): major, minor, orientation.
    std::vector<double> D1_top, D2_top, theta_top;
    std::vector<double> D1_bot, D2_bot, theta_bot;
    std::vector<double> norm_drift_max;
    // Per-sample dissipation split (empty unless record_dissipation).
    // diss_trans + diss_def = diss_total to round-off; v_fit is the
    // collective velocity the split recovers, for cross-check.
    std::vector<double> diss_trans, diss_def, diss_total;
    std::vector<double> v_fit_x, v_fit_y;
    double L_x = 0.0, L_y = 0.0;   // box extent (m); track width = L_y
    double T_sub = 0.0, j_current = 0.0;   // cell coordinates (echo)
    // Pre-drive (J=0) finite-T equilibrium size + equilibration outcome.
    double D1_relaxed_top = 0.0, D2_relaxed_top = 0.0;
    int    n_relax_used = 0;
    bool   equil_converged = true;
    double T_effective = 0.0, sigma_noise = 0.0;
    bool   alive_at_end = true;
    int    flip_index = -1;
    double v_x = 0, v_y = 0, velocity = 0, hall_deg = 0, sigma_y = 0;
    double v_x_bot = 0, v_y_bot = 0, velocity_bot = 0, hall_deg_bot = 0;
    // Final top-layer m_z snapshot (row-major ny x nx, float32): input
    // to the field-classifier survival criterion applied at aggregation.
    std::vector<float> mz_final_top;
    int ny = 0, nx = 0;
};

// Run one trajectory. If `snaps` is non-null it is filled with the
// field history (caller writes it afterwards).
StochasticPayload run_trajectory(const StochasticConfig& cfg,
                                 SnapshotBuffer* snaps);

} // namespace stochastic
} // namespace skyrmion
