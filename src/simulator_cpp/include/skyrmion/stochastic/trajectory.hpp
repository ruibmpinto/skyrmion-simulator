// Single-trajectory stochastic-LLGS runner. Port of
// src/stochastic_llgs/production/run_single.py::trajectory_worker.
//
// Builds a SAF skyrmion, relaxes under J = 0 (with thermal noise),
// drives it at the configured current with uniform Joule heating
// (T = T_sub + R_th j^2), and records the full per-trajectory payload.
// An optional SnapshotBuffer captures the field history (relax phase 0,
// drive phase 1) for animation.
#pragma once

#include "skyrmion/io_npz.hpp"
#include "skyrmion/types.hpp"

#include <vector>

namespace skyrmion {
namespace stochastic {

struct StochasticConfig {
    Real T_sub = 300.0;        // K (must be > 0; Joule heating via T_of_j)
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
    Real q_threshold = 0.5;
    int  k_consecutive = 10;
    Real skyrmion_R = 0.0;     // 0 => keep default_params value
    Real skyrmion_dw = 0.0;    // 0 => keep default
    Real D = 0.0;              // DMI override (0 => keep default)
    Real H_z = 0.0;            // external field z-component (Tesla)
    // Field-snapshot stream (animation). Disabled if dump_snapshots false.
    bool dump_snapshots = false;
    int  snapshot_every = 0;
    int  max_snapshot_frames = 500;
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
    std::vector<double> norm_drift_max;
    double T_effective = 0.0, sigma_noise = 0.0;
    bool   alive_at_end = true;
    int    flip_index = -1;
    double v_x = 0, v_y = 0, velocity = 0, hall_deg = 0, sigma_y = 0;
    double v_x_bot = 0, v_y_bot = 0, velocity_bot = 0, hall_deg_bot = 0;
};

// Run one trajectory. If `snaps` is non-null it is filled with the
// field history (caller writes it afterwards).
StochasticPayload run_trajectory(const StochasticConfig& cfg,
                                 SnapshotBuffer* snaps);

} // namespace stochastic
} // namespace skyrmion
