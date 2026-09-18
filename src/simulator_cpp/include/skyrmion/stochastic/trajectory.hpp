/// \file
/// Single-trajectory stochastic-LLGS runner. Port of
/// src/stochastic_llgs/production/run_single.py::trajectory_worker.
///
/// Builds a SAF skyrmion, relaxes under J = 0 (with thermal noise),
/// drives it at the configured current with uniform Joule heating
/// (T = T_sub + R_th j^2), and records the full per-trajectory
/// payload. An optional SnapshotBuffer captures the field history
/// (relax phase 0, drive phase 1) for animation.
#pragma once

#include "skyrmion/demag.hpp"   // DemagKind
#include "skyrmion/io_npz.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/types.hpp"

#include <memory>
#include <vector>

namespace skyrmion {
namespace stochastic {

/// Run configuration for one stochastic-LLGS trajectory.
struct StochasticConfig {
    /// K. Strictly positive selects the thermal (Heun) dynamics via
    /// T_of_j. Exactly 0 selects the noise-free mode: no Joule
    /// heating, sigma_noise = 0, and the Heun stepper then reproduces
    /// the deterministic path bit for bit -- used for the T = 0
    /// baseline.
    Real T_sub = 300.0;
    Real R_th = 0.0;           ///< Joule heating coefficient, K m^4 / A^2
    Real j_current = 4.0e11;   ///< Drive current density, A/m^2
    /// Lattice shape: nx columns, ny rows.
    int  nx = 256, ny = 256;
    Real dt = 5.0e-14;  ///< Integration time step, s
    int  n_relax = 10000;  ///< Relax steps when equilibrate is false
    int  n_drive = 20000;  ///< Drive-phase steps
    int  sample_every = 100;  ///< Drive steps between samples
    long long seed = 17;  ///< Thermal-RNG stream key
    Real tol_norm = 5.0e-3;  ///< Max per-step |m| drift before raising
    bool use_demag = false;  ///< FFT demag instead of local K_eff
    /// Demag formulation when use_demag (Set-A needs Newell).
    DemagKind demag_kind = DemagKind::Newell;
    Real demag_accuracy = 4.0;   ///< Newell quadrature density
    Real demag_tol_conv = 0.02;  ///< Newell convergence gate
    Real q_threshold = 0.5;  ///< |Q| below which the core counts gone
    int  k_consecutive = 10;  ///< Such samples in a row = annihilation
    Real skyrmion_R = 0.0;     ///< 0 => keep default_params value
    Real skyrmion_dw = 0.0;    ///< 0 => keep default
    Real D = 0.0;              ///< DMI override (0 => keep default)
    Real K_top = 0.0;          ///< top anisotropy override (0 => default)
    Real a = 0.0;              ///< lattice constant override (0 => default)
    Real H_z = 0.0;            ///< external field z-component (Tesla)
    /// Pre-relaxed starting field; both non-null => seed from these
    /// (skip saf_skyrmion), both null => fresh SAF skyrmion seed.
    /// Borrowed, not owned.
    const Field3* m_init_top = nullptr;
    const Field3* m_init_bot = nullptr;
    /// Drive-phase current profile. REQUIRED -- run_trajectory throws
    /// if it is unset; there is no implied DC fallback. For a DC
    /// drive pass ConstantPulse(j_current) explicitly. With a shaped
    /// pulse, j_current carries the PEAK amplitude and still sets the
    /// Joule-heating term through T_of_j, which is exact for R_th = 0
    /// as used by the racetrack campaign.
    std::shared_ptr<Pulse> drive_pulse;
    /// Thermal equilibration of the relax phase. If equilibrate, run
    /// the J=0 noisy dynamics until the LCC size plateaus (ignoring
    /// n_relax); else run a fixed n_relax steps.
    bool equilibrate = false;
    int  equil_check_every = 200;  ///< Steps between size checks
    int  equil_window = 10;  ///< Checks averaged per plateau window
    Real equil_tol = 0.02;  ///< Relative tolerance on the two means
    int  equil_k_consec = 3;  ///< Passing checks in a row to stop
    int  equil_max_steps = 300000;  ///< Equilibration step budget
    /// Field-snapshot stream (animation). Disabled if dump_snapshots
    /// false. Read by the caller, which sizes the SnapshotBuffer;
    /// run_trajectory itself only honours snapshot_every.
    bool dump_snapshots = false;
    int  snapshot_every = 0;  ///< Drive steps between snapshot frames
    int  max_snapshot_frames = 500;  ///< Caller's frame capacity
    /// Timestamped drive-phase progress every this-many steps (0
    /// silent).
    int  progress_every = 0;
    /// FFTW thread count for the demag transforms (0 =>
    /// single-threaded).
    int  fft_threads = 0;
    /// Record the per-sample translation/deformation split of the
    /// deterministic dm/dt. Off by default (two extra field
    /// evaluations per sample); enabled for the pulse-width study.
    /// Requires use_demag.
    bool record_dissipation = false;
};

/// Everything one trajectory records: per-sample drive-phase series
/// plus the derived scalars. All series are sampled every
/// cfg.sample_every drive steps and share t_sample's length.
struct StochasticPayload {
    /// Sample times measured from the start of the drive phase (s).
    std::vector<double> t_sample;
    /// Top-layer centre series (m). `*_wrapped` is the whole-lattice
    /// PBC centroid, NaN once the core has collapsed; `*_unwrapped`
    /// is the PBC-unwrapped LCC centre the velocity fit uses, NaN
    /// outside the fitted window.
    std::vector<double> cx_wrapped, cy_wrapped, cx_unwrapped, cy_unwrapped;
    /// Top-layer topological charge and equivalent-disk diameter (m).
    std::vector<double> Q, diameter;
    /// Bottom-layer whole-lattice PBC centroid (m), NaN on collapse.
    std::vector<double> cx_wrapped_bot, cy_wrapped_bot;
    /// Bottom-layer unwrapped LCC centre (m), NaN outside the fit.
    std::vector<double> cx_unwrapped_bot, cy_unwrapped_bot;
    /// Bottom-layer charge and equivalent-disk diameter (m).
    std::vector<double> Q_bot, diameter_bot;
    /// Wrapped largest-connected-component core centres (m), the
    /// noise-robust tracker fed to the velocity fit.
    std::vector<double> cx_lcc, cy_lcc, cx_lcc_bot, cy_lcc_bot;
    /// LCC equivalent-disk diameters, both layers (m).
    std::vector<double> diameter_lcc, diameter_lcc_bot;
    /// Elliptical axes (LCC second moments): major, minor,
    /// orientation. D1/D2 in m, theta in radians; NaN where the LCC
    /// had fewer than 3 sites.
    std::vector<double> D1_top, D2_top, theta_top;
    std::vector<double> D1_bot, D2_bot, theta_bot;
    /// Stepper norm drift: the largest per-site deviation of |m| from
    /// 1 seen in the step that produced each sample.
    std::vector<double> norm_drift_max;
    /// Per-sample dissipation split (empty unless
    /// record_dissipation). diss_trans + diss_def = diss_total to
    /// round-off; v_fit is the collective velocity the split
    /// recovers, for cross-check.
    std::vector<double> diss_trans, diss_def, diss_total;
    /// Velocity recovered by the dissipation fit (m/s).
    std::vector<double> v_fit_x, v_fit_y;
    double L_x = 0.0, L_y = 0.0;   ///< box extent (m); track width = L_y
    /// Configuration echo: substrate temperature (K) and drive
    /// current density (A/m^2).
    double T_sub = 0.0, j_current = 0.0;
    /// Pre-drive (J=0) finite-T equilibrium size + equilibration
    /// outcome. The relaxed LCC axes are in metres, NaN if the
    /// ellipse was never defined.
    double D1_relaxed_top = 0.0, D2_relaxed_top = 0.0;
    int    n_relax_used = 0;  ///< Relax steps actually taken
    bool   equil_converged = true;  ///< Plateau criterion met
    /// Effective temperature T_sub + R_th j^2 (K) and the thermal
    /// field amplitude sigma_noise (T*sqrt(s)) it produced.
    double T_effective = 0.0, sigma_noise = 0.0;
    bool   alive_at_end = true;  ///< No annihilation run detected
    int    flip_index = -1;  ///< Annihilation sample index, -1 if none
    /// Top-layer LCC drift fit: velocities (m/s), speed magnitude
    /// (m/s), Hall angle (degrees) and sigma_y, the standard
    /// deviation of the transverse position over the second half of
    /// the fitted window (m). All NaN if the fit was skipped.
    double v_x = 0, v_y = 0, velocity = 0, hall_deg = 0, sigma_y = 0;
    /// Same drift fit for the bottom layer (m/s and degrees).
    double v_x_bot = 0, v_y_bot = 0, velocity_bot = 0, hall_deg_bot = 0;
    /// Final top-layer m_z snapshot (row-major ny x nx, float32):
    /// input to the field-classifier survival criterion applied at
    /// aggregation.
    std::vector<float> mz_final_top;
    int ny = 0, nx = 0;  ///< Lattice shape of mz_final_top
};

/// Run one trajectory. If `snaps` is non-null it is filled with the
/// field history (caller writes it afterwards).
///
/// Phase 0 relaxes at J = 0 with the noise on (a fixed cfg.n_relax
/// steps, or to an LCC-size plateau when cfg.equilibrate); phase 1
/// drives through cfg.drive_pulse at the Joule-heated temperature
/// T_of_j(j_current, T_sub, R_th) and samples the observables.
/// \param cfg Run configuration; cfg.drive_pulse must be set.
/// \param snaps Optional snapshot sink for the field history (relax
///        phase 0, drive phase 1); nullptr records no frames.
/// \return The per-sample series and the derived velocity, Hall angle
///         and survival scalars.
/// \throws std::runtime_error if cfg.drive_pulse is unset, if
///         cfg.T_sub is 0 (noise-free mode) while cfg.R_th is
///         non-zero, or if cfg.record_dissipation is set without
///         cfg.use_demag.
StochasticPayload run_trajectory(const StochasticConfig& cfg,
                                 SnapshotBuffer* snaps);

} // namespace stochastic
} // namespace skyrmion
