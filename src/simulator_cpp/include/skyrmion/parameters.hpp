/// \file
/// Simulation parameters mirroring Python parameters.default_params().
#pragma once

#include "skyrmion/pulses.hpp"
#include "skyrmion/types.hpp"

#include <memory>
#include <string>

namespace skyrmion {

/// Magnetostatic boundary condition / kernel selector. None disables
/// the explicit demag field (the local-K_eff path folds a uniform demag
/// into the anisotropy); Slab uses the analytic thin-slab kernel;
/// Newell uses the Newell kernel with doubly periodic images.
///
/// NewellFreeBC: Newell kernel on a 2N zero-padded grid (isolated /
/// free-boundary, no periodic images; matches mumax3's default demag and
/// the Python kind='newell_freebc').
/// Racetrack: mixed track geometry -- periodic along x (circular
/// convolution, grid width nx) and free/isolated along y (grid height
/// 2*ny, zero-padded top/bottom). The strip is infinite along x and has
/// open edges across its width.
enum class DemagKind { None, Slab, Newell, NewellFreeBC, Racetrack };

/// Per-cell-pair Newell tensor evaluation. Closed: the Newell-Williams-
/// Dunlop 1993 closed form (exact to machine precision; matches the Python
/// _METHOD='closed' default). Quadrature: mumax3-style adaptive Gauss-
/// Legendre surface integration (converges to Closed as accuracy grows).
enum class DemagMethod { Closed, Quadrature };

/// Full run configuration: geometry, material constants, drive, time
/// integration, output settings and the prefactors derived from them.
struct Params {
    // -------- Lattice --------
    int  nx = 256;      ///< Lattice cells along x.
    int  ny = 256;      ///< Lattice cells along y.
    Real a  = 2.0e-9;   ///< Cell size (m).

    // -------- Material (SAF Co/Pt) --------
    Real Ms     = 1.43e6;     ///< Saturation magnetisation (A/m).
    Real A_ex   = 16.0e-12;   ///< Exchange stiffness (J/m).
    Real D      = 0.85e-3;    ///< Interfacial DMI constant (J/m^2).
    // Real K_top = 1.294e6;  // measured (Hk_top = 12.4 mT)
    // Paper sims raise Hk_top to 36 mT (supp. 1.1).
    Real K_top  = 1.3106e6;   ///< Top-layer uniaxial PMA (J/m^3).
    Real K_bot  = 1.31e6;     ///< Bottom-layer uniaxial PMA (J/m^3).
    Real alpha  = 0.14;       ///< Gilbert damping (dimensionless).
    Real t_Co   = 1.3e-9;     ///< Co layer thickness (m).
    Real d_Ru   = 1.35e-9;    ///< Ru spacer thickness (m).

    // -------- Constants --------
    /// Vacuum permeability (T m / A).
    Real mu0   = 4.0e-7 * 3.14159265358979323846;
    Real gamma_ = 194.8e9;   ///< Gyromagnetic ratio (rad / (s T)).

    // -------- Interlayer RKKY --------
    /// Antiparallel interlayer coupling field (T).
    Real H_RKKY = 0.205;

    // -------- External field --------
    Vec3 H_ext = {0.0, 0.0, 0.0};   ///< Uniform applied field (T).

    // -------- Spin-orbit torques --------
    Real DL_SOT    = 2.21e-14;  ///< Damping-like coeff. (T / (A/m^2)).
    Real FL_SOT    = 0.53e-14;  ///< Field-like coeff. (T / (A/m^2)).
    Vec3 p_hat     = {0.0, 1.0, 0.0};  ///< Spin polarisation direction.
    /// Drive J(t) in A/m^2, the only source of current for the
    /// integrators. No default: set it before integrating in time.
    std::shared_ptr<Pulse> pulse;

    // -------- Topological spin Hall --------
    /// Topological spin Hall coupling; 0 disables the TSH torque.
    Real lambda_sq      = 0.0;
    Real P              = 0.5;  ///< Current spin polarisation fraction.
    /// Bohr magneton over elementary charge (J / (T C)).
    Real mu_B_over_q_e  = 9.2740100783e-24 / 1.602176634e-19;

    // -------- Time integration --------
    Real dt      = 5.0e-14;   ///< Integration time step (s).
    int  n_relax = 10000;     ///< Steps in the relaxation phase.
    int  n_steps = 5000;      ///< Steps in the driven phase.

    // -------- Initial condition --------
    Real skyrmion_R  = 93.25e-9;   ///< Seed skyrmion radius (m).
    Real skyrmion_dw = 27.0e-9;    ///< Seed domain-wall width (m).

    // -------- Output / snapshots --------
    /// Directory receiving the .npz output files.
    std::string output_dir       = "output";
    /// File name of the snapshot archive inside output_dir.
    std::string snapshot_file    = "snapshots.npz";
    /// File name of the observables archive inside output_dir.
    std::string observables_file = "observables.npz";
    /// Whether spin-configuration frames are recorded at all.
    bool dump_snapshots          = true;
    /// Snapshot stride during relaxation, in steps.
    int  dump_every_relax        = 200;
    /// Snapshot stride during the driven phase, in steps.
    int  dump_every_drive        = 100;
    /// Upper bound on the number of recorded frames.
    int  max_dump_frames         = 500;

    // -------- Demag --------
    DemagKind demag_kind = DemagKind::None;  ///< matches Python rhs_local_keff
    /// Closed-form Newell tensor by default (exact, fast; matches Python).
    DemagMethod demag_method = DemagMethod::Closed;
    Real      demag_accuracy = 8.0;          ///< Newell quadrature only
    /// Relative tolerance of the Newell quadrature convergence check.
    Real      demag_tol_conv = 2.0e-2;       // Newell (typical floor at 32x32)
    /// Periodic image wraps per periodic direction (0 = minimum image);
    /// truncation residual ~ ((pbc_images + 1) * L)^-3.
    int       pbc_images     = 2;

    // -------- Thermal (stochastic LLG); filled by attach_thermal --------
    Real      T           = 0.0;             ///< bath temperature (K)
    /// Joule heating coefficient: T(j) = T_sub + R_th * j^2.
    Real      R_th        = 0.0;             // Joule thermal resistance (K m^4 / A^2)
    long long seed        = 0;               ///< master noise seed
    Real      k_B         = 1.380649e-23;    ///< Boltzmann constant (J/K)
    Real      V_cell      = 0.0;             ///< a^2 * t_Co (m^3)
    Real      sigma_noise = 0.0;             ///< FDT amplitude (T*sqrt(s))

    // -------- Precomputed (filled by precompute) --------
    Real C_ex       = 0.0;  ///< Exchange prefactor 2 A_ex/(Ms a^2) (T).
    Real C_dmi      = 0.0;  ///< DMI prefactor D/(Ms a) (T).
    Real C_anis_top = 0.0;  ///< 2 K_top/Ms - mu0 Ms (T).
    Real C_anis_bot = 0.0;  ///< 2 K_bot/Ms - mu0 Ms (T).
    Real gamma_p    = 0.0;  ///< gamma / (1 + alpha^2) (rad / (s T)).
};

/// Populate precomputed prefactors. Mirrors parameters._precompute()
/// in Python.
/// Fills C_ex, C_dmi, C_anis_top, C_anis_bot and gamma_p from the
/// material fields already set on `p`.
/// \param p Parameter set, updated in place.
void precompute(Params& p);

/// Default parameter set (Co/Pt SAF) with no drive (`pulse` null).
/// \return A fully precomputed parameter set.
Params make_default_params();

/// Bare (no thin-film K_eff correction) anisotropy prefactors in Tesla.
/// Used by the demag-aware field path so the slab demag isn't double-counted.
struct BareAnis {
    Real C_top;  ///< 2 K_top / Ms (T).
    Real C_bot;  ///< 2 K_bot / Ms (T).
};
/// \param p Parameter set supplying K_top, K_bot and Ms.
/// \return The two bare anisotropy prefactors, in Tesla.
BareAnis bare_anis_prefactors(const Params& p);

/// Effective anisotropy summary (J/m^3) — used by phase-diagram scripts.
struct EffectiveAnis {
    Real K_top;           ///< Bare top-layer anisotropy (J/m^3).
    Real K_bot;           ///< Bare bottom-layer anisotropy (J/m^3).
    Real K_eff_top;       ///< K_top - mu0 Ms^2 / 2 (J/m^3).
    Real K_eff_bot;       ///< K_bot - mu0 Ms^2 / 2 (J/m^3).
    Real K_eff_avg;       ///< Mean of K_eff_top and K_eff_bot (J/m^3).
    Real mu0_Ms2_over_2;  ///< Thin-film demag term mu0 Ms^2 / 2 (J/m^3).
};
/// \param p Parameter set supplying K_top, K_bot, mu0 and Ms.
/// \return The bare and thin-film-corrected anisotropies, in J/m^3.
EffectiveAnis effective_anisotropy(const Params& p);

/// Critical DMI for spontaneous domain formation, 4 sqrt(A_ex K_eff_avg)
/// / pi.
/// \param p Parameter set supplying A_ex and the effective anisotropy.
/// \return The critical DMI constant, in J/m^2.
/// \throws std::runtime_error If K_eff_avg <= 0 (the ratio is undefined).
Real critical_dmi(const Params& p);

/// Effective PMA anisotropy field, 2 K_eff_avg / Ms.
/// \param p Parameter set supplying Ms and the effective anisotropy.
/// \return The anisotropy field, in Tesla.
/// \throws std::runtime_error If K_eff_avg <= 0.
Real pma_anisotropy_field(const Params& p);

/// Compact JSON serialization of run-time parameters; embedded into the
/// snapshot .npz so Python plotters can recover provenance.
/// \param p Parameter set to serialize.
/// \return A one-line JSON object with 17 significant digits.
std::string params_to_json(const Params& p);

/// Output-directory tag for a DMI value, e.g. 0.545e-3 -> "D0p545",
/// 0.85e-3 -> "D0p85" (mJ/m^2, trailing zeros stripped, '.' -> 'p'). The
/// canonical D-tag shared by every tool that writes a per-D subdirectory.
/// \param D DMI constant in J/m^2.
/// \return The directory tag.
std::string dmi_dir_tag(Real D);

} // namespace skyrmion
