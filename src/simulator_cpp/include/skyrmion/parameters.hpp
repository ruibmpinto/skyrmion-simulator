// Simulation parameters mirroring Python parameters.default_params().
#pragma once

#include "skyrmion/pulses.hpp"
#include "skyrmion/types.hpp"

#include <memory>
#include <string>

namespace skyrmion {

// NewellFreeBC: Newell kernel on a 2N zero-padded grid (isolated /
// free-boundary, no periodic images; matches mumax3's default demag and
// the Python kind='newell_freebc').
// Racetrack: mixed track geometry -- periodic along x (circular
// convolution, grid width nx) and free/isolated along y (grid height
// 2*ny, zero-padded top/bottom). The strip is infinite along x and has
// open edges across its width.
enum class DemagKind { None, Slab, Newell, NewellFreeBC, Racetrack };

// Per-cell-pair Newell tensor evaluation. Closed: the Newell-Williams-
// Dunlop 1993 closed form (exact to machine precision; matches the Python
// _METHOD='closed' default). Quadrature: mumax3-style adaptive Gauss-
// Legendre surface integration (converges to Closed as accuracy grows).
enum class DemagMethod { Closed, Quadrature };

struct Params {
    // -------- Lattice --------
    int  nx = 256;
    int  ny = 256;
    Real a  = 2.0e-9;

    // -------- Material (SAF Co/Pt) --------
    Real Ms     = 1.43e6;
    Real A_ex   = 16.0e-12;
    Real D      = 0.85e-3;
    Real K_top  = 1.294e6;
    Real K_bot  = 1.31e6;
    Real alpha  = 0.14;
    Real t_Co   = 1.3e-9;
    Real d_Ru   = 1.35e-9;

    // -------- Constants --------
    Real mu0   = 4.0e-7 * 3.14159265358979323846;
    Real gamma_ = 194.8e9;

    // -------- Interlayer RKKY --------
    Real H_RKKY = 0.205;

    // -------- External field --------
    Vec3 H_ext = {0.0, 0.0, 0.0};

    // -------- Spin-orbit torques --------
    Real DL_SOT    = 2.21e-14;
    Real FL_SOT    = 0.53e-14;
    Real J_current = 4.0e11;
    Vec3 p_hat     = {0.0, 1.0, 0.0};
    std::shared_ptr<Pulse> pulse;  // set in make_default_params()

    // -------- Topological spin Hall --------
    Real lambda_sq      = 0.0;
    Real P              = 0.5;
    Real mu_B_over_q_e  = 9.2740100783e-24 / 1.602176634e-19;

    // -------- Time integration --------
    Real dt      = 5.0e-14;
    int  n_relax = 10000;
    int  n_steps = 5000;

    // -------- Initial condition --------
    Real skyrmion_R  = 93.25e-9;
    Real skyrmion_dw = 27.0e-9;

    // -------- Output / snapshots --------
    std::string output_dir       = "output";
    std::string snapshot_file    = "snapshots.npz";
    std::string observables_file = "observables.npz";
    bool dump_snapshots          = true;
    int  dump_every_relax        = 200;
    int  dump_every_drive        = 100;
    int  max_dump_frames         = 500;

    // -------- Demag --------
    DemagKind demag_kind = DemagKind::None;  // matches Python rhs_local_keff
    // Closed-form Newell tensor by default (exact, fast; matches Python).
    DemagMethod demag_method = DemagMethod::Closed;
    Real      demag_accuracy = 8.0;          // Newell quadrature only
    Real      demag_tol_conv = 2.0e-2;       // Newell (typical floor at 32x32)

    // -------- Thermal (stochastic LLG); filled by attach_thermal --------
    Real      T           = 0.0;             // bath temperature (K)
    Real      R_th        = 0.0;             // Joule thermal resistance (K m^4 / A^2)
    long long seed        = 0;               // master noise seed
    Real      k_B         = 1.380649e-23;    // Boltzmann constant (J/K)
    Real      V_cell      = 0.0;             // a^2 * t_Co (m^3)
    Real      sigma_noise = 0.0;             // FDT amplitude (T*sqrt(s))

    // -------- Precomputed (filled by precompute) --------
    Real C_ex       = 0.0;
    Real C_dmi      = 0.0;
    Real C_anis_top = 0.0;
    Real C_anis_bot = 0.0;
    Real H_DL       = 0.0;
    Real H_FL       = 0.0;
    Real gamma_p    = 0.0;
};

// Populate precomputed prefactors. Mirrors parameters._precompute() in Python.
void precompute(Params& p);

// Default parameter set (Co/Pt SAF) + ConstantPulse(p.J_current).
Params make_default_params();

// Bare (no thin-film K_eff correction) anisotropy prefactors in Tesla.
// Used by the demag-aware field path so the slab demag isn't double-counted.
struct BareAnis {
    Real C_top;
    Real C_bot;
};
BareAnis bare_anis_prefactors(const Params& p);

// Effective anisotropy summary (J/m^3) — used by phase-diagram scripts.
struct EffectiveAnis {
    Real K_top;
    Real K_bot;
    Real K_eff_top;
    Real K_eff_bot;
    Real K_eff_avg;
    Real mu0_Ms2_over_2;
};
EffectiveAnis effective_anisotropy(const Params& p);

Real critical_dmi(const Params& p);
Real pma_anisotropy_field(const Params& p);

// Compact JSON serialization of run-time parameters; embedded into the
// snapshot .npz so Python plotters can recover provenance.
std::string params_to_json(const Params& p);

// Output-directory tag for a DMI value, e.g. 0.545e-3 -> "D0p545",
// 0.85e-3 -> "D0p85" (mJ/m^2, trailing zeros stripped, '.' -> 'p'). The
// canonical D-tag shared by every tool that writes a per-D subdirectory.
std::string dmi_dir_tag(Real D);

} // namespace skyrmion
