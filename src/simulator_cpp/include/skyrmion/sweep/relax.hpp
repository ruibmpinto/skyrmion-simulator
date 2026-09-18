/// \file
/// Convergence-stop relaxation. Port of
/// src/phase_diagram/relaxation.py::relax with an added local-K_eff
/// path (matching the inlined K_eff relax in sweep_D_S41.py).
///
/// SOT is forcibly disabled (J=0) and Gilbert damping optionally
/// overridden for an over-damped quench; both are restored on exit.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

/// Outcome of one relaxation run.
struct RelaxResult {
    Field3 m_top;      ///< Relaxed top-layer magnetization
    Field3 m_bot;      ///< Relaxed bottom layer; empty if single-layer
    bool   converged;  ///< True if the gates were met before max_steps
    int    n_steps;    ///< RK4 steps actually taken
    double E_final;    ///< Total energy (J) at exit;
                       ///< NaN on the K_eff path (no energy-trend check)
    double tau_max;    ///< final max |m x (m x H)| (Tesla)
};

/// Quench a configuration to equilibrium with RK4 steps at J = 0,
/// stopping once the convergence gates are met.
///
/// `demag == nullptr` selects the local-K_eff path: torque from the
/// local-K_eff effective field, no energy-trend check (only tau_torque).
/// `demag != nullptr` selects the bare-K + FFT-demag path with both the
/// torque and relative-energy convergence gates.
///
/// `mask` (row-major ny*nx, 1 = inside the magnetic region, or nullptr
/// for the periodic path) selects a free-boundary geometry; it is
/// forwarded to the field assembly so cells outside the region receive
/// no field and the DMI edge condition applies.
///
/// Single-layer mode is selected by an empty `m_bot` (n_sites() == 0): a
/// lone ferromagnet relaxes via the no-demag local-K_eff field and
/// rk4_step_single, converging on the tangential torque alone. It
/// requires `demag == nullptr` (a single layer has no interlayer demag);
/// otherwise the call raises. The returned `m_bot` is empty and
/// `E_final` is NaN.
/// Matches Python phase_diagram.relaxation.relax(mask=..., m_bot=None).
///
/// \param m_top Initial top-layer magnetization; moved into the result.
/// \param m_bot Initial bottom-layer magnetization; empty selects
///        single-layer mode.
/// \param p Run configuration. alpha, gamma_p, H_DL, H_FL and pulse are
///        overridden for the quench and restored before returning.
/// \param demag FFT demag state, or nullptr for the local-K_eff path.
/// \param max_steps Step budget; the run stops unconverged at this many
///        RK4 steps.
/// \param alpha_relax Gilbert-damping override for the quench; a
///        negative value means no override (Python's None).
/// \param tol_torque Convergence threshold on max |m x (m x H)|, in
///        tesla.
/// \param tol_dE Convergence threshold on the relative total-energy
///        change between checks; used on the demag path only.
/// \param check_every Steps between convergence checks.
/// \param print_every Steps between progress lines; <= 0 prints none.
/// \param mask Row-major ny*nx geometry mask, or nullptr for periodic.
/// \return The relaxed fields plus the convergence flag, step count,
///         final energy and final max tangential torque.
/// \throws std::runtime_error if alpha_relax == 0 (zero damping cannot
///         relax), or if single-layer mode is combined with a non-null
///         `demag`.
RelaxResult relax(Field3 m_top, Field3 m_bot, Params& p,
                  DemagState* demag,
                  int max_steps, double alpha_relax,
                  double tol_torque, double tol_dE,
                  int check_every, int print_every,
                  const std::uint8_t* mask);

} // namespace sweep
} // namespace skyrmion
