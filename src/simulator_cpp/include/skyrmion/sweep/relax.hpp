// Convergence-stop relaxation. Port of
// src/phase_diagram/relaxation.py::relax with an added local-K_eff
// path (matching the inlined K_eff relax in sweep_D_S41.py).
//
// SOT is forcibly disabled (J=0) and Gilbert damping optionally
// overridden for an over-damped quench; both are restored on exit.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

struct RelaxResult {
    Field3 m_top;
    Field3 m_bot;
    bool   converged;
    int    n_steps;
    double E_final;    // NaN on the K_eff path (no energy-trend check)
    double tau_max;    // final max |m x (m x H)| (Tesla)
};

// `demag == nullptr` selects the local-K_eff path: torque from the
// local-K_eff effective field, no energy-trend check (only tau_torque).
// `demag != nullptr` selects the bare-K + FFT-demag path with both the
// torque and relative-energy convergence gates.
//
// `mask` (row-major ny*nx, 1 = inside the magnetic region, or nullptr for
// the periodic path) selects a free-boundary geometry; it is forwarded to
// the field assembly so cells outside the region receive no field and the
// DMI edge condition applies.
//
// Single-layer mode is selected by an empty `m_bot` (n_sites() == 0): a
// lone ferromagnet relaxes via the no-demag local-K_eff field and
// rk4_step_single, converging on the tangential torque alone. It requires
// `demag == nullptr` (a single layer has no interlayer demag); otherwise
// the call raises. The returned `m_bot` is empty and `E_final` is NaN.
// Matches Python phase_diagram.relaxation.relax(mask=..., m_bot=None).
RelaxResult relax(Field3 m_top, Field3 m_bot, Params& p,
                  DemagState* demag,
                  int max_steps, double alpha_relax,
                  double tol_torque, double tol_dE,
                  int check_every, int print_every,
                  const std::uint8_t* mask);

} // namespace sweep
} // namespace skyrmion
