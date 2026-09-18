/// \file
/// Per-frame scalar observables for sweep traces. Mirrors
/// src/sweeps/observers.py::observe_state -- one consistent payload for
/// every S41-S49 analysis. Top-layer polarity +1, bottom layer -1.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

/// Scalar observables extracted from one two-layer frame.
///
/// Every S41-S49 sweep records this same payload, so traces from
/// different analyses stay directly comparable.
struct Observations {
    Real cx_top, cy_top, cx_bot, cy_bot;   ///< Centroids (m), PBC-safe
    Real d_top, d_bot;                     ///< Equiv-disk diameters (m)
    Real D1_top, D2_top, theta_top;        ///< Top ellipse (m, m, rad)
    Real D1_bot, D2_bot, theta_bot;        ///< Bottom ellipse
    Real psi_top, psi_bot;                 ///< Right-DW angle (rad)
    Real Q_top, Q_bot;                     ///< Topological charges
};

/// Measure the observables of one frame.
/// \param m_top Top-layer magnetization.
/// \param m_bot Bottom-layer magnetization.
/// \param p Run configuration, for lattice geometry and boundary
///        conditions.
/// \return The per-frame observable payload.
Observations observe_state(const Field3& m_top, const Field3& m_bot,
                           const Params& p);

} // namespace sweep
} // namespace skyrmion
