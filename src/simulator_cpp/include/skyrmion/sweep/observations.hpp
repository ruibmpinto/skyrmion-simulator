// Per-frame scalar observables for sweep traces. Mirrors
// src/sweeps/observers.py::observe_state — one consistent payload for
// every S41-S49 analysis. Top-layer polarity +1, bottom layer -1.
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace sweep {

struct Observations {
    Real cx_top, cy_top, cx_bot, cy_bot;   // centroids (m), PBC-safe
    Real d_top, d_bot;                     // equiv-disk diameters (m)
    Real D1_top, D2_top, theta_top;        // top ellipse (m, m, rad)
    Real D1_bot, D2_bot, theta_bot;        // bottom ellipse
    Real psi_top, psi_bot;                 // right-DW in-plane angle (rad)
    Real Q_top, Q_bot;                     // topological charges
};

Observations observe_state(const Field3& m_top, const Field3& m_bot,
                           const Params& p);

} // namespace sweep
} // namespace skyrmion
