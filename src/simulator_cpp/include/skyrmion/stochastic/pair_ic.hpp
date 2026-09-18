/// \file
/// Two-skyrmion initial condition for the inter-skyrmion potential
/// scan. Port of the IC helpers in
/// src/stochastic_llgs/production/pair_potential.py.
#pragma once

#include "skyrmion/initial_conditions.hpp"   // SAFPair
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

/// Single Neel skyrmion centred at (cx, cy) in metres (cf.
/// skyrmion_profile, which centres at the box midpoint).
///
/// The wall profile is theta(r) = 2*atan(exp(-(r - R)/dw)), with the
/// in-plane component radial (outward Neel); polarity -1 maps theta to
/// pi - theta.
/// \param nx Lattice columns.
/// \param ny Lattice rows.
/// \param a Lattice constant (m).
/// \param R Skyrmion radius (m).
/// \param dw Domain-wall width (m).
/// \param polarity +1 for a core along -z; -1 flips the profile.
/// \param cx Centre x coordinate (m), measured from site (0, 0).
/// \param cy Centre y coordinate (m), measured from site (0, 0).
/// \return Unit-magnitude field of shape (ny, nx, 3).
Field3 skyrmion_at_position(int nx, int ny, Real a, Real R, Real dw,
                            int polarity, Real cx, Real cy);

/// SAF pair containing two skyrmions at c1 and c2. The two single-
/// skyrmion fields are merged per site by the core-favouring z rule
/// used in pair_potential.py.
///
/// Per site the profile with the lower m_z wins, and the bottom layer
/// is the full flip -m_top (cores reversed, INWARD radial in-plane):
/// the DMI-favoured chirality in both layers, with antiparallel wall
/// rings.
/// \param nx Lattice columns.
/// \param ny Lattice rows.
/// \param a Lattice constant (m).
/// \param R Skyrmion radius (m).
/// \param dw Domain-wall width (m).
/// \param polarity_top Polarity passed to both top-layer profiles.
/// \param c1x First centre x (m).
/// \param c1y First centre y (m).
/// \param c2x Second centre x (m).
/// \param c2y Second centre y (m).
/// \return The top/bottom pair, each of shape (ny, nx, 3).
SAFPair two_skyrmion_pair_ic(int nx, int ny, Real a, Real R, Real dw,
                             int polarity_top,
                             Real c1x, Real c1y, Real c2x, Real c2y);

/// Pair separation r(t): split the top layer into left/right column
/// halves at j = nx/2, take the PBC core centroid of each (the right
/// centroid shifted back by the split column), and return their
/// distance. Sets `alive_both` false and returns NaN if either half
/// has no core (sum(1 - m_z) <= 0). Port of the measurement in
/// pair_potential.py.
/// \param m_top Top-layer magnetization, shape (ny, nx, 3).
/// \param a Lattice constant (m).
/// \param alive_both Set to false when either half has lost its core;
///        left untouched otherwise, so the caller supplies the
///        initial true value.
/// \return Centre-to-centre distance (m), or NaN when a core is gone.
Real pair_separation(const Field3& m_top, Real a, bool& alive_both);

} // namespace stochastic
} // namespace skyrmion
