/// \file
/// Lattice geometry. Periodic boundary conditions are handled at the
/// call site via the inline helpers below, not by building shifted
/// copies of the field (faster, less memory).
#pragma once

#include "skyrmion/types.hpp"

#include <cstdint>
#include <vector>

namespace skyrmion {

/// Wrap an index onto a periodic axis.
/// \param i Index along the axis; may be negative or >= n.
/// \param n Axis length in cells.
/// \return The wrapped index, in [0, n).
inline int pbc_index(int i, int n) {
    // Modular index for PBC, valid for any i. Uses a division (%), so
    // the hot stencils prefer pbc_pm for their +-1 neighbour offsets.
    int r = i % n;
    return r < 0 ? r + n : r;
}

/// Periodic -1 / +1 neighbour indices along one axis of length n.
/// Branch form (no division) for the unit offsets that dominate the
/// nearest-neighbour stencils. Call once per row in the outer loop and
/// once per column in the inner loop so the row indices stay hoisted.
/// \param i Index along the axis, assumed to lie in [0, n).
/// \param n Axis length in cells.
/// \param prev Set to the index of the i-1 neighbour.
/// \param next Set to the index of the i+1 neighbour.
inline void pbc_pm(int i, int n, int& prev, int& next) {
    prev = (i == 0)     ? n - 1 : i - 1;
    next = (i + 1 == n) ? 0     : i + 1;
}

/// Physical coordinates for a flat layer at z = 0, shape (ny, nx, 3).
/// \param nx Number of cells along x.
/// \param ny Number of cells along y.
/// \param a Cell size in metres.
/// \return Site positions (x = j*a, y = i*a, z = 0) in metres.
Field3 lattice_positions(int nx, int ny, Real a);

/// Racetrack mask (row-major ny*nx, 1 = inside the magnetic region):
/// full periodic x and a centred band of physical width `width_m` along
/// y, vacuum (0) elsewhere. The band edges are the free top/bottom edges
/// (Rohart-Thiaville tilt) of a periodic-x / free-y (Racetrack)
/// track. Throws if the band does not leave at least one vacuum row each
/// side, so a width that cannot form a free edge fails loudly.
/// \param nx Number of cells along x.
/// \param ny Number of cells along y.
/// \param a Cell size in metres.
/// \param width_m Track width in metres, rounded to width_m / a cells.
/// \return Row-major ny*nx mask, 1 inside the track and 0 in vacuum.
/// \throws std::runtime_error If the rounded width is below one cell or
///         exceeds ny - 2 cells (no vacuum row left on each side).
std::vector<std::uint8_t> track_mask(int nx, int ny, Real a, Real width_m);

} // namespace skyrmion
