// Lattice geometry. Periodic boundary conditions are handled at the
// call site via the inline helpers below, not by building shifted
// copies of the field (faster, less memory).
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {

inline int pbc_index(int i, int n) {
    // Modular index for PBC, valid for any i. Uses a division (%), so
    // the hot stencils prefer pbc_pm for their +-1 neighbour offsets.
    int r = i % n;
    return r < 0 ? r + n : r;
}

// Periodic -1 / +1 neighbour indices along one axis of length n.
// Branch form (no division) for the unit offsets that dominate the
// nearest-neighbour stencils. Call once per row in the outer loop and
// once per column in the inner loop so the row indices stay hoisted.
inline void pbc_pm(int i, int n, int& prev, int& next) {
    prev = (i == 0)     ? n - 1 : i - 1;
    next = (i + 1 == n) ? 0     : i + 1;
}

// Physical coordinates for a flat layer at z = 0, shape (ny, nx, 3).
Field3 lattice_positions(int nx, int ny, Real a);

} // namespace skyrmion
