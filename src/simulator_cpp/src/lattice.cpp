#include "skyrmion/lattice.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace skyrmion {

Field3 lattice_positions(int nx, int ny, Real a) {
    Field3 pos(ny, nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            pos(i, j, 0) = static_cast<Real>(j) * a;
            pos(i, j, 1) = static_cast<Real>(i) * a;
            pos(i, j, 2) = 0.0;
        }
    }
    return pos;
}

std::vector<std::uint8_t> track_mask(int nx, int ny, Real a, Real width_m) {
    const int w_cells = static_cast<int>(std::lround(width_m / a));
    if (w_cells < 1 || w_cells > ny - 2) {
        throw std::runtime_error(
            "track_mask: width " + std::to_string(width_m * 1e9)
            + " nm = " + std::to_string(w_cells) + " cells does not fit "
            "ny=" + std::to_string(ny) + " with a >=1-cell vacuum margin "
            "each side.");
    }
    const int y_lo = (ny - w_cells) / 2;
    const int y_hi = y_lo + w_cells;
    std::vector<std::uint8_t> mask(
        static_cast<std::size_t>(ny) * nx, 0);
    for (int i = y_lo; i < y_hi; ++i)
        for (int j = 0; j < nx; ++j)
            mask[static_cast<std::size_t>(i) * nx + j] = 1;
    return mask;
}

} // namespace skyrmion
