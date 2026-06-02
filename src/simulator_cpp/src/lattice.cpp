#include "skyrmion/lattice.hpp"

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

} // namespace skyrmion
