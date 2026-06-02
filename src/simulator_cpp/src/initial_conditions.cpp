#include "skyrmion/initial_conditions.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {

Field3 skyrmion_profile(int nx, int ny, Real a, Real R, Real dw, int polarity) {
    if (polarity != +1 && polarity != -1) {
        throw std::runtime_error("skyrmion_profile: polarity must be +1 or -1.");
    }
    const Real x0 = (nx - 1) * a / 2.0;
    const Real y0 = (ny - 1) * a / 2.0;
    Field3 m(ny, nx);
    for (int i = 0; i < ny; ++i) {
        const Real y = static_cast<Real>(i) * a - y0;
        for (int j = 0; j < nx; ++j) {
            const Real x = static_cast<Real>(j) * a - x0;
            const Real r = std::sqrt(x * x + y * y);
            const Real phi = std::atan2(y, x);
            Real theta = 2.0 * std::atan(std::exp(-(r - R) / dw));
            if (polarity == -1) theta = 3.14159265358979323846 - theta;
            const Real st = std::sin(theta);
            const Real ct = std::cos(theta);
            m(i, j, 0) = st * std::cos(phi);
            m(i, j, 1) = st * std::sin(phi);
            m(i, j, 2) = ct;
        }
    }
    return m;
}

Field3 uniform_state(int nx, int ny, const Vec3& dir) {
    Real n = std::sqrt(dir[0]*dir[0] + dir[1]*dir[1] + dir[2]*dir[2]);
    if (n == 0.0) {
        throw std::runtime_error("uniform_state: zero direction vector.");
    }
    const Real ux = dir[0] / n, uy = dir[1] / n, uz = dir[2] / n;
    Field3 m(ny, nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            m(i, j, 0) = ux;
            m(i, j, 1) = uy;
            m(i, j, 2) = uz;
        }
    }
    return m;
}

SAFPair saf_skyrmion(int nx, int ny, Real a, Real R, Real dw) {
    SAFPair pair;
    pair.m_top = skyrmion_profile(nx, ny, a, R, dw, +1);
    pair.m_bot = skyrmion_profile(nx, ny, a, R, dw, -1);
    return pair;
}

} // namespace skyrmion
