#include "skyrmion/stochastic/pair_ic.hpp"

#include "skyrmion/observables.hpp"

#include <cmath>
#include <limits>

namespace skyrmion {
namespace stochastic {

namespace {

constexpr Real kPi = 3.14159265358979323846;

// Core weight sum(1 - m_z) over a (sub-)field (polarity +1 convention).
Real core_weight_up(const Field3& m) {
    Real w = 0.0;
    for (int i = 0; i < m.ny; ++i)
        for (int j = 0; j < m.nx; ++j) w += 1.0 - m(i, j, 2);
    return w;
}

Field3 column_slice(const Field3& m, int j0, int j1) {
    Field3 s(m.ny, j1 - j0);
    for (int i = 0; i < m.ny; ++i)
        for (int j = j0; j < j1; ++j)
            for (int k = 0; k < 3; ++k) s(i, j - j0, k) = m(i, j, k);
    return s;
}

} // namespace

Field3 skyrmion_at_position(int nx, int ny, Real a, Real R, Real dw,
                            int polarity, Real cx, Real cy) {
    Field3 m(ny, nx);
    for (int i = 0; i < ny; ++i) {
        const Real y = static_cast<Real>(i) * a - cy;
        for (int j = 0; j < nx; ++j) {
            const Real x = static_cast<Real>(j) * a - cx;
            const Real r = std::sqrt(x * x + y * y);
            const Real phi = std::atan2(y, x);
            Real theta = 2.0 * std::atan(std::exp(-(r - R) / dw));
            if (polarity == -1) theta = kPi - theta;
            const Real st = std::sin(theta), ct = std::cos(theta);
            m(i, j, 0) = st * std::cos(phi);
            m(i, j, 1) = st * std::sin(phi);
            m(i, j, 2) = ct;
        }
    }
    return m;
}

SAFPair two_skyrmion_pair_ic(int nx, int ny, Real a, Real R, Real dw,
                             int polarity_top,
                             Real c1x, Real c1y, Real c2x, Real c2y) {
    const Field3 t1 = skyrmion_at_position(nx, ny, a, R, dw, polarity_top, c1x, c1y);
    const Field3 t2 = skyrmion_at_position(nx, ny, a, R, dw, polarity_top, c2x, c2y);
    const int pol_bot = -polarity_top;
    const Field3 b1 = skyrmion_at_position(nx, ny, a, R, dw, pol_bot, c1x, c1y);
    const Field3 b2 = skyrmion_at_position(nx, ny, a, R, dw, pol_bot, c2x, c2y);

    SAFPair out;
    out.m_top = Field3(ny, nx);
    out.m_bot = Field3(ny, nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            // Top: keep the field with the lower (more core-like) m_z.
            const bool pick1_top = t1(i, j, 2) <= t2(i, j, 2);
            const Field3& st = pick1_top ? t1 : t2;
            // Bot: opposite polarity, keep the higher m_z.
            const bool pick1_bot = b1(i, j, 2) >= b2(i, j, 2);
            const Field3& sb = pick1_bot ? b1 : b2;
            for (int k = 0; k < 3; ++k) {
                out.m_top(i, j, k) = st(i, j, k);
                out.m_bot(i, j, k) = sb(i, j, k);
            }
        }
    }
    return out;
}

Real pair_separation(const Field3& m_top, Real a, bool& alive_both) {
    const int nx = m_top.nx;
    const int mid = nx / 2;
    const Field3 left = column_slice(m_top, 0, mid);
    const Field3 right = column_slice(m_top, mid, nx);
    if (core_weight_up(left) > 0.0 && core_weight_up(right) > 0.0) {
        Center2D cl = skyrmion_center_pbc(left, a, +1);
        Center2D cr = skyrmion_center_pbc(right, a, +1);
        cr.cx += mid * a;
        return std::hypot(cr.cx - cl.cx, cr.cy - cl.cy);
    }
    alive_both = false;
    return std::numeric_limits<Real>::quiet_NaN();
}

} // namespace stochastic
} // namespace skyrmion
