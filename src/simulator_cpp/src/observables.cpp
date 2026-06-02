#include "skyrmion/observables.hpp"

#include "skyrmion/lattice.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {

namespace { constexpr Real kPi = 3.14159265358979323846; }

Real topological_charge(const Field3& m, Real a) {
    const int ny = m.ny, nx = m.nx;
    const Real inv_2a = 1.0 / (2.0 * a);
    Real sum = 0.0;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for reduction(+:sum) schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        int im, ip;
        pbc_pm(i, ny, im, ip);
        for (int j = 0; j < nx; ++j) {
            int jm, jp;
            pbc_pm(j, nx, jm, jp);
            const Real dmx_dx = (m(i, jp, 0) - m(i, jm, 0)) * inv_2a;
            const Real dmy_dx = (m(i, jp, 1) - m(i, jm, 1)) * inv_2a;
            const Real dmz_dx = (m(i, jp, 2) - m(i, jm, 2)) * inv_2a;
            const Real dmx_dy = (m(ip, j, 0) - m(im, j, 0)) * inv_2a;
            const Real dmy_dy = (m(ip, j, 1) - m(im, j, 1)) * inv_2a;
            const Real dmz_dy = (m(ip, j, 2) - m(im, j, 2)) * inv_2a;
            const Real cx = dmy_dx * dmz_dy - dmz_dx * dmy_dy;
            const Real cy = dmz_dx * dmx_dy - dmx_dx * dmz_dy;
            const Real cz = dmx_dx * dmy_dy - dmy_dx * dmx_dy;
            sum += m(i, j, 0) * cx + m(i, j, 1) * cy + m(i, j, 2) * cz;
        }
    }
    return sum * a * a / (4.0 * kPi);
}

Center2D skyrmion_center(const Field3& m, Real a, int core_polarity) {
    if (core_polarity != +1 && core_polarity != -1) {
        throw std::runtime_error("skyrmion_center: core_polarity must be +1 or -1.");
    }
    const int ny = m.ny, nx = m.nx;
    Real ws = 0.0, sum_jx = 0.0, sum_iy = 0.0;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for reduction(+:ws,sum_jx,sum_iy) schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real w = 0.5 * (1.0 - core_polarity * m(i, j, 2));
            ws += w;
            sum_jx += w * static_cast<Real>(j);
            sum_iy += w * static_cast<Real>(i);
        }
    }
    if (ws == 0.0) {
        throw std::runtime_error("skyrmion_center: empty core mask.");
    }
    return {sum_jx * a / ws, sum_iy * a / ws};
}

Center2D skyrmion_center_pbc(const Field3& m, Real a, int core_polarity) {
    if (core_polarity != +1 && core_polarity != -1) {
        throw std::runtime_error(
            "skyrmion_center_pbc: core_polarity must be +1 or -1.");
    }
    if (!(a > 0.0)) {
        throw std::runtime_error("skyrmion_center_pbc: a must be > 0.");
    }
    const int ny = m.ny, nx = m.nx;
    Real ws = 0.0, Sx = 0.0, Cx = 0.0, Sy = 0.0, Cy = 0.0;
    const Real kx = 2.0 * kPi / nx;
    const Real ky = 2.0 * kPi / ny;
    for (int i = 0; i < ny; ++i) {
        const Real ty = ky * static_cast<Real>(i);
        const Real sy = std::sin(ty), cy = std::cos(ty);
        for (int j = 0; j < nx; ++j) {
            const Real w = 0.5 * (1.0 - core_polarity * m(i, j, 2));
            ws += w;
            const Real tx = kx * static_cast<Real>(j);
            Sx += w * std::sin(tx);
            Cx += w * std::cos(tx);
            Sy += w * sy;
            Cy += w * cy;
        }
    }
    if (ws <= 0.0) {
        throw std::runtime_error(
            "skyrmion_center_pbc: non-positive total weight "
            "(skyrmion may have annihilated).");
    }
    Real ang_x = std::atan2(Sx, Cx);
    if (ang_x < 0.0) ang_x += 2.0 * kPi;
    Real ang_y = std::atan2(Sy, Cy);
    if (ang_y < 0.0) ang_y += 2.0 * kPi;
    return {ang_x * nx * a / (2.0 * kPi), ang_y * ny * a / (2.0 * kPi)};
}

Real skyrmion_diameter(const Field3& m, Real a, int core_polarity) {
    if (core_polarity != +1 && core_polarity != -1) {
        throw std::runtime_error("skyrmion_diameter: core_polarity must be +1 or -1.");
    }
    const int ny = m.ny, nx = m.nx;
    std::size_t n_inside = 0;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for reduction(+:n_inside) schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (core_polarity * m(i, j, 2) < 0.0) ++n_inside;
        }
    }
    const Real area = static_cast<Real>(n_inside) * a * a;
    return 2.0 * std::sqrt(area / kPi);
}

Ellipse skyrmion_ellipse(const Field3& m, Real a, int core_polarity) {
    if (core_polarity != +1 && core_polarity != -1) {
        throw std::runtime_error("skyrmion_ellipse: core_polarity must be +1 or -1.");
    }
    const int ny = m.ny, nx = m.nx;
    std::size_t n_inside = 0;
    Real sum_x = 0.0, sum_y = 0.0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (core_polarity * m(i, j, 2) < 0.0) {
                sum_x += static_cast<Real>(j) * a;
                sum_y += static_cast<Real>(i) * a;
                ++n_inside;
            }
        }
    }
    if (n_inside < 3) {
        throw std::runtime_error("skyrmion_ellipse: mask too small for covariance.");
    }
    const Real cx = sum_x / static_cast<Real>(n_inside);
    const Real cy = sum_y / static_cast<Real>(n_inside);
    Real sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (core_polarity * m(i, j, 2) < 0.0) {
                const Real x = static_cast<Real>(j) * a - cx;
                const Real y = static_cast<Real>(i) * a - cy;
                sxx += x * x;
                syy += y * y;
                sxy += x * y;
            }
        }
    }
    sxx /= static_cast<Real>(n_inside);
    syy /= static_cast<Real>(n_inside);
    sxy /= static_cast<Real>(n_inside);
    const Real tr = sxx + syy;
    const Real disc = 0.25 * (sxx - syy) * (sxx - syy) + sxy * sxy;
    const Real sd = std::sqrt(std::max(disc, 0.0));
    const Real lam1 = 0.5 * tr + sd;
    const Real lam2 = 0.5 * tr - sd;
    Real theta = 0.0;
    if (!(std::abs(sxx - syy) < 1e-30 && std::abs(sxy) < 1e-30)) {
        theta = 0.5 * std::atan2(2.0 * sxy, sxx - syy);
    }
    Ellipse e;
    e.D1 = 4.0 * std::sqrt(std::max(lam1, 0.0));
    e.D2 = 4.0 * std::sqrt(std::max(lam2, 0.0));
    e.theta = theta;
    return e;
}

Real dw_angle(const Field3& m, Real a, int core_polarity, Real mz_thresh) {
    const Center2D c = skyrmion_center(m, a, core_polarity);
    const int ny = m.ny, nx = m.nx;
    Real mx_sum = 0.0, my_sum = 0.0;
    std::size_t n_dw = 0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real x = static_cast<Real>(j) * a;
            if (x <= c.cx) continue;
            if (std::abs(m(i, j, 2)) >= mz_thresh) continue;
            mx_sum += m(i, j, 0);
            my_sum += m(i, j, 1);
            ++n_dw;
        }
    }
    if (n_dw == 0) {
        throw std::runtime_error("dw_angle: no DW sites in +x half-plane.");
    }
    const Real mx_avg = mx_sum / static_cast<Real>(n_dw);
    const Real my_avg = my_sum / static_cast<Real>(n_dw);
    const Real s = (mx_avg >= 0.0) ? 1.0 : -1.0;
    return std::atan2(s * my_avg, s * mx_avg);
}

} // namespace skyrmion
