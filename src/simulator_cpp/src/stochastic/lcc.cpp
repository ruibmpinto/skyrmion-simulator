#include "skyrmion/stochastic/lcc.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

namespace { constexpr Real kPi = 3.14159265358979323846; }

std::vector<std::uint8_t> largest_core_mask_pbc(const Field3& m,
                                                int core_polarity) {
    if (core_polarity != +1 && core_polarity != -1) {
        throw std::runtime_error(
            "largest_core_mask_pbc: core_polarity must be +1 or -1.");
    }
    const int ny = m.ny, nx = m.nx;
    const std::size_t n = static_cast<std::size_t>(ny) * nx;
    std::vector<std::uint8_t> core(n, 0);
    std::vector<std::uint8_t> out(n, 0);
    bool any = false;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (core_polarity * m(i, j, 2) < 0.0) {
                core[static_cast<std::size_t>(i) * nx + j] = 1;
                any = true;
            }
        }
    }
    if (!any) return out;

    // Raster-order 4-connectivity flood-fill labelling (no PBC yet).
    // Discovery order matches scipy.ndimage.label numbering, so the
    // smallest-root tie-break below picks the same component as Python.
    std::vector<int> lab(n, 0);
    int n_lab = 0;
    std::vector<int> stack;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const int start = i * nx + j;
            if (!core[start] || lab[start]) continue;
            ++n_lab;
            lab[start] = n_lab;
            stack.clear();
            stack.push_back(start);
            while (!stack.empty()) {
                const int cur = stack.back();
                stack.pop_back();
                const int ci = cur / nx, cj = cur % nx;
                if (ci > 0)      { int nb = cur - nx; if (core[nb] && !lab[nb]) { lab[nb] = n_lab; stack.push_back(nb); } }
                if (ci < ny - 1) { int nb = cur + nx; if (core[nb] && !lab[nb]) { lab[nb] = n_lab; stack.push_back(nb); } }
                if (cj > 0)      { int nb = cur - 1;  if (core[nb] && !lab[nb]) { lab[nb] = n_lab; stack.push_back(nb); } }
                if (cj < nx - 1) { int nb = cur + 1;  if (core[nb] && !lab[nb]) { lab[nb] = n_lab; stack.push_back(nb); } }
            }
        }
    }

    // Union-find over labels; root is the smallest label in a group
    // (parent[max] = min), matching the Python fusion convention.
    std::vector<int> parent(n_lab + 1);
    for (int i = 0; i <= n_lab; ++i) parent[i] = i;
    auto find = [&](int x) {
        while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; }
        return x;
    };
    auto uni = [&](int a, int b) {
        const int ra = find(a), rb = find(b);
        if (ra != rb) parent[ra < rb ? rb : ra] = (ra < rb ? ra : rb);
    };
    // Top-bottom wrap (axis 0) and left-right wrap (axis 1).
    for (int x = 0; x < nx; ++x) {
        const int a = lab[x];
        const int b = lab[static_cast<std::size_t>(ny - 1) * nx + x];
        if (a && b) uni(a, b);
    }
    for (int y = 0; y < ny; ++y) {
        const int a = lab[static_cast<std::size_t>(y) * nx];
        const int b = lab[static_cast<std::size_t>(y) * nx + (nx - 1)];
        if (a && b) uni(a, b);
    }

    // Component sizes by root; largest wins, smallest root on a tie.
    std::vector<int> size(n_lab + 1, 0);
    for (std::size_t k = 0; k < n; ++k) {
        if (lab[k]) ++size[find(lab[k])];
    }
    int big = 0, best = 0;
    for (int r = 1; r <= n_lab; ++r) {
        if (size[r] > best) { best = size[r]; big = r; }
    }
    for (std::size_t k = 0; k < n; ++k) {
        if (lab[k] && find(lab[k]) == big) out[k] = 1;
    }
    return out;
}

Real skyrmion_diameter_lcc(const Field3& m, Real a, int core_polarity) {
    if (!(a > 0.0)) {
        throw std::runtime_error("skyrmion_diameter_lcc: a must be > 0.");
    }
    const std::vector<std::uint8_t> mask = largest_core_mask_pbc(m, core_polarity);
    std::size_t n_inside = 0;
    for (std::uint8_t v : mask) n_inside += v;
    if (n_inside == 0) return 0.0;
    const Real area = static_cast<Real>(n_inside) * a * a;
    return 2.0 * std::sqrt(area / kPi);
}

Center2D skyrmion_center_lcc_pbc(const Field3& m, Real a, int core_polarity) {
    if (!(a > 0.0)) {
        throw std::runtime_error("skyrmion_center_lcc_pbc: a must be > 0.");
    }
    const int ny = m.ny, nx = m.nx;
    const std::vector<std::uint8_t> mask = largest_core_mask_pbc(m, core_polarity);
    std::size_t ws = 0;
    for (std::uint8_t v : mask) ws += v;
    if (ws == 0) {
        throw std::runtime_error(
            "skyrmion_center_lcc_pbc: empty core mask "
            "(skyrmion may have annihilated).");
    }
    const Real kx = 2.0 * kPi / nx;
    const Real ky = 2.0 * kPi / ny;
    Real Sx = 0.0, Cx = 0.0, Sy = 0.0, Cy = 0.0;
    for (int i = 0; i < ny; ++i) {
        const Real ty = ky * i;
        const Real sy = std::sin(ty), cyv = std::cos(ty);
        for (int j = 0; j < nx; ++j) {
            if (!mask[static_cast<std::size_t>(i) * nx + j]) continue;
            const Real tx = kx * j;
            Sx += std::sin(tx);
            Cx += std::cos(tx);
            Sy += sy;
            Cy += cyv;
        }
    }
    Real ang_x = std::atan2(Sx, Cx);
    if (ang_x < 0.0) ang_x += 2.0 * kPi;
    Real ang_y = std::atan2(Sy, Cy);
    if (ang_y < 0.0) ang_y += 2.0 * kPi;
    return {ang_x * nx * a / (2.0 * kPi), ang_y * ny * a / (2.0 * kPi)};
}

Ellipse skyrmion_ellipse_lcc(const Field3& m, Real a, int core_polarity) {
    if (!(a > 0.0)) {
        throw std::runtime_error("skyrmion_ellipse_lcc: a must be > 0.");
    }
    const int ny = m.ny, nx = m.nx;
    const std::vector<std::uint8_t> mask =
        largest_core_mask_pbc(m, core_polarity);
    std::size_t n_inside = 0;
    for (std::uint8_t v : mask) n_inside += v;
    if (n_inside < 3) {
        throw std::runtime_error(
            "skyrmion_ellipse_lcc: largest component has < 3 sites.");
    }
    // Circular-mean reference index per axis (the unwrap anchor) so a
    // component straddling a wrap collapses into one contiguous window.
    const Real kx = 2.0 * kPi / nx;
    const Real ky = 2.0 * kPi / ny;
    Real Sx = 0.0, Cx = 0.0, Sy = 0.0, Cy = 0.0;
    for (int i = 0; i < ny; ++i) {
        const Real ty = ky * i;
        const Real syv = std::sin(ty), cyv = std::cos(ty);
        for (int j = 0; j < nx; ++j) {
            if (!mask[static_cast<std::size_t>(i) * nx + j]) continue;
            Sx += std::sin(kx * j);
            Cx += std::cos(kx * j);
            Sy += syv;
            Cy += cyv;
        }
    }
    Real ang_x = std::atan2(Sx, Cx);
    if (ang_x < 0.0) ang_x += 2.0 * kPi;
    Real ang_y = std::atan2(Sy, Cy);
    if (ang_y < 0.0) ang_y += 2.0 * kPi;
    const Real ref_x = ang_x / (2.0 * kPi) * nx;
    const Real ref_y = ang_y / (2.0 * kPi) * ny;
    // Positive-modulo unwrap into [ref - N/2, ref + N/2), in meters.
    auto pmod = [](Real v, Real N) {
        Real r = std::fmod(v, N);
        if (r < 0.0) r += N;
        return r;
    };
    Real sx = 0.0, sy = 0.0, sxx = 0.0, syy = 0.0, sxy = 0.0;
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            if (!mask[static_cast<std::size_t>(i) * nx + j]) continue;
            const Real xu =
                (pmod(j - ref_x + 0.5 * nx, nx) - 0.5 * nx + ref_x) * a;
            const Real yu =
                (pmod(i - ref_y + 0.5 * ny, ny) - 0.5 * ny + ref_y) * a;
            sx += xu; sy += yu;
            sxx += xu * xu; syy += yu * yu; sxy += xu * yu;
        }
    }
    const Real inv = 1.0 / static_cast<Real>(n_inside);
    const Real mx = sx * inv, my = sy * inv;
    // Centered covariance entries.
    const Real cxx = sxx * inv - mx * mx;
    const Real cyy = syy * inv - my * my;
    const Real cxy = sxy * inv - mx * my;
    const Real tr = cxx + cyy;
    Real disc = 0.25 * (cxx - cyy) * (cxx - cyy) + cxy * cxy;
    if (disc < 0.0) disc = 0.0;
    const Real sqrt_disc = std::sqrt(disc);
    const Real lam1 = 0.5 * tr + sqrt_disc;
    const Real lam2 = 0.5 * tr - sqrt_disc;
    Real theta = 0.0;
    if (std::fabs(cxx - cyy) >= 1e-30 || std::fabs(cxy) >= 1e-30) {
        theta = 0.5 * std::atan2(2.0 * cxy, cxx - cyy);
    }
    Ellipse e;
    e.D1 = 4.0 * std::sqrt(lam1 > 0.0 ? lam1 : 0.0);
    e.D2 = 4.0 * std::sqrt(lam2 > 0.0 ? lam2 : 0.0);
    e.theta = theta;
    return e;
}

} // namespace stochastic
} // namespace skyrmion
