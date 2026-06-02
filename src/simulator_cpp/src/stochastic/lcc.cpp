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

} // namespace stochastic
} // namespace skyrmion
