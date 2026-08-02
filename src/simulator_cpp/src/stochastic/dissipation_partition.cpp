#include "skyrmion/stochastic/dissipation_partition.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

namespace {

// Central x-difference (periodic) and x/free y-difference of one lattice
// vector component, returned as the pair (d/dx, d/dy) divided by the
// spacing. y uses one-sided differences at the edges when free_y.
inline void grad_at(const Field3& m, int i, int j, int k, Real inv_2a,
                    Real inv_a, bool free_y, Real& gx, Real& gy) {
    const int ny = m.ny, nx = m.nx;
    const int jp = (j + 1) % nx;
    const int jm = (j - 1 + nx) % nx;
    gx = (m(i, jp, k) - m(i, jm, k)) * inv_2a;
    if (free_y) {
        if (i == 0) {
            gy = (m(1, j, k) - m(0, j, k)) * inv_a;
        } else if (i == ny - 1) {
            gy = (m(ny - 1, j, k) - m(ny - 2, j, k)) * inv_a;
        } else {
            gy = (m(i + 1, j, k) - m(i - 1, j, k)) * inv_2a;
        }
    } else {
        const int ip = (i + 1) % ny;
        const int im = (i - 1 + ny) % ny;
        gy = (m(ip, j, k) - m(im, j, k)) * inv_2a;
    }
}

}  // namespace

DissipationSplit dissipation_partition(const Field3& m, const Field3& dmdt,
                                       Real a, bool free_y) {
    if (m.ny != dmdt.ny || m.nx != dmdt.nx) {
        throw std::runtime_error(
            "dissipation_partition: m and dmdt shapes differ.");
    }
    if (!(a > 0.0)) {
        throw std::runtime_error(
            "dissipation_partition: lattice constant a must be positive.");
    }
    const Real inv_2a = 1.0 / (2.0 * a);
    const Real inv_a = 1.0 / a;
    // Normal-equation accumulators for the 2x2 least-squares fit of the
    // velocity that best explains dm/dt as -(v . grad) m, plus the total
    // |dm/dt|^2 needed for the split.
    double s_xx = 0.0, s_xy = 0.0, s_yy = 0.0;
    double b_x = 0.0, b_y = 0.0;
    double total = 0.0;
    for (int i = 0; i < m.ny; ++i) {
        for (int j = 0; j < m.nx; ++j) {
            for (int k = 0; k < 3; ++k) {
                Real gx, gy;
                grad_at(m, i, j, k, inv_2a, inv_a, free_y, gx, gy);
                const Real d = dmdt(i, j, k);
                s_xx += gx * gx;
                s_xy += gx * gy;
                s_yy += gy * gy;
                // Fit dm/dt ~ -(vx gx + vy gy): minimise
                // |dm/dt + vx gx + vy gy|^2, so the RHS carries -gx.d.
                b_x += -gx * d;
                b_y += -gy * d;
                total += d * d;
            }
        }
    }
    DissipationSplit out;
    out.total = total * a * a;
    const double det = s_xx * s_yy - s_xy * s_xy;
    if (!(std::abs(det) > 0.0)) {
        // No gradient (uniform field): nothing translates, so all of the
        // rate, if any, is deformation.
        out.def = out.total;
        return out;
    }
    const double vx = (s_yy * b_x - s_xy * b_y) / det;
    const double vy = (s_xx * b_y - s_xy * b_x) / det;
    out.v_x = vx;
    out.v_y = vy;
    // Second pass: accumulate the two orthogonal parts directly rather
    // than deriving def from the residual sum, so round-off in the fit
    // does not push either part negative.
    double p_trans = 0.0, p_def = 0.0;
    for (int i = 0; i < m.ny; ++i) {
        for (int j = 0; j < m.nx; ++j) {
            for (int k = 0; k < 3; ++k) {
                Real gx, gy;
                grad_at(m, i, j, k, inv_2a, inv_a, free_y, gx, gy);
                const Real trans = vx * gx + vy * gy;  // = -(v.grad)m sign
                const Real def = dmdt(i, j, k) + trans;
                p_trans += trans * trans;
                p_def += def * def;
            }
        }
    }
    out.trans = p_trans * a * a;
    out.def = p_def * a * a;
    return out;
}

} // namespace stochastic
} // namespace skyrmion
