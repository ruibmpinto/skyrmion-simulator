// Unit tests for the translation/deformation split of dm/dt.
//
// The split is exercised in its two exactly-known limits, built from
// synthetic fields so no dynamics are involved:
//
//   rigid translation : dm/dt = -(v . grad) m for a chosen v. The fit
//                        must recover v and put all of the rate in the
//                        translational part.
//   pure breathing    : dm/dt is a radial (dilation) field with no net
//                        translation. The fit must return v ~ 0 and put
//                        all of the rate in deformation.
//
// A third check confirms the parts sum to the total (orthogonality).
#include "skyrmion/stochastic/dissipation_partition.hpp"
#include "skyrmion/types.hpp"
#include "test_common.hpp"

#include <cmath>

using namespace skyrmion;
using namespace skyrmion::stochastic;
using test_common::TestRunner;

namespace {

// A Bloch-like skyrmion profile centred at (cx, cy), used only as a
// smooth, localised texture with non-trivial gradients.
Field3 skyrmion_like(int ny, int nx, Real a, Real cx, Real cy, Real R) {
    Field3 m(ny, nx);
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real x = j * a - cx;
            const Real y = i * a - cy;
            const Real r = std::sqrt(x * x + y * y);
            const Real theta = 2.0 * std::atan2(R, r + 1e-30);  // pi at r=0
            const Real phi = std::atan2(y, x);
            m(i, j, 0) = std::sin(theta) * (-std::sin(phi));
            m(i, j, 1) = std::sin(theta) * (std::cos(phi));
            m(i, j, 2) = std::cos(theta);
        }
    }
    return m;
}

// Central-difference gradient, matching the partition's stencil, used to
// synthesise dm/dt = -(v.grad)m for the rigid-translation limit.
Field3 minus_v_dot_grad(const Field3& m, Real a, Real vx, Real vy,
                        bool free_y) {
    Field3 out(m.ny, m.nx);
    const Real inv_2a = 1.0 / (2.0 * a);
    const Real inv_a = 1.0 / a;
    for (int i = 0; i < m.ny; ++i) {
        for (int j = 0; j < m.nx; ++j) {
            const int jp = (j + 1) % m.nx;
            const int jm = (j - 1 + m.nx) % m.nx;
            for (int k = 0; k < 3; ++k) {
                const Real gx = (m(i, jp, k) - m(i, jm, k)) * inv_2a;
                Real gy;
                if (free_y) {
                    if (i == 0)
                        gy = (m(1, j, k) - m(0, j, k)) * inv_a;
                    else if (i == m.ny - 1)
                        gy = (m(m.ny - 1, j, k) - m(m.ny - 2, j, k)) * inv_a;
                    else
                        gy = (m(i + 1, j, k) - m(i - 1, j, k)) * inv_2a;
                } else {
                    const int ip = (i + 1) % m.ny;
                    const int im = (i - 1 + m.ny) % m.ny;
                    gy = (m(ip, j, k) - m(im, j, k)) * inv_2a;
                }
                out(i, j, k) = -(vx * gx + vy * gy);
            }
        }
    }
    return out;
}

}  // namespace

int main() {
    TestRunner r;
    const int ny = 64, nx = 96;
    const Real a = 2.0e-9;
    const Real cx = 0.5 * nx * a, cy = 0.5 * ny * a, R = 20.0 * a;
    Field3 m = skyrmion_like(ny, nx, a, cx, cy, R);

    // ---- Rigid translation ------------------------------------------------
    {
        const Real vx = 120.0, vy = -35.0;
        Field3 dmdt = minus_v_dot_grad(m, a, vx, vy, /*free_y=*/false);
        DissipationSplit s = dissipation_partition(m, dmdt, a, false);
        r.expect("rigid_recovers_vx", std::abs(s.v_x - vx) < 1e-6 * 120.0);
        r.expect("rigid_recovers_vy", std::abs(s.v_y - vy) < 1e-6 * 120.0);
        const double frac_def = s.def / s.total;
        r.expect("rigid_def_fraction_zero", frac_def < 1e-12);
        r.expect("rigid_sum_equals_total",
                 std::abs(s.trans + s.def - s.total)
                 < 1e-9 * s.total);
    }
    // ---- Pure breathing (radial dilation, no translation) -----------------
    {
        // dm/dt = (r . grad) m is a dilation: it changes the texture's
        // size, not its position, so it is orthogonal to both gradient
        // directions in the least-squares sense only on average. Build it
        // and require the fitted velocity to be negligible and the rate
        // to be almost entirely deformation.
        Field3 dmdt(ny, nx);
        const Real inv_2a = 1.0 / (2.0 * a);
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                const Real x = j * a - cx;
                const Real y = i * a - cy;
                const int jp = (j + 1) % nx, jm = (j - 1 + nx) % nx;
                const int ip = (i + 1) % ny, im = (i - 1 + ny) % ny;
                for (int k = 0; k < 3; ++k) {
                    const Real gx = (m(i, jp, k) - m(i, jm, k)) * inv_2a;
                    const Real gy = (m(ip, j, k) - m(im, j, k)) * inv_2a;
                    dmdt(i, j, k) = x * gx + y * gy;  // radial dilation
                }
            }
        }
        DissipationSplit s = dissipation_partition(m, dmdt, a, false);
        const double speed = std::sqrt(s.v_x * s.v_x + s.v_y * s.v_y);
        // A symmetric texture dilated about its centre carries no net
        // translation; the fitted speed is tiny next to the dilation rate.
        r.expect("breathing_velocity_small", speed < 1.0);
        const double frac_trans = s.trans / s.total;
        r.expect("breathing_trans_fraction_small", frac_trans < 1e-3);
    }
    // ---- Free-y stencil runs and stays orthogonal -------------------------
    {
        Field3 dmdt = minus_v_dot_grad(m, a, 80.0, 10.0, /*free_y=*/true);
        DissipationSplit s = dissipation_partition(m, dmdt, a, true);
        r.expect("freey_sum_equals_total",
                 std::abs(s.trans + s.def - s.total) < 1e-9 * s.total);
        r.expect("freey_def_fraction_zero", s.def / s.total < 1e-12);
    }
    return r.report("test_dissipation_partition");
}
