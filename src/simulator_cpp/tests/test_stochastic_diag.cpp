// Parity tests for the stochastic trajectory diagnostics:
// unwrap_trajectory, hall_angle, detect_annihilation, against Python on
// a synthetic moving-and-wrapping centre history.
#include "skyrmion/stochastic/diagnostics.hpp"
#include "test_common.hpp"

#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    auto t = ref.vec<double>("diag_t");
    auto cx = ref.vec<double>("diag_cx_wrap");
    auto cy = ref.vec<double>("diag_cy_wrap");
    const double Lx = ref.scalar<double>("diag_L_x");
    const double Ly = ref.scalar<double>("diag_L_y");

    // ---- unwrap_trajectory --------------------------------------------------
    auto u = stochastic::unwrap_trajectory(cx, cy, Lx, Ly, true);
    auto ex = ref.vec<double>("diag_cx_unwrap");
    auto ey = ref.vec<double>("diag_cy_unwrap");
    r.check("unwrap_cx", test_common::array(u.cx.data(), ex.data(), u.cx.size()));
    r.check("unwrap_cy", test_common::array(u.cy.data(), ey.data(), u.cy.size()));

    // Free-y (racetrack) mode: y must pass through untouched.
    auto uf = stochastic::unwrap_trajectory(cx, cy, Lx, Ly, false);
    bool y_untouched = true;
    for (std::size_t i = 0; i < cy.size(); ++i) {
        if (uf.cy[i] != cy[i]) y_untouched = false;
    }
    r.expect("unwrap_free_y_untouched", y_untouched);

    // ---- hall_angle ---------------------------------------------------------
    stochastic::HallFit f = stochastic::hall_angle(t, u.cx, u.cy, 0.5);
    r.check("hall_v_x", scalar(f.v_x, ref.scalar<double>("diag_v_x")));
    r.check("hall_v_y", scalar(f.v_y, ref.scalar<double>("diag_v_y")));
    r.check("hall_deg", scalar(f.theta_deg, ref.scalar<double>("diag_hall_deg")));

    // ---- detect_annihilation ------------------------------------------------
    auto Q = ref.vec<double>("diag_Q_hist");
    const int flip = stochastic::detect_annihilation(
        Q, ref.scalar<double>("diag_q_threshold"),
        static_cast<int>(ref.scalar<int64_t>("diag_k_consecutive")));
    r.expect("detect_annihilation_flip_index",
             flip == static_cast<int>(ref.scalar<int64_t>("diag_flip_index")));

    // ---- explicit-raise contract --------------------------------------------
    r.expect_throws("unwrap_bad_L",
                    [&] { stochastic::unwrap_trajectory(cx, cy, -1.0, Ly, true); });
    r.expect_throws("hall_half_out_of_range",
                    [&] { stochastic::hall_angle(t, u.cx, u.cy, 1.5); });
    r.expect_throws("detect_bad_threshold",
                    [&] { stochastic::detect_annihilation(Q, 0.0, 10); });

    return r.report("test_stochastic_diag");
}
