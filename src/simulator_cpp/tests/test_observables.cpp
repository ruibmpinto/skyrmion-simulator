// Parity tests for topological_charge, skyrmion_center, skyrmion_diameter,
// skyrmion_ellipse, dw_angle.
#include "skyrmion/observables.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    const double a = ref.scalar<double>("state_a");

    r.check("topological_charge_top",
            test_common::scalar(topological_charge(m_top, a),
                           ref.scalar<double>("observables_Q_top")));
    r.check("topological_charge_bot",
            test_common::scalar(topological_charge(m_bot, a),
                           ref.scalar<double>("observables_Q_bot")));

    Center2D c = skyrmion_center(m_top, a, +1);
    r.check("skyrmion_center_cx",
            test_common::scalar(c.cx, ref.scalar<double>("observables_cx_top")));
    r.check("skyrmion_center_cy",
            test_common::scalar(c.cy, ref.scalar<double>("observables_cy_top")));

    r.check("skyrmion_diameter",
            test_common::scalar(skyrmion_diameter(m_top, a, +1),
                           ref.scalar<double>("observables_diameter_top")));

    Ellipse e = skyrmion_ellipse(m_top, a, +1);
    r.check("skyrmion_ellipse_D1",
            test_common::scalar(e.D1, ref.scalar<double>("observables_D1_top")));
    r.check("skyrmion_ellipse_D2",
            test_common::scalar(e.D2, ref.scalar<double>("observables_D2_top")));
    r.check("skyrmion_ellipse_theta",
            test_common::scalar(e.theta, ref.scalar<double>("observables_theta_top")));

    r.check("dw_angle",
            test_common::scalar(dw_angle(m_top, a, +1, 0.5),
                           ref.scalar<double>("observables_psi_top")));
    return r.report("test_observables");
}
