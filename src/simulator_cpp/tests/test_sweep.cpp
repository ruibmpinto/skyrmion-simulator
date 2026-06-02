// Parity tests for the sweep-layer building blocks against Python:
// skyrmion_center_pbc and sweep::observe_state (the 16-scalar payload).
#include "skyrmion/observables.hpp"
#include "skyrmion/sweep/observations.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    const double a = ref.scalar<double>("state_a");

    // ---- skyrmion_center_pbc ------------------------------------------------
    Center2D ct = skyrmion_center_pbc(m_top, a, +1);
    Center2D cb = skyrmion_center_pbc(m_bot, a, -1);
    r.check("center_pbc_cx_top",
            scalar(ct.cx, ref.scalar<double>("observables_cx_top_pbc")));
    r.check("center_pbc_cy_top",
            scalar(ct.cy, ref.scalar<double>("observables_cy_top_pbc")));
    r.check("center_pbc_cx_bot",
            scalar(cb.cx, ref.scalar<double>("observables_cx_bot_pbc")));
    r.check("center_pbc_cy_bot",
            scalar(cb.cy, ref.scalar<double>("observables_cy_bot_pbc")));

    // ---- observe_state (16 scalars) -----------------------------------------
    sweep::Observations o = sweep::observe_state(m_top, m_bot, p);
    r.check("observe_cx_top", scalar(o.cx_top, ref.scalar<double>("observe_cx_top")));
    r.check("observe_cy_top", scalar(o.cy_top, ref.scalar<double>("observe_cy_top")));
    r.check("observe_cx_bot", scalar(o.cx_bot, ref.scalar<double>("observe_cx_bot")));
    r.check("observe_cy_bot", scalar(o.cy_bot, ref.scalar<double>("observe_cy_bot")));
    r.check("observe_d_top",  scalar(o.d_top,  ref.scalar<double>("observe_d_top")));
    r.check("observe_d_bot",  scalar(o.d_bot,  ref.scalar<double>("observe_d_bot")));
    r.check("observe_D1_top", scalar(o.D1_top, ref.scalar<double>("observe_D1_top")));
    r.check("observe_D2_top", scalar(o.D2_top, ref.scalar<double>("observe_D2_top")));
    r.check("observe_theta_top", scalar(o.theta_top, ref.scalar<double>("observe_theta_top")));
    r.check("observe_D1_bot", scalar(o.D1_bot, ref.scalar<double>("observe_D1_bot")));
    r.check("observe_D2_bot", scalar(o.D2_bot, ref.scalar<double>("observe_D2_bot")));
    r.check("observe_theta_bot", scalar(o.theta_bot, ref.scalar<double>("observe_theta_bot")));
    r.check("observe_psi_top", scalar(o.psi_top, ref.scalar<double>("observe_psi_top")));
    r.check("observe_psi_bot", scalar(o.psi_bot, ref.scalar<double>("observe_psi_bot")));
    r.check("observe_Q_top",  scalar(o.Q_top,  ref.scalar<double>("observe_Q_top")));
    r.check("observe_Q_bot",  scalar(o.Q_bot,  ref.scalar<double>("observe_Q_bot")));

    return r.report("test_sweep");
}
