// End-to-end parity: 50 relax steps (J=0) + 20 drive steps with the
// local-K_eff RHS. Compares m_top and m_bot after each phase to the
// Python reference state stored in references.npz.
#include "skyrmion/integrator.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "test_common.hpp"

#include <memory>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    const int n_relax = static_cast<int>(ref.scalar<int64_t>("simulation_n_relax"));
    const int n_steps = static_cast<int>(ref.scalar<int64_t>("simulation_n_steps"));

    // Phase 0: relax (J=0)
    const auto pulse_save = p.pulse;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    {
        RHSLocalKeff rhs(p, /*mask=*/nullptr);
        double t = 0.0;
        for (int s = 0; s < n_relax; ++s) {
            rk4_step(rhs, m_top, m_bot, t, p.dt, p);
            t += p.dt;
        }
    }
    Field3 exp_t = ref.field3("simulation_m_top_after_relax");
    Field3 exp_b = ref.field3("simulation_m_bot_after_relax");
    r.check("after_relax_top",
            test_common::array(m_top.data.data(), exp_t.data.data(), m_top.data.size()));
    r.check("after_relax_bot",
            test_common::array(m_bot.data.data(), exp_b.data.data(), m_bot.data.size()));

    // Phase 1: drive
    p.pulse = pulse_save;
    {
        RHSLocalKeff rhs(p, /*mask=*/nullptr);
        double t = 0.0;
        for (int s = 0; s < n_steps; ++s) {
            rk4_step(rhs, m_top, m_bot, t, p.dt, p);
            t += p.dt;
        }
    }
    Field3 exp_td = ref.field3("simulation_m_top_after_drive");
    Field3 exp_bd = ref.field3("simulation_m_bot_after_drive");
    r.check("after_drive_top",
            test_common::array(m_top.data.data(), exp_td.data.data(), m_top.data.size()));
    r.check("after_drive_bot",
            test_common::array(m_bot.data.data(), exp_bd.data.data(), m_bot.data.size()));

    return r.report("test_simulation");
}
