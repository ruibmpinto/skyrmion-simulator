// Parity tests for skyrmion_profile, uniform_state, saf_skyrmion.
#include "skyrmion/initial_conditions.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    const int nx = static_cast<int>(ref.scalar<int64_t>("state_nx"));
    const int ny = static_cast<int>(ref.scalar<int64_t>("state_ny"));
    const double a  = ref.scalar<double>("state_a");
    const double R  = ref.scalar<double>("state_R");
    const double dw = ref.scalar<double>("state_dw");

    // skyrmion_profile polarity = +1
    {
        Field3 m = skyrmion_profile(nx, ny, a, R, dw, +1);
        Field3 exp = ref.field3("init_skyrmion_pol_plus");
        r.check("skyrmion_profile_polarity_plus_1",
                test_common::array(m.data.data(), exp.data.data(), m.data.size()));
    }
    // skyrmion_profile polarity = -1
    {
        Field3 m = skyrmion_profile(nx, ny, a, R, dw, -1);
        Field3 exp = ref.field3("init_skyrmion_pol_minus");
        r.check("skyrmion_profile_polarity_minus_1",
                test_common::array(m.data.data(), exp.data.data(), m.data.size()));
    }
    // uniform_state
    {
        auto dir = ref.vec<double>("init_uniform_dir");
        Field3 m = uniform_state(nx, ny, {dir[0], dir[1], dir[2]});
        Field3 exp = ref.field3("init_uniform");
        r.check("uniform_state",
                test_common::array(m.data.data(), exp.data.data(), m.data.size()));
    }
    // saf_skyrmion
    {
        SAFPair pair = saf_skyrmion(nx, ny, a, R, dw);
        Field3 exp_t = ref.field3("init_saf_top");
        Field3 exp_b = ref.field3("init_saf_bot");
        r.check("saf_skyrmion_top",
                test_common::array(pair.m_top.data.data(), exp_t.data.data(),
                                         pair.m_top.data.size()));
        r.check("saf_skyrmion_bot",
                test_common::array(pair.m_bot.data.data(), exp_b.data.data(),
                                         pair.m_bot.data.size()));
    }
    return r.report("test_initial_conditions");
}
