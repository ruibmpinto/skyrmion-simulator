// Parity test for energy.total_energy with both demag kernels.
#include "skyrmion/demag.hpp"
#include "skyrmion/energy.hpp"
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

    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        const double E = total_energy(m_top, m_bot, p, s);
        r.check("total_energy_slab",
                test_common::scalar(E, ref.scalar<double>("energy_E_slab")));
    }
    {
        p.demag_kind = DemagKind::Newell;
        p.demag_accuracy = 8.0;
        p.demag_tol_conv = 2.0e-2;
        DemagState s(p, 0);
        const double E = total_energy(m_top, m_bot, p, s);
        r.check("total_energy_newell",
                test_common::scalar(E, ref.scalar<double>("energy_E_newell")));
    }
    return r.report("test_energy");
}
