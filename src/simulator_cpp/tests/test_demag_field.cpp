// Parity test for DemagState::compute (the runtime demag field
// evaluation). Run against both slab and Newell kernels.
#include "skyrmion/demag.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");

    // ---- slab kernel --------------------------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        s.compute(m_top, m_bot, H_top, H_bot);
        Field3 exp_t = ref.field3("demag_field_slab_H_top");
        Field3 exp_b = ref.field3("demag_field_slab_H_bot");
        r.check("demag_field_slab_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("demag_field_slab_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
    }
    // ---- Newell kernel ------------------------------------------------------
    {
        p.demag_kind = DemagKind::Newell;
        p.demag_accuracy = 8.0;
        p.demag_tol_conv = 2.0e-2;
        DemagState s(p, 0);
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        s.compute(m_top, m_bot, H_top, H_bot);
        Field3 exp_t = ref.field3("demag_field_newell_H_top");
        Field3 exp_b = ref.field3("demag_field_newell_H_bot");
        r.check("demag_field_newell_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("demag_field_newell_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
    }
    return r.report("test_demag_field");
}
