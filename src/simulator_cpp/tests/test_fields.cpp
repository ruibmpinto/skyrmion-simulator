// Parity tests for fields.effective_field and effective_field_demag.
#include "skyrmion/demag.hpp"
#include "skyrmion/fields.hpp"
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

    // ---- effective_field (local-K_eff) --------------------------------------
    {
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        effective_field(m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top, nullptr, /*free_y=*/false);
        effective_field(m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY, H_bot, nullptr, /*free_y=*/false);
        Field3 exp_t = ref.field3("fields_H_top_keff");
        Field3 exp_b = ref.field3("fields_H_bot_keff");
        r.check("effective_field_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("effective_field_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
    }
    // ---- effective_field_demag (slab kernel) --------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        effective_field_demag(m_top, m_bot, p, s, H_top, H_bot, nullptr);
        Field3 exp_t = ref.field3("fields_H_top_demag");
        Field3 exp_b = ref.field3("fields_H_bot_demag");
        r.check("effective_field_demag_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("effective_field_demag_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
        p.demag_kind = DemagKind::None;
    }
    // ---- effective_field + free-boundary mask (local-K_eff) -----------------
    {
        std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        effective_field(m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top, mask.data(), /*free_y=*/false);
        effective_field(m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY, H_bot, mask.data(), /*free_y=*/false);
        Field3 exp_t = ref.field3("fields_H_top_keff_mask");
        Field3 exp_b = ref.field3("fields_H_bot_keff_mask");
        r.check("effective_field_mask_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("effective_field_mask_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
    }
    // ---- effective_field_demag + free-boundary mask (slab kernel) -----------
    {
        std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        effective_field_demag(m_top, m_bot, p, s, H_top, H_bot, mask.data());
        Field3 exp_t = ref.field3("fields_H_top_demag_mask");
        Field3 exp_b = ref.field3("fields_H_bot_demag_mask");
        r.check("effective_field_demag_mask_top",
                test_common::array(H_top.data.data(), exp_t.data.data(), H_top.data.size()));
        r.check("effective_field_demag_mask_bot",
                test_common::array(H_bot.data.data(), exp_b.data.data(), H_bot.data.size()));
        p.demag_kind = DemagKind::None;
    }
    return r.report("test_fields");
}
