// Parity tests for normalize_inplace, llgs_rhs, RHSLocalKeff, RHSDemag,
// and rk4_step.
#include "skyrmion/demag.hpp"
#include "skyrmion/fields.hpp"
#include "skyrmion/integrator.hpp"
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

    // ---- normalize_inplace --------------------------------------------------
    {
        Field3 mp = ref.field3("state_m_perturbed");
        normalize_inplace(mp);
        Field3 exp = ref.field3("integrator_normalized");
        r.check("normalize_inplace",
                test_common::array(mp.data.data(), exp.data.data(), mp.data.size()));
    }
    // ---- llgs_rhs ------------------------------------------------------------
    {
        // Build effective fields with the local-K_eff path; then evaluate
        // llgs_rhs on each layer at t = 0.
        Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
        effective_field(m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top, nullptr, /*free_y=*/false);
        effective_field(m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY, H_bot, nullptr, /*free_y=*/false);
        Field3 dt_top(p.ny, p.nx), dt_bot(p.ny, p.nx);
        const double t = ref.scalar<double>("integrator_t_eval");
        llgs_rhs(m_top, H_top, p, t, dt_top);
        llgs_rhs(m_bot, H_bot, p, t, dt_bot);
        Field3 exp_t = ref.field3("integrator_dmdt_top_keff");
        Field3 exp_b = ref.field3("integrator_dmdt_bot_keff");
        r.check("llgs_rhs_top",
                test_common::array(dt_top.data.data(), exp_t.data.data(), dt_top.data.size()));
        r.check("llgs_rhs_bot",
                test_common::array(dt_bot.data.data(), exp_b.data.data(), dt_bot.data.size()));
    }
    // ---- RHSLocalKeff -------------------------------------------------------
    {
        RHSLocalKeff rhs(p, nullptr);
        Field3 dt_top(p.ny, p.nx), dt_bot(p.ny, p.nx);
        rhs(m_top, m_bot, 0.0, dt_top, dt_bot);
        Field3 exp_t = ref.field3("integrator_rhs_keff_top");
        Field3 exp_b = ref.field3("integrator_rhs_keff_bot");
        r.check("RHSLocalKeff_top",
                test_common::array(dt_top.data.data(), exp_t.data.data(), dt_top.data.size()));
        r.check("RHSLocalKeff_bot",
                test_common::array(dt_bot.data.data(), exp_b.data.data(), dt_bot.data.size()));
    }
    // ---- RHSDemag (slab) ----------------------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        RHSDemag rhs(p, s, nullptr);
        Field3 dt_top(p.ny, p.nx), dt_bot(p.ny, p.nx);
        rhs(m_top, m_bot, 0.0, dt_top, dt_bot);
        Field3 exp_t = ref.field3("integrator_rhs_demag_top");
        Field3 exp_b = ref.field3("integrator_rhs_demag_bot");
        r.check("RHSDemag_top",
                test_common::array(dt_top.data.data(), exp_t.data.data(), dt_top.data.size()));
        r.check("RHSDemag_bot",
                test_common::array(dt_bot.data.data(), exp_b.data.data(), dt_bot.data.size()));
        p.demag_kind = DemagKind::None;  // restore for rk4 below
    }
    // ---- rk4_step (local-K_eff) ---------------------------------------------
    {
        Field3 mt = m_top;  // deep copy
        Field3 mb = m_bot;
        RHSLocalKeff rhs(p, nullptr);
        rk4_step(rhs, mt, mb, 0.0, p.dt, p);
        Field3 exp_t = ref.field3("integrator_rk4_keff_top");
        Field3 exp_b = ref.field3("integrator_rk4_keff_bot");
        r.check("rk4_step_keff_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
        r.check("rk4_step_keff_bot",
                test_common::array(mb.data.data(), exp_b.data.data(), mb.data.size()));
    }
    // ---- rk4_step (slab demag) ----------------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        RHSDemag rhs(p, s, nullptr);
        Field3 mt = m_top;
        Field3 mb = m_bot;
        rk4_step(rhs, mt, mb, 0.0, p.dt, p);
        Field3 exp_t = ref.field3("integrator_rk4_demag_top");
        Field3 exp_b = ref.field3("integrator_rk4_demag_bot");
        r.check("rk4_step_demag_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
        r.check("rk4_step_demag_bot",
                test_common::array(mb.data.data(), exp_b.data.data(), mb.data.size()));
    }
    // ---- rk4_step_single (single layer, local-K_eff) ------------------------
    {
        Field3 m = m_top;  // deep copy
        RHSSingleKeff rhs(p, nullptr);
        rk4_step_single(rhs, m, 0.0, p.dt, p);
        Field3 exp = ref.field3("integrator_rk4_single");
        r.check("rk4_step_single",
                test_common::array(m.data.data(), exp.data.data(), m.data.size()));
    }
    // ---- rk4_step_single + free-boundary mask -------------------------------
    {
        std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");
        Field3 m = m_top;  // deep copy
        RHSSingleKeff rhs(p, mask.data());
        rk4_step_single(rhs, m, 0.0, p.dt, p);
        Field3 exp = ref.field3("integrator_rk4_single_mask");
        r.check("rk4_step_single_mask",
                test_common::array(m.data.data(), exp.data.data(), m.data.size()));
    }
    return r.report("test_integrator");
}
