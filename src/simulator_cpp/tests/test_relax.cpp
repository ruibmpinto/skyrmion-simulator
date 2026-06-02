// Parity tests for sweep::relax with a free-boundary mask and the
// single-layer mode. Mirrors phase_diagram.relaxation.relax(mask=...,
// m_bot=None). The short runs are chosen not to converge, so converged,
// n_steps, and the final state are all deterministic parity targets.
#include "skyrmion/demag.hpp"
#include "skyrmion/sweep/relax.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <vector>

using namespace skyrmion;
using skyrmion::sweep::relax;
using skyrmion::sweep::RelaxResult;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");

    const int max_steps =
        static_cast<int>(ref.scalar<int64_t>("relax_max_steps"));
    const double alpha = ref.scalar<double>("relax_alpha");
    const double tol_torque = ref.scalar<double>("relax_tol_torque");
    const double tol_dE = ref.scalar<double>("relax_tol_dE");
    const int check_every =
        static_cast<int>(ref.scalar<int64_t>("relax_check_every"));

    // ---- single-layer + mask (empty m_bot, no demag) ------------------------
    {
        RelaxResult res = relax(m_top, Field3(), p, /*demag=*/nullptr,
                                max_steps, alpha, tol_torque, tol_dE,
                                check_every, /*print_every=*/0, mask.data());
        Field3 exp_t = ref.field3("relax_single_mask_m_top");
        r.check("relax_single_mask_m_top",
                test_common::array(res.m_top.data.data(), exp_t.data.data(),
                                   res.m_top.data.size()));
        r.expect("relax_single_mask_converged",
                 res.converged ==
                     (ref.scalar<int64_t>("relax_single_mask_converged") != 0));
        r.expect("relax_single_mask_n_steps",
                 res.n_steps == static_cast<int>(
                     ref.scalar<int64_t>("relax_single_mask_n_steps")));
        r.check("relax_single_mask_tau_max",
                test_common::scalar(
                    res.tau_max,
                    ref.scalar<double>("relax_single_mask_tau_max")));
    }
    // ---- pair + mask (slab demag) -------------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState demag(p, 0);
        RelaxResult res = relax(m_top, m_bot, p, &demag,
                                max_steps, alpha, tol_torque, tol_dE,
                                check_every, /*print_every=*/0, mask.data());
        Field3 exp_t = ref.field3("relax_pair_mask_m_top");
        Field3 exp_b = ref.field3("relax_pair_mask_m_bot");
        r.check("relax_pair_mask_m_top",
                test_common::array(res.m_top.data.data(), exp_t.data.data(),
                                   res.m_top.data.size()));
        r.check("relax_pair_mask_m_bot",
                test_common::array(res.m_bot.data.data(), exp_b.data.data(),
                                   res.m_bot.data.size()));
        r.expect("relax_pair_mask_converged",
                 res.converged ==
                     (ref.scalar<int64_t>("relax_pair_mask_converged") != 0));
        r.expect("relax_pair_mask_n_steps",
                 res.n_steps == static_cast<int>(
                     ref.scalar<int64_t>("relax_pair_mask_n_steps")));
        r.check("relax_pair_mask_tau_max",
                test_common::scalar(
                    res.tau_max,
                    ref.scalar<double>("relax_pair_mask_tau_max")));
        p.demag_kind = DemagKind::None;
    }
    // ---- single-layer + demag must raise ------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState demag(p, 0);
        r.expect_throws("relax_single_with_demag_raises", [&] {
            relax(m_top, Field3(), p, &demag, 10, alpha, tol_torque, tol_dE,
                  check_every, 0, mask.data());
        });
        p.demag_kind = DemagKind::None;
    }
    return r.report("test_relax");
}
