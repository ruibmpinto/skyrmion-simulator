// Parity tests for parameters.precompute and the derived helpers
// (bare_anis_prefactors, effective_anisotropy, critical_dmi,
// pma_anisotropy_field).
#include "skyrmion/parameters.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    Params p = test_common::build_params_from_ref(ref);

    // precompute (called inside build_params_from_ref).
    r.check("precompute_C_ex",
            test_common::scalar(p.C_ex, ref.scalar<double>("precompute_C_ex")));
    r.check("precompute_C_dmi",
            test_common::scalar(p.C_dmi, ref.scalar<double>("precompute_C_dmi")));
    r.check("precompute_C_anis_top",
            test_common::scalar(p.C_anis_top, ref.scalar<double>("precompute_C_anis_top")));
    r.check("precompute_C_anis_bot",
            test_common::scalar(p.C_anis_bot, ref.scalar<double>("precompute_C_anis_bot")));
    r.check("precompute_H_DL",
            test_common::scalar(p.H_DL, ref.scalar<double>("precompute_H_DL")));
    r.check("precompute_H_FL",
            test_common::scalar(p.H_FL, ref.scalar<double>("precompute_H_FL")));
    r.check("precompute_gamma_p",
            test_common::scalar(p.gamma_p, ref.scalar<double>("precompute_gamma_p")));

    // bare_anis_prefactors
    BareAnis ba = bare_anis_prefactors(p);
    r.check("bare_anis_C_top",
            test_common::scalar(ba.C_top, ref.scalar<double>("precompute_bare_C_top")));
    r.check("bare_anis_C_bot",
            test_common::scalar(ba.C_bot, ref.scalar<double>("precompute_bare_C_bot")));

    // effective_anisotropy
    EffectiveAnis ea = effective_anisotropy(p);
    r.check("eff_anis_K_eff_top",
            test_common::scalar(ea.K_eff_top, ref.scalar<double>("precompute_K_eff_top")));
    r.check("eff_anis_K_eff_bot",
            test_common::scalar(ea.K_eff_bot, ref.scalar<double>("precompute_K_eff_bot")));
    r.check("eff_anis_K_eff_avg",
            test_common::scalar(ea.K_eff_avg, ref.scalar<double>("precompute_K_eff_avg")));
    r.check("eff_anis_mu0_Ms2_over_2",
            test_common::scalar(ea.mu0_Ms2_over_2, ref.scalar<double>("precompute_mu0_Ms2_over_2")));

    r.check("critical_dmi",
            test_common::scalar(critical_dmi(p), ref.scalar<double>("precompute_D_c")));
    r.check("pma_anisotropy_field",
            test_common::scalar(pma_anisotropy_field(p),
                           ref.scalar<double>("precompute_H_K")));

    return r.report("test_parameters");
}
