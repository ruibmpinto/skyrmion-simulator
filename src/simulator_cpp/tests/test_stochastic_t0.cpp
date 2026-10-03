// T=0 bit-parity gate for the Stratonovich-Heun stepper. At sigma = 0
// the stepper draws no noise and reduces to a deterministic RK2, so it
// must match the Python heun_stochastic_step (zero noise) to machine
// precision, for both the local-K_eff and slab-demag field models.
#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "test_common.hpp"

#include <memory>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    Params p = test_common::build_params_from_ref(ref);
    precompute(p);

    const int N = static_cast<int>(ref.scalar<int64_t>("heun_t0_N"));
    Field3 m_top0 = ref.field3("state_m_top");
    Field3 m_bot0 = ref.field3("state_m_bot");
    stochastic::ThermalRng rng(12345);   // unused at sigma = 0

    // ---- local-K_eff path (demag = nullptr) ---------------------------------
    {
        Field3 mt = m_top0, mb = m_bot0;
        stochastic::HeunStochasticStepper stepper(
            p, /*demag=*/nullptr, rng, /*sigma=*/0.0, /*tol_norm=*/1.0,
            /*mask=*/nullptr);
        Real t = 0.0;
        for (int s = 0; s < N; ++s) {
            stepper.step(mt, mb, t, p.dt, p);
            t += p.dt;
        }
        Field3 exp_t = ref.field3("heun_t0_nodemag_m_top");
        Field3 exp_b = ref.field3("heun_t0_nodemag_m_bot");
        r.check("heun_t0_nodemag_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
        r.check("heun_t0_nodemag_bot",
                test_common::array(mb.data.data(), exp_b.data.data(), mb.data.size()));
    }
    // ---- slab-demag path ----------------------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState demag(p, 0);
        Field3 mt = m_top0, mb = m_bot0;
        stochastic::HeunStochasticStepper stepper(
            p, &demag, rng, /*sigma=*/0.0, /*tol_norm=*/1.0,
            /*mask=*/nullptr);
        Real t = 0.0;
        for (int s = 0; s < N; ++s) {
            stepper.step(mt, mb, t, p.dt, p);
            t += p.dt;
        }
        Field3 exp_t = ref.field3("heun_t0_slab_m_top");
        Field3 exp_b = ref.field3("heun_t0_slab_m_bot");
        r.check("heun_t0_slab_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
        r.check("heun_t0_slab_bot",
                test_common::array(mb.data.data(), exp_b.data.data(), mb.data.size()));
    }
    // ---- single-layer + free-boundary mask (sigma = 0) ---------------------
    {
        std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");
        Field3 mt = m_top0;
        Field3 empty;  // n_sites() == 0 selects single-layer mode
        stochastic::HeunStochasticStepper stepper(
            p, /*demag=*/nullptr, rng, /*sigma=*/0.0, /*tol_norm=*/1.0,
            mask.data());
        Real t = 0.0;
        for (int s = 0; s < N; ++s) {
            stepper.step(mt, empty, t, p.dt, p);
            t += p.dt;
        }
        Field3 exp_t = ref.field3("heun_single_mask_m_top");
        r.check("heun_single_mask_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
    }
    // ---- pair + free-boundary mask (slab, sigma = 0) ------------------------
    {
        std::vector<std::uint8_t> mask = ref.vec<std::uint8_t>("mask_disk");
        p.demag_kind = DemagKind::Slab;
        DemagState demag(p, 0);
        Field3 mt = m_top0, mb = m_bot0;
        stochastic::HeunStochasticStepper stepper(
            p, &demag, rng, /*sigma=*/0.0, /*tol_norm=*/1.0, mask.data());
        Real t = 0.0;
        for (int s = 0; s < N; ++s) {
            stepper.step(mt, mb, t, p.dt, p);
            t += p.dt;
        }
        Field3 exp_t = ref.field3("heun_pair_mask_m_top");
        Field3 exp_b = ref.field3("heun_pair_mask_m_bot");
        r.check("heun_pair_mask_top",
                test_common::array(mt.data.data(), exp_t.data.data(), mt.data.size()));
        r.check("heun_pair_mask_bot",
                test_common::array(mb.data.data(), exp_b.data.data(), mb.data.size()));
        p.demag_kind = DemagKind::None;
    }
    // ---- single-layer + demag must raise ------------------------------------
    {
        p.demag_kind = DemagKind::Slab;
        DemagState demag(p, 0);
        stochastic::HeunStochasticStepper stepper(
            p, &demag, rng, /*sigma=*/0.0, /*tol_norm=*/1.0,
            /*mask=*/nullptr);
        Field3 mt = m_top0;
        Field3 empty;
        r.expect_throws("heun_single_with_demag_raises", [&] {
            stepper.step(mt, empty, 0.0, p.dt, p);
        });
        p.demag_kind = DemagKind::None;
    }
    return r.report("test_stochastic_t0");
}
