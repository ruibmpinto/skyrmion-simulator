// Edge-case + explicit-raise contract tests. Verifies that every C++
// function raises loudly on invalid input (no silent fallbacks) and
// that boundary values match the Python contract. Python and C++ now
// raise on the same invalid inputs (the former skyrmion_center /
// uniform_state / normalize NaN divergences have been aligned).
#include "skyrmion/demag.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/integrator.hpp"
#include "skyrmion/io_npz.hpp"
#include "skyrmion/observables.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "test_common.hpp"

#include <cmath>
#include <memory>
#include <vector>

using namespace skyrmion;
using test_common::TestRunner;
using test_common::Diff;

namespace {

Field3 uniform_field(int ny, int nx, Real mx, Real my, Real mz) {
    Field3 m(ny, nx);
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            m(i, j, 0) = mx; m(i, j, 1) = my; m(i, j, 2) = mz;
        }
    return m;
}

} // namespace

int main() {
    TestRunner r;
    constexpr Real kPi = 3.14159265358979323846;

    // ===== Pulses: boundary values ==========================================
    {
        ConstantPulse c(3.0);
        r.expect("ConstantPulse_const", c(-1e9) == 3.0 && c(1e9) == 3.0);

        SquarePulse sq(2.0, 0.0, 1.0e-9);
        r.expect("SquarePulse_at_t_start_inclusive", sq(0.0) == 2.0);
        r.expect("SquarePulse_at_t_end_inclusive", sq(1.0e-9) == 2.0);
        r.expect("SquarePulse_before_window", sq(-1.0e-12) == 0.0);
        r.expect("SquarePulse_after_window", sq(1.001e-9) == 0.0);

        GaussianPulse g(5.0, 1.0e-9, 5.0e-10);
        r.expect("GaussianPulse_peak_at_center",
                 std::abs(g(1.0e-9) - 5.0) < 1e-12);
        r.expect("GaussianPulse_decays_off_center", g(2.0e-9) < 5.0);
    }
    // ===== Pulses: constructor raises =======================================
    r.expect_throws("SquarePulse_t_end_le_t_start",
                    [] { SquarePulse(1.0, 1.0, 1.0); });
    r.expect_throws("SquarePulse_inverted_window",
                    [] { SquarePulse(1.0, 1.0e-9, 0.0); });
    r.expect_throws("GaussianPulse_zero_FWHM",
                    [] { GaussianPulse(1.0, 0.0, 0.0); });
    r.expect_throws("GaussianPulse_negative_FWHM",
                    [] { GaussianPulse(1.0, 0.0, -1.0e-9); });
    r.expect_throws("SuperpositionPulse_empty", [] {
        std::vector<std::unique_ptr<Pulse>> empty;
        SuperpositionPulse(std::move(empty));
    });

    // ===== Parameters: K_eff <= 0 raises ====================================
    r.expect_throws("critical_dmi_Keff_nonpositive", [] {
        Params p = make_default_params();
        p.K_top = 1.0e3; p.K_bot = 1.0e3;   // << 0.5 mu0 Ms^2 ~ 1.3e6
        precompute(p);
        (void)critical_dmi(p);
    });
    r.expect_throws("pma_field_Keff_nonpositive", [] {
        Params p = make_default_params();
        p.K_top = 1.0e3; p.K_bot = 1.0e3;
        precompute(p);
        (void)pma_anisotropy_field(p);
    });
    r.expect_no_throw("critical_dmi_normal", [] {
        Params p = make_default_params();
        (void)critical_dmi(p);
    });

    // ===== Initial conditions: polarity + zero-direction raises =============
    r.expect_throws("skyrmion_profile_polarity_0",
                    [] { skyrmion_profile(8, 8, 2e-9, 6e-9, 3e-9, 0); });
    r.expect_throws("skyrmion_profile_polarity_2",
                    [] { skyrmion_profile(8, 8, 2e-9, 6e-9, 3e-9, 2); });
    r.expect_throws("uniform_state_zero_dir",
                    [] { uniform_state(8, 8, Vec3{0.0, 0.0, 0.0}); });
    r.expect_no_throw("uniform_state_normal",
                      [] { uniform_state(8, 8, Vec3{0.0, 0.0, 1.0}); });

    // ===== Observables: polarity validation =================================
    {
        Field3 m = uniform_field(8, 8, 0.0, 0.0, 1.0);
        r.expect_throws("skyrmion_center_polarity_0",
                        [&] { skyrmion_center(m, 2e-9, 0); });
        r.expect_throws("skyrmion_diameter_polarity_0",
                        [&] { skyrmion_diameter(m, 2e-9, 0); });
        r.expect_throws("skyrmion_ellipse_polarity_0",
                        [&] { skyrmion_ellipse(m, 2e-9, 0); });
        r.expect_throws("skyrmion_center_pbc_polarity_0",
                        [&] { skyrmion_center_pbc(m, 2e-9, 0); });
        r.expect_throws("skyrmion_center_pbc_a_nonpositive",
                        [&] { skyrmion_center_pbc(m, 0.0, +1); });
    }
    // ===== Observables: degenerate textures raise ===========================
    {
        // Uniform m_z=+1, polarity +1 -> core weight is zero everywhere.
        Field3 up = uniform_field(8, 8, 0.0, 0.0, 1.0);
        r.expect_throws("skyrmion_center_empty_mask",
                        [&] { skyrmion_center(up, 2e-9, +1); });
        r.expect_throws("skyrmion_center_pbc_zero_weight",
                        [&] { skyrmion_center_pbc(up, 2e-9, +1); });
        r.expect_throws("skyrmion_ellipse_too_few_core_sites",
                        [&] { skyrmion_ellipse(up, 2e-9, +1); });
        r.expect_throws("dw_angle_no_domain_wall",
                        [&] { dw_angle(up, 2e-9, +1, 0.5); });
        // topological_charge of a uniform field is ~0 (no raise).
        const Real Q = topological_charge(up, 2e-9);
        r.expect("topological_charge_uniform_is_zero", std::abs(Q) < 1e-12);
    }
    // ===== Observables: skyrmion_diameter zero core (no raise, d=0) =========
    {
        Field3 up = uniform_field(8, 8, 0.0, 0.0, 1.0);
        const Real d = skyrmion_diameter(up, 2e-9, +1);  // empty core mask
        r.expect("skyrmion_diameter_empty_core_is_zero", d == 0.0);
    }

    // ===== Demag: parameter validation raises ===============================
    r.expect_throws("demag_slab_tCo_nonpositive", [] {
        Params p = make_default_params();
        p.nx = 8; p.ny = 8; p.t_Co = -1.0e-9; precompute(p);
        (void)precompute_demag_slab(p);
    });
    r.expect_throws("demag_slab_dRu_negative", [] {
        Params p = make_default_params();
        p.nx = 8; p.ny = 8; p.d_Ru = -1.0e-9; precompute(p);
        (void)precompute_demag_slab(p);
    });
    r.expect_throws("demag_newell_accuracy_nonpositive", [] {
        Params p = make_default_params(); p.nx = 8; p.ny = 8; precompute(p);
        (void)precompute_demag_newell(p, 0.0, 0.02);
    });
    r.expect_throws("demag_newell_tol_ge_one", [] {
        Params p = make_default_params(); p.nx = 8; p.ny = 8; precompute(p);
        (void)precompute_demag_newell(p, 4.0, 1.0);
    });
    r.expect_throws("demag_newell_tol_nonpositive", [] {
        Params p = make_default_params(); p.nx = 8; p.ny = 8; precompute(p);
        (void)precompute_demag_newell(p, 4.0, 0.0);
    });
    r.expect_throws("demag_newell_nx_too_small", [] {
        Params p = make_default_params(); p.nx = 1; p.ny = 8; precompute(p);
        (void)precompute_demag_newell(p, 4.0, 0.02);
    });
    r.expect_throws("demag_dispatch_kind_None", [] {
        Params p = make_default_params();
        p.nx = 8; p.ny = 8; p.demag_kind = DemagKind::None; precompute(p);
        (void)precompute_demag_kernels(p);
    });

    // ===== SnapshotBuffer: explicit raises ==================================
    r.expect_throws("snapshot_buffer_bad_size",
                    [] { SnapshotBuffer(0, 8, 4); });
    r.expect_throws("snapshot_buffer_overflow", [] {
        SnapshotBuffer buf(4, 4, 2);
        Field3 m(4, 4);
        for (int k = 0; k < 3; ++k)  // capacity 2, 3rd append must raise
            buf.append(m, m, k, 0.0, 1, 0,0,0,0,0,0,0,0,0);
    });
    r.expect_throws("snapshot_buffer_shape_mismatch", [] {
        SnapshotBuffer buf(4, 4, 2);
        Field3 wrong(8, 8);
        buf.append(wrong, wrong, 0, 0.0, 1, 0,0,0,0,0,0,0,0,0);
    });

    // ===== normalize_inplace: restores |m| = 1 ==============================
    {
        Field3 m = uniform_field(8, 8, 3.0, 0.0, 4.0);  // |m| = 5
        normalize_inplace(m);
        Real worst = 0.0;
        for (int i = 0; i < 8; ++i)
            for (int j = 0; j < 8; ++j) {
                const Real n = std::sqrt(m(i, j, 0) * m(i, j, 0)
                                         + m(i, j, 1) * m(i, j, 1)
                                         + m(i, j, 2) * m(i, j, 2));
                worst = std::max(worst, std::abs(n - 1.0));
            }
        r.expect("normalize_inplace_unit_norm", worst < 1e-15);
    }
    // normalize_inplace raises on a zero-magnitude spin (matches Python
    // normalize, which now raises rather than dividing -> NaN).
    r.expect_throws("normalize_inplace_zero_vector", [] {
        Field3 z = uniform_field(4, 4, 0.0, 0.0, 0.0);
        normalize_inplace(z);
    });

    // ===== Skyrmion at lattice center is well-formed (no raise) =============
    r.expect_no_throw("skyrmion_profile_normal_well_formed", [&] {
        Field3 m = skyrmion_profile(32, 32, 2e-9, 12e-9, 6e-9, +1);
        const Real Q = topological_charge(m, 2e-9);
        if (std::abs(std::abs(Q) - 1.0) > 0.2)
            throw std::runtime_error("Q not near +/-1");
        (void)kPi;
    });

    return r.report("test_edge_cases");
}
