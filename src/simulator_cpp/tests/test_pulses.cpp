// Parity tests for ConstantPulse, SquarePulse, GaussianPulse,
// SuperpositionPulse against the Python implementation.
#include "skyrmion/pulses.hpp"
#include "test_common.hpp"

#include <memory>
#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    auto t_grid = ref.vec<double>("pulses_t_grid");
    const std::size_t T = t_grid.size();

    // ---- ConstantPulse ------------------------------------------------------
    {
        ConstantPulse p(ref.scalar<double>("pulses_const_J0"));
        std::vector<double> got(T);
        for (std::size_t i = 0; i < T; ++i) got[i] = p(t_grid[i]);
        auto exp = ref.vec<double>("pulses_const_out");
        r.check("ConstantPulse", test_common::array(got.data(), exp.data(), T));
    }
    // ---- SquarePulse --------------------------------------------------------
    {
        SquarePulse p(ref.scalar<double>("pulses_square_J0"),
                      ref.scalar<double>("pulses_square_t_start"),
                      ref.scalar<double>("pulses_square_t_end"));
        std::vector<double> got(T);
        for (std::size_t i = 0; i < T; ++i) got[i] = p(t_grid[i]);
        auto exp = ref.vec<double>("pulses_square_out");
        r.check("SquarePulse", test_common::array(got.data(), exp.data(), T));
    }
    // ---- GaussianPulse ------------------------------------------------------
    {
        GaussianPulse p(ref.scalar<double>("pulses_gauss_J0"),
                        ref.scalar<double>("pulses_gauss_t_center"),
                        ref.scalar<double>("pulses_gauss_fwhm"));
        std::vector<double> got(T);
        for (std::size_t i = 0; i < T; ++i) got[i] = p(t_grid[i]);
        auto exp = ref.vec<double>("pulses_gauss_out");
        r.check("GaussianPulse", test_common::array(got.data(), exp.data(), T));
    }
    // ---- SuperpositionPulse -------------------------------------------------
    {
        std::vector<std::unique_ptr<Pulse>> pulses;
        pulses.push_back(std::make_unique<SquarePulse>(
            ref.scalar<double>("pulses_square_J0"),
            ref.scalar<double>("pulses_square_t_start"),
            ref.scalar<double>("pulses_square_t_end")));
        pulses.push_back(std::make_unique<GaussianPulse>(
            ref.scalar<double>("pulses_gauss_J0"),
            ref.scalar<double>("pulses_gauss_t_center"),
            ref.scalar<double>("pulses_gauss_fwhm")));
        SuperpositionPulse sp(std::move(pulses));
        std::vector<double> got(T);
        for (std::size_t i = 0; i < T; ++i) got[i] = sp(t_grid[i]);
        auto exp = ref.vec<double>("pulses_super_out");
        r.check("SuperpositionPulse",
                test_common::array(got.data(), exp.data(), T));
    }
    return r.report("test_pulses");
}
