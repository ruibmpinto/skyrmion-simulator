// Parity tests for the validation-gate analytic targets: Langevin
// function, Neel-Brown reversal time, and the discrete magnon stiffness
// grid. These are closed forms and bit-checkable against Python
// (src/skyrmion_simulator/stochastic_llgs/validation/*.py).
#include "skyrmion/validation/analytic.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    // ---- Langevin function ----------------------------------------------
    const std::vector<double> lx = ref.vec<double>("val_langevin_x");
    const std::vector<double> lL = ref.vec<double>("val_langevin_L");
    std::vector<double> cppL(lx.size());
    for (std::size_t i = 0; i < lx.size(); ++i)
        cppL[i] = validation::langevin_function(lx[i]);
    r.check("val_langevin_L",
            test_common::array(cppL.data(), lL.data(), lL.size()));

    // ---- Brown reversal time --------------------------------------------
    const double b_alpha = ref.scalar<double>("val_brown_alpha");
    const double b_gamma = ref.scalar<double>("val_brown_gamma");
    const std::vector<double> bd = ref.vec<double>("val_brown_delta");
    const std::vector<double> bt = ref.vec<double>("val_brown_tau");
    std::vector<double> cppT(bd.size());
    for (std::size_t i = 0; i < bd.size(); ++i)
        cppT[i] = validation::brown_tau(b_alpha, b_gamma, bd[i]);
    r.check("val_brown_tau",
            test_common::array(cppT.data(), bt.data(), bt.size()));

    // ---- Magnon stiffness grid ------------------------------------------
    const int mny = static_cast<int>(ref.scalar<int64_t>("val_magnon_ny"));
    const int mnx = static_cast<int>(ref.scalar<int64_t>("val_magnon_nx"));
    const double mBz = ref.scalar<double>("val_magnon_Bz");
    const double mCex = ref.scalar<double>("val_magnon_Cex");
    const std::vector<double> Hk_ref = ref.vec<double>("val_magnon_Hk");
    const std::vector<Real> Hk = validation::magnon_stiffness_grid(
        mny, mnx, mBz, mCex);
    r.check("val_magnon_Hk",
            test_common::array(Hk.data(), Hk_ref.data(), Hk_ref.size()));

    return r.report("test_validation");
}
