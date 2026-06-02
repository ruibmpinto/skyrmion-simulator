// Parity tests for the thermal-parameter layer: attach_thermal
// (sigma_noise, V_cell) and T_of_j against Python.
#include "skyrmion/stochastic/thermal.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    Params p = test_common::build_params_from_ref(ref);
    stochastic::attach_thermal(p, ref.scalar<double>("thermal_T"), 0.0, 12345);
    r.check("attach_thermal_sigma_noise",
            scalar(p.sigma_noise, ref.scalar<double>("thermal_sigma_noise")));
    r.check("attach_thermal_V_cell",
            scalar(p.V_cell, ref.scalar<double>("thermal_V_cell")));

    const double Tj = stochastic::T_of_j(4.0e11, 300.0, 1.0e-20);
    r.check("T_of_j", scalar(Tj, ref.scalar<double>("thermal_T_of_j")));

    // Explicit-raise contract.
    r.expect_throws("attach_thermal_negative_T",
                    [&] { Params q = p; stochastic::attach_thermal(q, -1.0, 0.0, 0); });
    r.expect_throws("T_of_j_nonpositive_Tsub",
                    [] { (void)stochastic::T_of_j(1.0, 0.0, 0.0); });

    return r.report("test_thermal");
}
