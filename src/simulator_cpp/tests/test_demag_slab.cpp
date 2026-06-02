// Parity test for the slab demag kernel (precompute_demag_slab).
#include "skyrmion/demag.hpp"
#include "test_common.hpp"

#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::Complex;

namespace {

test_common::Diff max_rel(const std::vector<Complex>& a,
               const npy::tensor<Complex>& b) {
    return test_common::array_complex(a.data(), b.data(), a.size());
}

} // namespace

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);

    DemagKernels K = precompute_demag_slab(p);

    auto compare = [&](const char* name, const std::vector<Complex>& cpp_kern) {
        auto py = ref.tensor<Complex>(std::string("demag_slab_") + name);
        r.check(std::string("demag_slab_") + name, max_rel(cpp_kern, py));
    };
    compare("Nxx_self",  K.Nxx_self);
    compare("Nyy_self",  K.Nyy_self);
    compare("Nxy_self",  K.Nxy_self);
    compare("Nzz_self",  K.Nzz_self);
    compare("Nxx_inter", K.Nxx_inter);
    compare("Nyy_inter", K.Nyy_inter);
    compare("Nxy_inter", K.Nxy_inter);
    compare("Nzz_inter", K.Nzz_inter);
    compare("Nxz_inter", K.Nxz_inter);
    compare("Nyz_inter", K.Nyz_inter);
    return r.report("test_demag_slab");
}
