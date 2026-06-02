// Parity test for lattice.lattice_positions.
#include "skyrmion/lattice.hpp"
#include "test_common.hpp"

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;

    const int nx = static_cast<int>(ref.scalar<int64_t>("state_nx"));
    const int ny = static_cast<int>(ref.scalar<int64_t>("state_ny"));
    const double a = ref.scalar<double>("state_a");
    Field3 pos = lattice_positions(nx, ny, a);
    Field3 exp = ref.field3("lattice_pos");
    r.check("lattice_positions",
            test_common::array(pos.data.data(), exp.data.data(), pos.data.size()));
    return r.report("test_lattice");
}
