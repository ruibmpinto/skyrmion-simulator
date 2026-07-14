// Parity test for energy.total_energy with both demag kernels.
#include "skyrmion/demag.hpp"
#include "skyrmion/energy.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    Params p = test_common::build_params_from_ref(ref);
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");

    {
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        const double E = total_energy(m_top, m_bot, p, s, nullptr);
        r.check("total_energy_slab",
                test_common::scalar(E, ref.scalar<double>("energy_E_slab")));
    }
    {
        p.demag_kind = DemagKind::Newell;
        p.demag_accuracy = 8.0;
        p.demag_tol_conv = 2.0e-2;
        DemagState s(p, 0);
        const double E = total_energy(m_top, m_bot, p, s, nullptr);
        r.check("total_energy_newell",
                test_common::scalar(E, ref.scalar<double>("energy_E_newell")));
    }
    {
        // Masked energy: vacuum cells excluded, mask-zeroed demag source.
        p.demag_kind = DemagKind::Slab;
        DemagState s(p, 0);
        const std::vector<std::uint8_t> msk =
            ref.vec<std::uint8_t>("mask_disk");
        const double E = total_energy(m_top, m_bot, p, s, msk.data());
        r.check("total_energy_slab_mask",
                test_common::scalar(
                    E, ref.scalar<double>("energy_E_slab_mask")));
    }
    {
        // Racetrack energy: free-y exchange/DMI + free-y demag.
        p.demag_kind = DemagKind::Racetrack;
        p.demag_accuracy = 8.0;
        p.demag_tol_conv = 2.0e-2;
        DemagState s(p, 0);
        const double E = total_energy(m_top, m_bot, p, s, nullptr);
        r.check("total_energy_racetrack",
                test_common::scalar(
                    E, ref.scalar<double>("energy_E_racetrack")));
    }
    return r.report("test_energy");
}
