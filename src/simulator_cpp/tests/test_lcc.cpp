// Parity tests for the LCC diagnostics: largest_core_mask_pbc (exact
// boolean-mask match vs scipy.ndimage.label + PBC fusion),
// skyrmion_diameter_lcc, skyrmion_center_lcc_pbc. Includes a
// half-box-rolled state that straddles both wraps to exercise the
// periodic union-find fusion.
#include "skyrmion/stochastic/lcc.hpp"
#include "test_common.hpp"

#include <cstdint>
#include <vector>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;
using test_common::scalar;

namespace {

// Count element mismatches between a C++ uint8 mask and the reference
// (loaded as int64 by libnpy-friendly tensor read).
std::size_t mask_mismatches(const std::vector<std::uint8_t>& got,
                            TestRef& ref, const std::string& key) {
    auto t = ref.tensor<std::uint8_t>(key);
    if (t.size() != got.size()) return got.size() + t.size();  // shape error
    std::size_t bad = 0;
    for (std::size_t k = 0; k < got.size(); ++k) {
        if (got[k] != t.data()[k]) ++bad;
    }
    return bad;
}

} // namespace

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    const double a = ref.scalar<double>("state_a");
    Field3 m_top = ref.field3("state_m_top");
    Field3 m_bot = ref.field3("state_m_bot");
    Field3 m_rolled = ref.field3("lcc_rolled_m");

    // ---- exact mask parity --------------------------------------------------
    auto mask_top = stochastic::largest_core_mask_pbc(m_top, +1);
    auto mask_bot = stochastic::largest_core_mask_pbc(m_bot, -1);
    auto mask_rolled = stochastic::largest_core_mask_pbc(m_rolled, +1);
    r.expect("lcc_mask_top_exact",
             mask_mismatches(mask_top, ref, "lcc_mask_top") == 0);
    r.expect("lcc_mask_bot_exact",
             mask_mismatches(mask_bot, ref, "lcc_mask_bot") == 0);
    r.expect("lcc_mask_rolled_exact (PBC fusion)",
             mask_mismatches(mask_rolled, ref, "lcc_mask_rolled") == 0);

    // ---- diameter / centre parity -------------------------------------------
    r.check("lcc_diam_top",
            scalar(stochastic::skyrmion_diameter_lcc(m_top, a, +1),
                   ref.scalar<double>("lcc_diam_top")));
    Center2D c = stochastic::skyrmion_center_lcc_pbc(m_top, a, +1);
    r.check("lcc_cx_top", scalar(c.cx, ref.scalar<double>("lcc_cx_top")));
    r.check("lcc_cy_top", scalar(c.cy, ref.scalar<double>("lcc_cy_top")));

    r.check("lcc_diam_rolled",
            scalar(stochastic::skyrmion_diameter_lcc(m_rolled, a, +1),
                   ref.scalar<double>("lcc_diam_rolled")));
    Center2D cr = stochastic::skyrmion_center_lcc_pbc(m_rolled, a, +1);
    r.check("lcc_cx_rolled", scalar(cr.cx, ref.scalar<double>("lcc_cx_rolled")));
    r.check("lcc_cy_rolled", scalar(cr.cy, ref.scalar<double>("lcc_cy_rolled")));

    // ---- explicit-raise contract --------------------------------------------
    r.expect_throws("lcc_mask_bad_polarity",
                    [&] { stochastic::largest_core_mask_pbc(m_top, 0); });

    return r.report("test_lcc");
}
