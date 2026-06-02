// Parity test for the two-skyrmion pair initial condition.
//
// skyrmion_at_position (the ported per-site Neel profile) is checked for
// bit-level parity against Python. The merged pair field is checked
// site-by-site at every grid point where the core-favouring m_z pick is
// unambiguous; the measure-zero seam where |t1_z - t2_z| is at the ULP
// level is excluded, because the np.where(<=) tie-break there is decided
// by a cross-library (numpy vs libm) 1-ULP difference and cannot be made
// bit-reproducible. Those seam sites are counted and bounded explicitly.
#include "skyrmion/stochastic/pair_ic.hpp"
#include "test_common.hpp"

#include <cmath>
#include <cstdio>

using namespace skyrmion;
using test_common::TestRef;
using test_common::TestRunner;

namespace {

// Compare merged field to reference at every site whose pick is decided by
// a m_z gap larger than `tie_eps`; tally ambiguous seam sites separately.
bool check_merge(const char* name, const Field3& got, const Field3& exp,
                 const Field3& f1, const Field3& f2, double tie_eps,
                 int& ambiguous) {
    const std::size_t n = got.data.size();
    double max_diff = 0.0;
    ambiguous = 0;
    for (int i = 0; i < got.ny; ++i) {
        for (int j = 0; j < got.nx; ++j) {
            if (std::abs(f1(i, j, 2) - f2(i, j, 2)) <= tie_eps) {
                ++ambiguous;
                continue;
            }
            for (int k = 0; k < 3; ++k) {
                const double d = std::abs(got(i, j, k) - exp(i, j, k));
                if (d > max_diff) max_diff = d;
            }
        }
    }
    const bool ok = max_diff <= 1e-12;
    std::printf("  %-40s diff=%.3e ambiguous=%d  %s\n",
                name, max_diff, ambiguous, ok ? "OK" : "FAIL");
    (void)n;
    return ok;
}

} // namespace

int main() {
    TestRef ref(REFERENCE_NPZ);
    TestRunner r;
    const int nx = static_cast<int>(ref.scalar<int64_t>("pair_ic_nx"));
    const int ny = static_cast<int>(ref.scalar<int64_t>("pair_ic_ny"));
    const double a = ref.scalar<double>("state_a");
    const double R = ref.scalar<double>("pair_ic_R");
    const double dw = ref.scalar<double>("pair_ic_dw");
    const double c1x = ref.scalar<double>("pair_ic_c1x");
    const double c1y = ref.scalar<double>("pair_ic_c1y");
    const double c2x = ref.scalar<double>("pair_ic_c2x");
    const double c2y = ref.scalar<double>("pair_ic_c2y");

    // (1) Single off-centre skyrmion: bit-exact parity.
    Field3 single = stochastic::skyrmion_at_position(
        nx, ny, a, R, dw, +1, c1x, c1y);
    Field3 exp_single = ref.field3("pair_ic_single_m");
    r.check("pair_ic_single_m",
            test_common::array(single.data.data(), exp_single.data.data(),
                               single.data.size()));

    // (2) Merge: exact at every unambiguously-picked site.
    SAFPair pair = stochastic::two_skyrmion_pair_ic(
        nx, ny, a, R, dw, +1, c1x, c1y, c2x, c2y);
    Field3 t1 = stochastic::skyrmion_at_position(nx, ny, a, R, dw, +1,
                                                 c1x, c1y);
    Field3 t2 = stochastic::skyrmion_at_position(nx, ny, a, R, dw, +1,
                                                 c2x, c2y);
    Field3 b1 = stochastic::skyrmion_at_position(nx, ny, a, R, dw, -1,
                                                 c1x, c1y);
    Field3 b2 = stochastic::skyrmion_at_position(nx, ny, a, R, dw, -1,
                                                 c2x, c2y);
    Field3 exp_t = ref.field3("pair_ic_m_top");
    Field3 exp_b = ref.field3("pair_ic_m_bot");

    int amb_top = 0, amb_bot = 0;
    bool ok_top = check_merge("pair_ic_m_top", pair.m_top, exp_t, t1, t2,
                              1e-12, amb_top);
    bool ok_bot = check_merge("pair_ic_m_bot", pair.m_bot, exp_b, b1, b2,
                              1e-12, amb_bot);
    // The seam is a thin curve; a 64x64 box must not be mostly ambiguous.
    const int n_sites = nx * ny;
    if (amb_top > n_sites / 20 || amb_bot > n_sites / 20) {
        std::printf("  FAIL: too many ambiguous seam sites "
                    "(top=%d bot=%d of %d)\n", amb_top, amb_bot, n_sites);
        ok_top = ok_bot = false;
    }
    r.expect("pair_ic_merge_top_unambiguous", ok_top);
    r.expect("pair_ic_merge_bot_unambiguous", ok_bot);

    return r.report("test_pair_ic");
}
