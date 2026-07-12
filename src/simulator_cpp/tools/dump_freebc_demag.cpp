// Cross-validation dump for the free-BC (zero-padded) Newell demag.
// Builds the 2N kernel and the demag field for a seeded SAF skyrmion on a
// small box, and writes the k-space kernel spectra + the input field +
// the output field to NPZ. scripts/compare_freebc_demag.py recomputes the
// same with the Python newell_freebc path and asserts they match.
#include "skyrmion/demag.hpp"
#include "skyrmion/fields.hpp"
#include "skyrmion/initial_conditions.hpp"
#include "skyrmion/parameters.hpp"

#include <npy/npy.h>

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

npy::tensor<double> re2d(const std::vector<Complex>& v, int ny, int nx) {
    npy::tensor<double> t({static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)});
    std::vector<double> r(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) r[k] = v[k].real();
    t.copy_from(r.data(), r.size());
    return t;
}

npy::tensor<double> im2d(const std::vector<Complex>& v, int ny, int nx) {
    npy::tensor<double> t({static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)});
    std::vector<double> r(v.size());
    for (std::size_t k = 0; k < v.size(); ++k) r[k] = v[k].imag();
    t.copy_from(r.data(), r.size());
    return t;
}

npy::tensor<double> field3(const Field3& m) {
    npy::tensor<double> t({static_cast<std::size_t>(m.ny),
                           static_cast<std::size_t>(m.nx), 3});
    t.copy_from(m.data.data(), m.data.size());
    return t;
}

}  // namespace

int main() {
    // Small non-square box for a fast, rectangular-sensitive check.
    Params p = make_default_params();
    p.nx = 24; p.ny = 20; p.a = 2.0e-9;
    p.demag_accuracy = 8.0;   // matches the periodic-newell parity test
    p.demag_tol_conv = 0.05;
    precompute(p);

    SAFPair ic = saf_skyrmion(p.nx, p.ny, p.a, 12.0e-9, 6.0e-9);

    // Free-BC (zero-padded) demag field + its 2N kernel.
    p.demag_kind = DemagKind::NewellFreeBC;
    const DemagKernels K = precompute_demag_newell_freebc(
        p, p.demag_accuracy, p.demag_tol_conv);
    DemagState demag(p, 0);
    Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
    demag.compute(ic.m_top, ic.m_bot, H_top, H_bot);

    // Periodic newell field on the SAME input, as the same-grid baseline
    // (both C++ and Python use the closed-form tensor, so periodic and
    // freebc should each match Python to FFT round-off).
    p.demag_kind = DemagKind::Newell;
    DemagState demag_per(p, 0);
    Field3 H_top_per(p.ny, p.nx), H_bot_per(p.ny, p.nx);
    demag_per.compute(ic.m_top, ic.m_bot, H_top_per, H_bot_per);

    // Mixed track BC (periodic x, free y) on the SAME input, for the
    // racetrack parity check against Python.
    p.demag_kind = DemagKind::Racetrack;
    DemagState demag_y(p, 0);
    Field3 H_top_y(p.ny, p.nx), H_bot_y(p.ny, p.nx);
    demag_y.compute(ic.m_top, ic.m_bot, H_top_y, H_bot_y);

    // Complete racetrack field: free-y demag PLUS free-y exchange/DMI
    // (the bare-K local terms with the R-T free edge derived from
    // demag_kind == Racetrack). Cross-validated against Python
    // effective_field_demag_pair with kind='racetrack'.
    Field3 Hf_top_y(p.ny, p.nx), Hf_bot_y(p.ny, p.nx);
    effective_field_demag(ic.m_top, ic.m_bot, p, demag_y,
                          Hf_top_y, Hf_bot_y, nullptr);

    const std::string out = "output/stochastic_llgs/validation/"
                            "demag_freebc_cpp.npz";
    std::filesystem::create_directories(
        "output/stochastic_llgs/validation");
    npy::npzfilewriter w(out);
    // Physical + kernel grid sizes.
    auto sc = [](double v) {
        npy::tensor<double> t(std::vector<std::size_t>{});
        t.copy_from(&v, 1); return t; };
    w.write("nx", sc(p.nx)); w.write("ny", sc(p.ny));
    w.write("a", sc(p.a));
    w.write("ker_ny", sc(K.ny)); w.write("ker_nx", sc(K.nx));
    // k-space kernel spectra (2N grid), real+imag per component.
    const int kny = K.ny, knx = K.nx;
    w.write("Nxx_self_re", re2d(K.Nxx_self, kny, knx));
    w.write("Nxx_self_im", im2d(K.Nxx_self, kny, knx));
    w.write("Nzz_self_re", re2d(K.Nzz_self, kny, knx));
    w.write("Nxy_self_re", re2d(K.Nxy_self, kny, knx));
    w.write("Nzz_inter_re", re2d(K.Nzz_inter, kny, knx));
    w.write("Nxz_inter_re", re2d(K.Nxz_inter, kny, knx));
    // Input field and demag output (physical size).
    w.write("m_top", field3(ic.m_top));
    w.write("m_bot", field3(ic.m_bot));
    w.write("H_top", field3(H_top));
    w.write("H_bot", field3(H_bot));
    w.write("H_top_per", field3(H_top_per));
    w.write("H_bot_per", field3(H_bot_per));
    w.write("H_top_y", field3(H_top_y));
    w.write("H_bot_y", field3(H_bot_y));
    // Full racetrack effective field (free-y demag + free-y exchange/DMI).
    w.write("Hfull_top_y", field3(Hf_top_y));
    w.write("Hfull_bot_y", field3(Hf_bot_y));
    std::printf("wrote %s (box %dx%d, kernel %dx%d)\n",
                out.c_str(), p.nx, p.ny, K.nx, K.ny);
    return 0;
}
