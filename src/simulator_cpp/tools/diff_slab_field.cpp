// Dump my C++ slab demag field for inspection in Python.
#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/initial_conditions.hpp"

#include <npy/npy.h>

#include <cstdio>

using namespace skyrmion;

int main() {
    Params p = make_default_params();
    p.nx = 32;
    p.ny = 32;
    p.a = 2.0e-9;
    p.demag_kind = DemagKind::Slab;
    precompute(p);

    SAFPair sp = saf_skyrmion(p.nx, p.ny, p.a, 30.0e-9, 15.0e-9);
    Field3 m_top = std::move(sp.m_top);
    Field3 m_bot = std::move(sp.m_bot);

    DemagState ds(p, 0);
    Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
    ds.compute(m_top, m_bot, H_top, H_bot);

    npy::npzfilewriter w("diff_slab_field.npz");
    npy::tensor<double> t_top({static_cast<std::size_t>(p.ny),
                               static_cast<std::size_t>(p.nx),
                               3});
    t_top.copy_from(H_top.data.data(), H_top.data.size());
    npy::tensor<double> t_bot({static_cast<std::size_t>(p.ny),
                               static_cast<std::size_t>(p.nx),
                               3});
    t_bot.copy_from(H_bot.data.data(), H_bot.data.size());
    w.write("H_top", t_top);
    w.write("H_bot", t_bot);
    w.close();
    std::puts("wrote diff_slab_field.npz");
    return 0;
}
