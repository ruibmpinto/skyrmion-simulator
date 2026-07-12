// Benchmark: closed-form vs Gauss-Legendre quadrature Newell demag kernel
// precompute, periodic and free-BC, over a few box sizes. Reports wall time
// per build and the closed-vs-quadrature speedup.
#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"

#include <chrono>
#include <cstdio>
#include <vector>

using namespace skyrmion;

namespace {

double time_build(const Params& p, bool freebc) {
    const auto t0 = std::chrono::steady_clock::now();
    if (freebc) {
        precompute_demag_newell_freebc(p, p.demag_accuracy, p.demag_tol_conv);
    } else {
        precompute_demag_newell(p, p.demag_accuracy, p.demag_tol_conv);
    }
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

int main() {
    struct Box { int nx, ny; };
    const std::vector<Box> boxes = {{256, 256}, {350, 350}, {350, 500}};

    std::printf("%-12s %-8s %12s %12s %10s\n",
                "box", "kind", "closed[s]", "quad[s]", "speedup");
    for (const Box& b : boxes) {
        for (int fb = 0; fb < 2; ++fb) {
            const bool freebc = (fb == 1);
            Params pc = make_default_params();
            pc.nx = b.nx; pc.ny = b.ny;
            pc.demag_accuracy = 8.0; pc.demag_tol_conv = 2.0e-2;
            precompute(pc);
            Params pq = pc;
            pc.demag_method = DemagMethod::Closed;
            pq.demag_method = DemagMethod::Quadrature;
            // Warm FFTW plans / caches once, then time.
            time_build(pc, freebc);
            const double tc = time_build(pc, freebc);
            const double tq = time_build(pq, freebc);
            char tag[16];
            std::snprintf(tag, sizeof(tag), "%dx%d", b.nx, b.ny);
            std::printf("%-12s %-8s %12.4f %12.4f %9.1fx\n",
                        tag, freebc ? "freebc" : "periodic",
                        tc, tq, tq / tc);
        }
    }
    return 0;
}
