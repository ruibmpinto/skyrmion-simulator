// Dumps the C++ Newell k-space demag kernel to newell_kernel.npz so
// the Python implementation can be cross-validated against it. Each
// component is stored as a complex (ny, nx/2+1) array matching the
// FFTW r2c half-spectrum used at run time.
#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"

#include <npy/npy.h>

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

npy::tensor<std::complex<double>>
to_tensor(const std::vector<std::complex<double>>& src, int ny, int nh) {
    npy::tensor<std::complex<double>> t({static_cast<std::size_t>(ny),
                                         static_cast<std::size_t>(nh)});
    t.copy_from(src.data(), src.size());
    return t;
}

template <typename T>
npy::tensor<T> scalar1d(T v) {
    npy::tensor<T> t({static_cast<std::size_t>(1)});
    t.copy_from(&v, 1);
    return t;
}

} // namespace

int main() {
    Params p = make_default_params();
    p.nx = 32;
    p.ny = 32;
    p.a  = 2.0e-9;
    p.demag_accuracy = 8.0;
    p.demag_tol_conv = 2.0e-2;

    DemagKernels K = precompute_demag_newell(p, p.demag_accuracy, p.demag_tol_conv);

    const int nh = p.nx / 2 + 1;
    npy::npzfilewriter w("newell_kernel.npz");
    w.write("Nxx_self",  to_tensor(K.Nxx_self,  p.ny, nh));
    w.write("Nyy_self",  to_tensor(K.Nyy_self,  p.ny, nh));
    w.write("Nzz_self",  to_tensor(K.Nzz_self,  p.ny, nh));
    w.write("Nxy_self",  to_tensor(K.Nxy_self,  p.ny, nh));
    w.write("Nxx_inter", to_tensor(K.Nxx_inter, p.ny, nh));
    w.write("Nyy_inter", to_tensor(K.Nyy_inter, p.ny, nh));
    w.write("Nzz_inter", to_tensor(K.Nzz_inter, p.ny, nh));
    w.write("Nxy_inter", to_tensor(K.Nxy_inter, p.ny, nh));
    w.write("Nxz_inter", to_tensor(K.Nxz_inter, p.ny, nh));
    w.write("Nyz_inter", to_tensor(K.Nyz_inter, p.ny, nh));
    w.write("ny", scalar1d<int64_t>(p.ny));
    w.write("nx", scalar1d<int64_t>(p.nx));
    w.write("a",  scalar1d<double>(p.a));
    w.write("t_Co", scalar1d<double>(p.t_Co));
    w.write("d_Ru", scalar1d<double>(p.d_Ru));
    w.write("mu0_Ms", scalar1d<double>(K.mu0_Ms));
    w.close();
    std::printf("Wrote newell_kernel.npz (ny=%d, nx=%d, nh=%d).\n",
                p.ny, p.nx, nh);
    return 0;
}
