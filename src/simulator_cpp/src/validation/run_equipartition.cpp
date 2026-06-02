// Spin-wave equipartition validation gate. Port of
// src/stochastic_llgs/validation/test_equipartition.py.
//
// A 16x16 single-layer FM patch (SAF bottom decoupled by H_RKKY = 0),
// no DMI/anisotropy, weak Zeeman along z. Equilibrate at low T, sample
// the transverse magnetization, and verify each Fourier mode obeys
// Rayleigh-Jeans equipartition
//   <|M_x(k)|^2 + |M_y(k)|^2> = 2 N k_B T / (M_s V_cell H_k).
// Pass: median(sim/theory) over low-k modes within tol_rel of 1.
#include "skyrmion/fft2d.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/stochastic/heun.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/stochastic/thermal.hpp"
#include "skyrmion/validation/analytic.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

constexpr double kPi = 3.14159265358979323846;

npy::tensor<double> arr2d(const std::vector<double>& v, int ny, int nx) {
    npy::tensor<double> t({static_cast<std::size_t>(ny),
                           static_cast<std::size_t>(nx)});
    t.copy_from(v.data(), v.size());
    return t;
}
npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}
npy::tensor<std::uint8_t> mask2d(const std::vector<std::uint8_t>& v,
                                 int ny, int nx) {
    npy::tensor<std::uint8_t> t({static_cast<std::size_t>(ny),
                                 static_cast<std::size_t>(nx)});
    t.copy_from(v.data(), v.size());
    return t;
}

} // namespace

int main() {
    // =========================== Run configuration ==========================
    const int nx = 16, ny = 16;
    const double t_kelvin = 5.0;
    const double b_z = 0.5;          // Tesla
    const double alpha = 0.1;
    const double a = 2.0e-9;
    const double t_co = 1.3e-9;
    const double ms = 1.43e6;
    const double a_ex = 16.0e-12;
    const double gamma = 194.8e9;
    const double dt = 5.0e-14;
    const int n_relax_steps = 20000;     // 1 ns
    const int n_steps = 200000;          // 10 ns
    const int sample_every = 200;        // 10 ps
    const long long seed = 41;
    const double tol_norm = 5.0e-3;
    const double tol_rel = 0.20;
    const double k_low_cut = 0.5;
    const std::string out_dir = "output/stochastic_llgs/validation";
    const std::string out_npz = "equipartition.npz";
    // ======================= End run configuration ==========================

    std::printf("Spin-wave equipartition gate\n");
    std::printf("nx=%d ny=%d T=%.1f K B_z=%.2f T alpha=%.2f dt=%.2e\n",
                nx, ny, t_kelvin, b_z, alpha, dt);

    // ---- Parameters: single-layer FM patch, local K_eff, no demag -------
    Params p = make_default_params();
    p.nx = nx; p.ny = ny;
    p.alpha = alpha; p.gamma_ = gamma; p.Ms = ms; p.a = a;
    p.t_Co = t_co; p.A_ex = a_ex;
    p.H_ext = {0.0, 0.0, b_z};
    p.K_top = 0.0; p.K_bot = 0.0; p.H_RKKY = 0.0;
    p.J_current = 0.0; p.D = 0.0;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    precompute(p);
    // Match the Python gate exactly: bare exchange prefactor, no DMI,
    // no anisotropy, no SOT.
    p.C_ex = 2.0 * a_ex / (ms * a * a);
    p.C_dmi = 0.0;
    p.C_anis_top = 0.0; p.C_anis_bot = 0.0;
    p.H_DL = 0.0; p.H_FL = 0.0;
    stochastic::attach_thermal(p, t_kelvin, 0.0, seed);

    const std::vector<double> H_k =
        validation::magnon_stiffness_grid(ny, nx, b_z, p.C_ex);

    // ---- Initial condition: aligned with B_z ----------------------------
    Field3 m_top(ny, nx), m_bot(ny, nx);
    for (int i = 0; i < ny; ++i)
        for (int j = 0; j < nx; ++j) {
            m_top(i, j, 2) = 1.0;
            m_bot(i, j, 2) = 1.0;
        }

    stochastic::ThermalRng rng(static_cast<std::uint64_t>(seed));
    const double sigma = p.sigma_noise;
    stochastic::HeunStochasticStepper stepper(p, nullptr, rng, sigma,
                                              tol_norm, /*mask=*/nullptr);

    // ---- Relaxation -----------------------------------------------------
    std::printf("Relaxation phase...\n");
    double t = 0.0;
    for (int step = 0; step < n_relax_steps; ++step) {
        stepper.step(m_top, m_bot, t, dt, p);
        t += dt;
    }

    // ---- Sampling: accumulate <|M_x|^2 + |M_y|^2> -----------------------
    std::printf("Sampling phase...\n");
    FFT2D fft(ny, nx, 1);
    const std::size_t N = static_cast<std::size_t>(ny) * nx;
    std::vector<double> accum(N, 0.0);
    int n_samples = 0;
    auto fft_abs2_accumulate = [&](int comp) {
        fftw_complex* in = fft.scratch_in();
        for (int i = 0; i < ny; ++i)
            for (int j = 0; j < nx; ++j) {
                const std::size_t k = static_cast<std::size_t>(i) * nx + j;
                in[k][0] = m_top(i, j, comp);
                in[k][1] = 0.0;
            }
        fft.execute_fwd();
        fftw_complex* out = fft.scratch_out();
        for (std::size_t k = 0; k < N; ++k)
            accum[k] += out[k][0] * out[k][0] + out[k][1] * out[k][1];
    };
    for (int step = 0; step < n_steps; ++step) {
        stepper.step(m_top, m_bot, t, dt, p);
        t += dt;
        if (step % sample_every == 0) {
            fft_abs2_accumulate(0);
            fft_abs2_accumulate(1);
            ++n_samples;
        }
    }
    if (n_samples == 0) {
        throw std::runtime_error("equipartition: no samples collected.");
    }

    // ---- Compare to theory over low-k modes -----------------------------
    std::vector<double> sim_var(N), th_var(N), ratio(N);
    std::vector<std::uint8_t> low_k(N);
    std::vector<double> ratio_low;
    const double theory_pref = 2.0 * static_cast<double>(N) * p.k_B * p.T
                               / (p.Ms * p.V_cell);
    for (int i = 0; i < ny; ++i) {
        const int my = (i < (ny + 1) / 2) ? i : i - ny;
        const double fy = static_cast<double>(my) / ny;
        for (int j = 0; j < nx; ++j) {
            const int mx = (j < (nx + 1) / 2) ? j : j - nx;
            const double fx = static_cast<double>(mx) / nx;
            const std::size_t k = static_cast<std::size_t>(i) * nx + j;
            sim_var[k] = accum[k] / n_samples;
            th_var[k] = theory_pref / H_k[k];
            ratio[k] = sim_var[k] / th_var[k];
            const double kmag = std::sqrt(fx * fx + fy * fy) / 0.5;
            low_k[k] = (kmag < k_low_cut) ? 1 : 0;
            if (low_k[k]) ratio_low.push_back(ratio[k]);
        }
    }
    std::sort(ratio_low.begin(), ratio_low.end());
    const std::size_t m = ratio_low.size();
    const double median_ratio = (m % 2 == 1)
        ? ratio_low[m / 2]
        : 0.5 * (ratio_low[m / 2 - 1] + ratio_low[m / 2]);
    double max_dev = 0.0;
    for (double rr : ratio_low) max_dev = std::max(max_dev, std::abs(rr - 1.0));

    // ---- Persist + report -----------------------------------------------
    std::filesystem::create_directories(out_dir);
    const std::string path = out_dir + "/" + out_npz;
    npy::npzfilewriter w(path);
    w.write("nx", sc(nx)); w.write("ny", sc(ny));
    w.write("t_kelvin", sc(t_kelvin)); w.write("b_z", sc(b_z));
    w.write("alpha", sc(alpha)); w.write("dt", sc(dt));
    w.write("n_samples", sc(static_cast<double>(n_samples)));
    w.write("H_k", arr2d(H_k, ny, nx));
    w.write("sim_mode_var", arr2d(sim_var, ny, nx));
    w.write("th_mode_var", arr2d(th_var, ny, nx));
    w.write("ratio", arr2d(ratio, ny, nx));
    w.write("low_k_mask", mask2d(low_k, ny, nx));
    w.write("median_ratio", sc(median_ratio));
    w.write("max_dev", sc(max_dev));
    w.write("tol_rel", sc(tol_rel));
    w.close();

    std::printf("median(sim/theory) low-k (|k|/k_max < %.1f) = %.3f, "
                "max|ratio-1| = %.3f; tol = %.2f\n",
                k_low_cut, median_ratio, max_dev, tol_rel);
    std::printf("Saved %s\n", path.c_str());
    (void)kPi;
    if (std::abs(median_ratio - 1.0) >= tol_rel) {
        std::printf("Equipartition gate FAILED.\n");
        throw std::runtime_error(
            "Equipartition gate FAILED: median ratio deviates from 1 "
            "by more than tol_rel.");
    }
    std::printf("Equipartition gate PASSED.\n");
    return 0;
}
