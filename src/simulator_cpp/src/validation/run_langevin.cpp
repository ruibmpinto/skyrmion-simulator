// Langevin-function macrospin validation gate. Port of
// src/stochastic_llgs/validation/test_langevin.py.
//
// A Zeeman macrospin (no anisotropy, alpha = 1) ensemble; the time- and
// ensemble-averaged <m_z> is compared to the exact Langevin function
//   L(x) = coth(x) - 1/x,  x = Ms V_cell B / (k_B T).
// Pass: RMS relative error over all (T, B) cells < tol_rms. Failure
// usually means sigma_noise or the Stratonovich drift is wrong.
#include "skyrmion/validation/analytic.hpp"
#include "skyrmion/validation/macrospin.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

using namespace skyrmion;

namespace {

npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}
npy::tensor<double> arr1d(const std::vector<double>& v) {
    npy::tensor<double> t({v.size()});
    t.copy_from(v.data(), v.size());
    return t;
}
npy::tensor<double> arr2d(const std::vector<double>& v, int r, int c) {
    npy::tensor<double> t({static_cast<std::size_t>(r),
                           static_cast<std::size_t>(c)});
    t.copy_from(v.data(), v.size());
    return t;
}

} // namespace

int main() {
    // =========================== Run configuration ==========================
    const std::vector<double> t_kelvin_list = {
        10.0, 25.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0};
    const std::vector<double> b_z_list = {0.25, 0.5};
    const double alpha = 1.0;
    const double a = 2.0e-9;
    const double t_co = 1.3e-9;
    const double ms = 1.43e6;
    const double gamma = 194.8e9;
    const double dt = 1.0e-14;
    const int n_steps = 200000;          // 2 ns
    const int n_relax_steps = 20000;     // discard 200 ps
    const int sample_every = 100;
    const int n_traj = 256;
    const long long seed_base = 17;
    const double tol_norm = 5.0e-3;
    const double tol_rms = 0.05;
    const std::string out_dir = "output/stochastic_llgs/validation";
    const std::string out_npz = "langevin.npz";
    // ======================= End run configuration ==========================

    const int n_T = static_cast<int>(t_kelvin_list.size());
    const int n_B = static_cast<int>(b_z_list.size());
    std::vector<double> mz_sim(n_T * n_B), mz_se(n_T * n_B);
    std::vector<double> mz_lan(n_T * n_B), x_vals(n_T * n_B);
    const int relax_samples = (n_relax_steps + sample_every - 1)
                              / sample_every;

    std::printf("Langevin macrospin gate\n");
    std::printf("n_traj=%d n_steps=%d dt=%.2e sample_every=%d\n",
                n_traj, n_steps, dt, sample_every);

    for (int i = 0; i < n_T; ++i) {
        for (int j = 0; j < n_B; ++j) {
            const double T = t_kelvin_list[i];
            const double B = b_z_list[j];
            const long long seed = seed_base + 1000LL * i + j;
            Params p = validation::make_macrospin_params(
                T, alpha, Vec3{0.0, 0.0, B}, 0.0, a, t_co, ms, gamma,
                seed, n_traj);
            const double mu = p.Ms * p.V_cell;
            const double x = mu * B / (p.k_B * T);
            const double L = validation::langevin_function(x);

            validation::MacrospinHistory h =
                validation::run_macrospin_ensemble(
                    p, Vec3{0.0, 0.0, 1.0}, dt, n_steps, sample_every,
                    tol_norm);
            const int ns = h.n_samples;
            const int n_post = ns - relax_samples;
            if (n_post <= 1) {
                throw std::runtime_error("langevin: too few post-relax "
                                         "samples.");
            }
            // Per-trajectory time-mean of m_z over post-relax samples.
            std::vector<double> mz_traj(n_traj, 0.0);
            for (int s = relax_samples; s < ns; ++s)
                for (int tr = 0; tr < n_traj; ++tr)
                    mz_traj[tr] +=
                        h.m_top[(static_cast<std::size_t>(s) * n_traj + tr)
                                * 3 + 2];
            double mz_mean = 0.0;
            for (int tr = 0; tr < n_traj; ++tr) {
                mz_traj[tr] /= n_post;
                mz_mean += mz_traj[tr];
            }
            mz_mean /= n_traj;
            double var = 0.0;
            for (int tr = 0; tr < n_traj; ++tr) {
                const double d = mz_traj[tr] - mz_mean;
                var += d * d;
            }
            var /= (n_traj - 1);
            const double se = std::sqrt(var) / std::sqrt(double(n_traj));

            const int idx = i * n_B + j;
            mz_sim[idx] = mz_mean; mz_se[idx] = se;
            mz_lan[idx] = L; x_vals[idx] = x;
            std::printf("  T=%6.1f K B=%.2f T x=%6.3f L=%+.4f "
                        "sim=%+.4f +/- %.4f\n",
                        T, B, x, L, mz_mean, se);
        }
    }

    // ---- Pass criterion -------------------------------------------------
    std::vector<double> rel_err(n_T * n_B);
    double rms = 0.0, max_abs = 0.0;
    for (int k = 0; k < n_T * n_B; ++k) {
        rel_err[k] = (mz_sim[k] - mz_lan[k]) / mz_lan[k];
        rms += rel_err[k] * rel_err[k];
        max_abs = std::max(max_abs, std::abs(rel_err[k]));
    }
    rms = std::sqrt(rms / (n_T * n_B));

    std::filesystem::create_directories(out_dir);
    const std::string path = out_dir + "/" + out_npz;
    npy::npzfilewriter w(path);
    w.write("t_kelvin", arr1d(t_kelvin_list));
    w.write("b_z", arr1d(b_z_list));
    w.write("mz_sim", arr2d(mz_sim, n_T, n_B));
    w.write("mz_se", arr2d(mz_se, n_T, n_B));
    w.write("mz_langevin", arr2d(mz_lan, n_T, n_B));
    w.write("x_vals", arr2d(x_vals, n_T, n_B));
    w.write("rel_err", arr2d(rel_err, n_T, n_B));
    w.write("rms", sc(rms)); w.write("max_abs", sc(max_abs));
    w.write("tol_rms", sc(tol_rms));
    w.write("n_traj", sc(n_traj)); w.write("n_steps", sc(n_steps));
    w.write("dt", sc(dt)); w.write("sample_every", sc(sample_every));
    w.write("n_relax_steps", sc(n_relax_steps));
    w.close();

    std::printf("RMS rel error = %.4f, max|rel_err| = %.4f; tol = %.4f\n",
                rms, max_abs, tol_rms);
    std::printf("Saved %s\n", path.c_str());
    if (rms >= tol_rms) {
        std::printf("Langevin gate FAILED.\n");
        throw std::runtime_error("Langevin gate FAILED: RMS >= tol_rms.");
    }
    std::printf("Langevin gate PASSED.\n");
    return 0;
}
