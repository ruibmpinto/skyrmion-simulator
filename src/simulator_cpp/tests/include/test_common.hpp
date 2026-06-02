// Shared helpers for the C++ vs Python parity tests.
//
// Each test executable loads `references.npz` (built by the Python
// `make_references.py`) and calls the analogous C++ function with the
// stored input arrays. A test passes if
//   max|a - b|  <=  atol + rtol * max|b|
// (numpy.allclose convention; atol prevents near-zero expected values
// from blowing up a pure relative-error metric).
#pragma once

#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/types.hpp"

#include <npy/npy.h>

#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace test_common {

constexpr double kDefaultRtol = 1.0e-10;
constexpr double kDefaultAtol = 1.0e-12;
using Complex = std::complex<double>;

class TestRef {
public:
    explicit TestRef(const std::string& path) : reader_(path) {}

    template <typename T>
    T scalar(const std::string& key) {
        auto t = reader_.read<npy::tensor<T>>(key);
        if (t.size() != 1) {
            throw std::runtime_error("scalar size != 1: " + key);
        }
        return *t.data();
    }

    template <typename T>
    std::vector<T> vec(const std::string& key) {
        auto t = reader_.read<npy::tensor<T>>(key);
        return std::vector<T>(t.data(), t.data() + t.size());
    }

    template <typename T>
    npy::tensor<T> tensor(const std::string& key) {
        return reader_.read<npy::tensor<T>>(key);
    }

    skyrmion::Field3 field3(const std::string& key) {
        auto t = reader_.read<npy::tensor<double>>(key);
        if (t.shape().size() != 3 || t.shape()[2] != 3) {
            throw std::runtime_error("field3: shape mismatch: " + key);
        }
        skyrmion::Field3 f(static_cast<int>(t.shape()[0]),
                           static_cast<int>(t.shape()[1]));
        std::copy(t.data(), t.data() + t.size(), f.data.begin());
        return f;
    }

private:
    npy::npzfilereader reader_;
};

struct Diff { double max_diff; double max_ref; };

inline Diff array(const double* a, const double* b, std::size_t n) {
    double mb = 0.0, md = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double be = std::abs(b[k]);
        const double de = std::abs(a[k] - b[k]);
        if (be > mb) mb = be;
        if (de > md) md = de;
    }
    return {md, mb};
}

inline Diff array_complex(const Complex* a, const Complex* b, std::size_t n) {
    double mb = 0.0, md = 0.0;
    for (std::size_t k = 0; k < n; ++k) {
        const double be = std::abs(b[k]);
        const double de = std::abs(a[k] - b[k]);
        if (be > mb) mb = be;
        if (de > md) md = de;
    }
    return {md, mb};
}

inline Diff scalar(double a, double b) {
    return {std::abs(a - b), std::abs(b)};
}

inline bool pass(Diff d, double atol = kDefaultAtol, double rtol = kDefaultRtol) {
    const double tol = atol + rtol * d.max_ref;
    return std::isfinite(d.max_diff) && d.max_diff <= tol;
}

inline skyrmion::Params build_params_from_ref(TestRef& r) {
    skyrmion::Params p;
    p.nx = static_cast<int>(r.scalar<int64_t>("state_nx"));
    p.ny = static_cast<int>(r.scalar<int64_t>("state_ny"));
    p.a            = r.scalar<double>("state_a");
    p.Ms           = r.scalar<double>("params_Ms");
    p.A_ex         = r.scalar<double>("params_A_ex");
    p.D            = r.scalar<double>("params_D");
    p.K_top        = r.scalar<double>("params_K_top");
    p.K_bot        = r.scalar<double>("params_K_bot");
    p.alpha        = r.scalar<double>("params_alpha");
    p.t_Co         = r.scalar<double>("params_t_Co");
    p.d_Ru         = r.scalar<double>("params_d_Ru");
    p.mu0          = r.scalar<double>("params_mu0");
    p.gamma_       = r.scalar<double>("params_gamma");
    p.H_RKKY       = r.scalar<double>("params_H_RKKY");
    p.DL_SOT       = r.scalar<double>("params_DL_SOT");
    p.FL_SOT       = r.scalar<double>("params_FL_SOT");
    p.J_current    = r.scalar<double>("params_J_current");
    p.lambda_sq    = r.scalar<double>("params_lambda_sq");
    p.P            = r.scalar<double>("params_P");
    p.mu_B_over_q_e = r.scalar<double>("params_mu_B_over_q_e");
    p.dt           = r.scalar<double>("params_dt");
    auto h_ext = r.vec<double>("params_H_ext");
    p.H_ext = {h_ext[0], h_ext[1], h_ext[2]};
    auto p_hat = r.vec<double>("params_p_hat");
    p.p_hat = {p_hat[0], p_hat[1], p_hat[2]};
    p.pulse = std::make_shared<skyrmion::ConstantPulse>(p.J_current);
    skyrmion::precompute(p);
    return p;
}

class TestRunner {
public:
    void check(const std::string& name, Diff d,
               double atol = kDefaultAtol, double rtol = kDefaultRtol) {
        const bool ok = pass(d, atol, rtol);
        std::printf("  %-44s diff=%.3e  ref=%.3e  %s\n",
                    name.c_str(), d.max_diff, d.max_ref,
                    ok ? "PASS" : "FAIL");
        if (ok) ++n_pass_;
        else ++n_fail_;
    }
    // Boolean assertion.
    void expect(const std::string& name, bool ok) {
        std::printf("  %-44s %s\n", name.c_str(), ok ? "PASS" : "FAIL");
        if (ok) ++n_pass_;
        else ++n_fail_;
    }
    // Passes iff `fn()` throws a std::exception (explicit-raise contract).
    template <typename F>
    void expect_throws(const std::string& name, F&& fn) {
        bool threw = false;
        try { fn(); } catch (const std::exception&) { threw = true; }
        expect(name, threw);
    }
    // Passes iff `fn()` does NOT throw.
    template <typename F>
    void expect_no_throw(const std::string& name, F&& fn) {
        bool ok = true;
        try { fn(); } catch (const std::exception&) { ok = false; }
        expect(name, ok);
    }
    int report(const char* suite) const {
        std::printf("%s: %d/%d passed.\n", suite,
                    n_pass_, n_pass_ + n_fail_);
        return n_fail_ == 0 ? 0 : 1;
    }
private:
    int n_pass_ = 0;
    int n_fail_ = 0;
};

} // namespace test_common
