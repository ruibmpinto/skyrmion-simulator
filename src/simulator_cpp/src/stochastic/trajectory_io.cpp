#include "skyrmion/stochastic/trajectory_io.hpp"

#include <npy/npy.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace skyrmion {
namespace stochastic {

SAFPair load_saf_npz(const std::string& path) {
    npy::npzfilereader r(path);
    auto read_one = [&](const std::string& key) {
        npy::tensor<Real> t = r.read<npy::tensor<Real>>(key);
        const std::vector<std::size_t>& sh = t.shape();
        int ny = 0, nx = 0;
        if (sh.size() == 4) {
            ny = static_cast<int>(sh[1]); nx = static_cast<int>(sh[2]);
        } else if (sh.size() == 3) {
            ny = static_cast<int>(sh[0]); nx = static_cast<int>(sh[1]);
        } else {
            throw std::runtime_error(
                "load_saf_npz: unexpected shape for " + key);
        }
        Field3 f(ny, nx);
        if (t.size() < f.data.size()) {
            throw std::runtime_error(
                "load_saf_npz: too few values for " + key);
        }
        std::copy(t.data(), t.data() + f.data.size(), f.data.begin());
        return f;
    };
    SAFPair p;
    p.m_top = read_one("m_top");
    p.m_bot = read_one("m_bot");
    return p;
}

namespace {

constexpr const char* kSchema = "stochastic_llgs/0.1";

npy::tensor<double> arr(const std::vector<double>& v) {
    npy::tensor<double> t({v.size()});
    if (!v.empty()) t.copy_from(v.data(), v.size());
    return t;
}

// 2-d float32 array (ny, nx) for field snapshots.
npy::tensor<float> arr2f(const std::vector<float>& v,
                         std::size_t ny, std::size_t nx) {
    if (v.size() != ny * nx) {
        throw std::runtime_error(
            "save_trajectory: mz_final_top size does not match ny*nx.");
    }
    npy::tensor<float> t({ny, nx});
    if (!v.empty()) t.copy_from(v.data(), v.size());
    return t;
}

// 0-d scalar arrays (empty shape) so numpy loads them as Python
// scalars, matching np.savez of a plain int/float (a 1-d (1,) array
// would make int()/float() raise under numpy 2.x).
npy::tensor<double> sc(double v) {
    npy::tensor<double> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}

npy::tensor<std::int64_t> sc_i(std::int64_t v) {
    npy::tensor<std::int64_t> t(std::vector<std::size_t>{});
    t.copy_from(&v, 1);
    return t;
}

npy::tensor<std::uint8_t> bytes(const std::string& s) {
    std::vector<std::uint8_t> b(s.begin(), s.end());
    npy::tensor<std::uint8_t> t({b.size()});
    if (!b.empty()) t.copy_from(b.data(), b.size());
    return t;
}

} // namespace

void save_trajectory(const std::string& path, const StochasticPayload& pl,
                     const std::string& config_json,
                     const std::vector<std::pair<std::string, double>>& extras) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) std::filesystem::create_directories(p.parent_path());
    npy::npzfilewriter w(path);

    w.write("t_sample",        arr(pl.t_sample));
    w.write("cx_wrapped",      arr(pl.cx_wrapped));
    w.write("cy_wrapped",      arr(pl.cy_wrapped));
    w.write("cx_unwrapped",    arr(pl.cx_unwrapped));
    w.write("cy_unwrapped",    arr(pl.cy_unwrapped));
    w.write("Q",               arr(pl.Q));
    w.write("diameter",        arr(pl.diameter));
    w.write("cx_wrapped_bot",  arr(pl.cx_wrapped_bot));
    w.write("cy_wrapped_bot",  arr(pl.cy_wrapped_bot));
    w.write("cx_unwrapped_bot", arr(pl.cx_unwrapped_bot));
    w.write("cy_unwrapped_bot", arr(pl.cy_unwrapped_bot));
    w.write("Q_bot",           arr(pl.Q_bot));
    w.write("diameter_bot",    arr(pl.diameter_bot));
    w.write("cx_lcc",          arr(pl.cx_lcc));
    w.write("cy_lcc",          arr(pl.cy_lcc));
    w.write("cx_lcc_bot",      arr(pl.cx_lcc_bot));
    w.write("cy_lcc_bot",      arr(pl.cy_lcc_bot));
    w.write("diameter_lcc",    arr(pl.diameter_lcc));
    w.write("diameter_lcc_bot", arr(pl.diameter_lcc_bot));
    w.write("D1_top",          arr(pl.D1_top));
    w.write("D2_top",          arr(pl.D2_top));
    w.write("theta_top",       arr(pl.theta_top));
    w.write("D1_bot",          arr(pl.D1_bot));
    w.write("D2_bot",          arr(pl.D2_bot));
    w.write("theta_bot",       arr(pl.theta_bot));
    w.write("norm_drift_max",  arr(pl.norm_drift_max));
    // Dissipation split, written only when recorded (empty otherwise),
    // so the key set of the default trajectories is unchanged.
    if (!pl.diss_trans.empty()) {
        w.write("diss_trans",  arr(pl.diss_trans));
        w.write("diss_def",    arr(pl.diss_def));
        w.write("diss_total",  arr(pl.diss_total));
        w.write("v_fit_x",     arr(pl.v_fit_x));
        w.write("v_fit_y",     arr(pl.v_fit_y));
    }
    w.write("L_x",             sc(pl.L_x));
    w.write("L_y",             sc(pl.L_y));
    w.write("T_sub",           sc(pl.T_sub));
    w.write("j_current",       sc(pl.j_current));
    w.write("D1_relaxed_top",  sc(pl.D1_relaxed_top));
    w.write("D2_relaxed_top",  sc(pl.D2_relaxed_top));
    w.write("n_relax_used",    sc_i(pl.n_relax_used));
    w.write("equil_converged", sc_i(pl.equil_converged ? 1 : 0));
    w.write("T_effective",     sc(pl.T_effective));
    w.write("sigma_noise",     sc(pl.sigma_noise));
    w.write("alive_at_end",    sc_i(pl.alive_at_end ? 1 : 0));
    w.write("flip_index",      sc_i(pl.flip_index));
    w.write("mz_final_top",
            arr2f(pl.mz_final_top,
                  static_cast<std::size_t>(pl.ny),
                  static_cast<std::size_t>(pl.nx)));
    w.write("v_x",             sc(pl.v_x));
    w.write("v_y",             sc(pl.v_y));
    w.write("velocity",        sc(pl.velocity));
    w.write("hall_deg",        sc(pl.hall_deg));
    w.write("sigma_y",         sc(pl.sigma_y));
    w.write("v_x_bot",         sc(pl.v_x_bot));
    w.write("v_y_bot",         sc(pl.v_y_bot));
    w.write("velocity_bot",    sc(pl.velocity_bot));
    w.write("hall_deg_bot",    sc(pl.hall_deg_bot));
    w.write("meta_schema_version", bytes(kSchema));
    w.write("meta_timestamp", sc(static_cast<double>(
        std::chrono::duration_cast<std::chrono::seconds>(
            std::chrono::system_clock::now().time_since_epoch()).count())));
    w.write("meta_config_repr", bytes(config_json));
    for (const auto& kv : extras) w.write(kv.first, sc(kv.second));
    w.close();
}

void save_pair_potential(const std::string& path,
                         const std::vector<double>& t_sample,
                         const std::vector<double>& r_pair,
                         const std::vector<double>& Q,
                         bool alive_both, double r_init, int ens_idx,
                         double T_sub, double sigma_noise) {
    std::filesystem::path p(path);
    if (p.has_parent_path()) {
        std::filesystem::create_directories(p.parent_path());
    }
    npy::npzfilewriter w(path);
    w.write("t_sample",    arr(t_sample));
    w.write("r_pair",      arr(r_pair));
    w.write("Q",           arr(Q));
    w.write("alive_both",  sc(alive_both ? 1.0 : 0.0));
    w.write("r_init",      sc(r_init));
    w.write("ens_idx",     sc(static_cast<double>(ens_idx)));
    w.write("T_sub",       sc(T_sub));
    w.write("sigma_noise", sc(sigma_noise));
    w.close();
}

} // namespace stochastic
} // namespace skyrmion
