#include "skyrmion/stochastic/diagnostics.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {
namespace stochastic {

namespace { constexpr double kPi = 3.14159265358979323846; }

Trajectory2D unwrap_trajectory(const std::vector<double>& cx,
                               const std::vector<double>& cy,
                               double L_x, double L_y, bool periodic_y) {
    if (!(std::isfinite(L_x) && L_x > 0.0 && std::isfinite(L_y) && L_y > 0.0)) {
        throw std::runtime_error(
            "unwrap_trajectory: L_x, L_y must be finite and > 0.");
    }
    if (cx.size() != cy.size()) {
        throw std::runtime_error(
            "unwrap_trajectory: cx and cy must have equal length.");
    }
    Trajectory2D out;
    const std::size_t n = cx.size();
    out.cx.resize(n);
    out.cy.resize(n);
    if (n == 0) return out;
    out.cx[0] = cx[0];
    out.cy[0] = cy[0];
    for (std::size_t i = 1; i < n; ++i) {
        double dx = cx[i] - cx[i - 1];
        if (dx > 0.5 * L_x)  dx -= L_x;
        if (dx < -0.5 * L_x) dx += L_x;
        double dy = cy[i] - cy[i - 1];
        if (periodic_y) {
            if (dy > 0.5 * L_y)  dy -= L_y;
            if (dy < -0.5 * L_y) dy += L_y;
        }
        out.cx[i] = out.cx[i - 1] + dx;
        out.cy[i] = out.cy[i - 1] + dy;
    }
    return out;
}

int detect_annihilation(const std::vector<double>& Q,
                        double q_threshold, int k_consecutive) {
    if (!(std::isfinite(q_threshold) && q_threshold > 0.0)) {
        throw std::runtime_error(
            "detect_annihilation: q_threshold must be finite and > 0.");
    }
    if (k_consecutive <= 0) {
        throw std::runtime_error(
            "detect_annihilation: k_consecutive must be a positive int.");
    }
    const int n = static_cast<int>(Q.size());
    if (n < k_consecutive) return -1;
    int run = 0;
    for (int i = 0; i < n; ++i) {
        run = (std::abs(Q[i]) < q_threshold) ? run + 1 : 0;
        if (run >= k_consecutive) return i;
    }
    return -1;
}

HallFit hall_angle(const std::vector<double>& t,
                   const std::vector<double>& cx,
                   const std::vector<double>& cy, double half) {
    if (t.size() != cx.size() || t.size() != cy.size()) {
        throw std::runtime_error("hall_angle: shape mismatch.");
    }
    if (!(half > 0.0 && half < 1.0)) {
        throw std::runtime_error("hall_angle: half must be in (0, 1).");
    }
    const int n = static_cast<int>(t.size());
    if (n < 4) {
        throw std::runtime_error("hall_angle: need at least 4 samples.");
    }
    const int i0 = static_cast<int>((1.0 - half) * n);
    // Linear least-squares slope (matches numpy.polyfit(t, y, 1)[0]).
    auto slope = [&](const std::vector<double>& y) {
        double St = 0, Sy = 0, Stt = 0, Sty = 0;
        const double mcount = static_cast<double>(n - i0);
        for (int i = i0; i < n; ++i) {
            St += t[i]; Sy += y[i]; Stt += t[i] * t[i]; Sty += t[i] * y[i];
        }
        return (mcount * Sty - St * Sy) / (mcount * Stt - St * St);
    };
    HallFit f;
    f.v_x = slope(cx);
    f.v_y = slope(cy);
    // Deflection off the drive (x) axis: -x drift with small v_y must
    // read as a small angle, not +-180 deg.
    f.theta_deg = std::atan2(f.v_y, std::abs(f.v_x)) * 180.0 / kPi;
    return f;
}

} // namespace stochastic
} // namespace skyrmion
