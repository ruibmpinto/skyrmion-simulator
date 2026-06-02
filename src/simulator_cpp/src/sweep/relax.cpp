#include "skyrmion/sweep/relax.hpp"

#include "skyrmion/energy.hpp"
#include "skyrmion/fields.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/pulses.hpp"
#include "skyrmion/sweep/stepper.hpp"

#include <cmath>
#include <cstdio>
#include <limits>
#include <memory>

namespace skyrmion {
namespace sweep {

namespace {

// Max |m x (m x H)| across both layers (Tesla). H uses the bare-K +
// demag pair when `demag` is set, else the local-K_eff field.
double max_tangential_torque(const Field3& m_top, const Field3& m_bot,
                             const Params& p, DemagState* demag,
                             Field3& H_top, Field3& H_bot) {
    if (demag) {
        effective_field_demag(m_top, m_bot, p, *demag, H_top, H_bot);
    } else {
        effective_field(m_top, m_bot, p.C_ex, p.C_dmi, p.C_anis_top,
                        p.H_ext, p.H_RKKY, H_top);
        effective_field(m_bot, m_top, p.C_ex, p.C_dmi, p.C_anis_bot,
                        p.H_ext, p.H_RKKY, H_bot);
    }
    const int ny = m_top.ny, nx = m_top.nx;
    double tau_max = 0.0;
    auto layer_max = [&](const Field3& m, const Field3& H) {
        double lmax = 0.0;
        for (int i = 0; i < ny; ++i) {
            for (int j = 0; j < nx; ++j) {
                const Real mx = m(i, j, 0), my = m(i, j, 1), mz = m(i, j, 2);
                const Real Hx = H(i, j, 0), Hy = H(i, j, 1), Hz = H(i, j, 2);
                // mxH = m x H
                const Real ax = my * Hz - mz * Hy;
                const Real ay = mz * Hx - mx * Hz;
                const Real az = mx * Hy - my * Hx;
                // m x (m x H)
                const Real tx = my * az - mz * ay;
                const Real ty = mz * ax - mx * az;
                const Real tz = mx * ay - my * ax;
                const double n = std::sqrt(tx * tx + ty * ty + tz * tz);
                if (n > lmax) lmax = n;
            }
        }
        return lmax;
    };
    tau_max = std::max(layer_max(m_top, H_top), layer_max(m_bot, H_bot));
    return tau_max;
}

} // namespace

RelaxResult relax(Field3 m_top, Field3 m_bot, Params& p,
                  DemagState* demag,
                  int max_steps, double alpha_relax,
                  double tol_torque, double tol_dE,
                  int check_every, int print_every) {
    const double alpha_save = p.alpha;
    const double gamma_p_save = p.gamma_p;
    const Real H_DL_save = p.H_DL;
    const Real H_FL_save = p.H_FL;
    auto pulse_save = p.pulse;

    p.H_DL = 0.0;
    p.H_FL = 0.0;
    p.pulse = std::make_shared<ConstantPulse>(0.0);
    if (alpha_relax > 0.0) {
        p.alpha = alpha_relax;
        p.gamma_p = p.gamma_ / (1.0 + p.alpha * p.alpha);
    }

    RelaxResult res;
    res.converged = false;
    res.n_steps = 0;
    res.tau_max = std::numeric_limits<double>::infinity();
    res.E_final = std::numeric_limits<double>::quiet_NaN();

    Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
    try {
        std::unique_ptr<Stepper> stepper;
        if (demag) stepper.reset(new RK4DemagStepper(p, *demag));
        else       stepper.reset(new RK4LocalKeffStepper(p));

        double E_prev = demag ? total_energy(m_top, m_bot, p, *demag) : 0.0;
        Real t = 0.0;
        for (int step = 1; step <= max_steps; ++step) {
            stepper->step(m_top, m_bot, t, p.dt, p);
            t += p.dt;
            res.n_steps = step;
            if (print_every > 0 && step % print_every == 0) {
                std::printf("    relax step %d/%d (%.0f ps), tau_max=%.2e T\n",
                            step, max_steps, step * p.dt * 1e12, res.tau_max);
            }
            if (step % check_every == 0) {
                res.tau_max = max_tangential_torque(m_top, m_bot, p, demag,
                                                    H_top, H_bot);
                bool torque_ok = res.tau_max < tol_torque;
                bool energy_ok = true;
                if (demag) {
                    const double E_now = total_energy(m_top, m_bot, p, *demag);
                    const double dE = (E_now != 0.0)
                        ? std::abs((E_now - E_prev) / E_now)
                        : std::abs(E_now - E_prev);
                    E_prev = E_now;
                    energy_ok = dE < tol_dE;
                }
                if (torque_ok && energy_ok) {
                    res.converged = true;
                    break;
                }
            }
        }
        if (demag) res.E_final = total_energy(m_top, m_bot, p, *demag);
    } catch (...) {
        p.alpha = alpha_save;
        p.gamma_p = gamma_p_save;
        p.H_DL = H_DL_save;
        p.H_FL = H_FL_save;
        p.pulse = pulse_save;
        throw;
    }
    p.alpha = alpha_save;
    p.gamma_p = gamma_p_save;
    p.H_DL = H_DL_save;
    p.H_FL = H_FL_save;
    p.pulse = pulse_save;

    res.m_top = std::move(m_top);
    res.m_bot = std::move(m_bot);
    return res;
}

} // namespace sweep
} // namespace skyrmion
