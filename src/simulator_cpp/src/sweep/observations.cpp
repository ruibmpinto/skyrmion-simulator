#include "skyrmion/sweep/observations.hpp"

#include "skyrmion/observables.hpp"

namespace skyrmion {
namespace sweep {

Observations observe_state(const Field3& m_top, const Field3& m_bot,
                           const Params& p) {
    const int pol_top = +1;
    const int pol_bot = -1;
    const Real a = p.a;

    Observations o;
    Center2D c_top = skyrmion_center_pbc(m_top, a, pol_top);
    Center2D c_bot = skyrmion_center_pbc(m_bot, a, pol_bot);
    o.cx_top = c_top.cx; o.cy_top = c_top.cy;
    o.cx_bot = c_bot.cx; o.cy_bot = c_bot.cy;

    o.d_top = skyrmion_diameter(m_top, a, pol_top);
    o.d_bot = skyrmion_diameter(m_bot, a, pol_bot);

    Ellipse e_top = skyrmion_ellipse(m_top, a, pol_top);
    Ellipse e_bot = skyrmion_ellipse(m_bot, a, pol_bot);
    o.D1_top = e_top.D1; o.D2_top = e_top.D2; o.theta_top = e_top.theta;
    o.D1_bot = e_bot.D1; o.D2_bot = e_bot.D2; o.theta_bot = e_bot.theta;

    o.psi_top = dw_angle(m_top, a, pol_top, 0.5);
    o.psi_bot = dw_angle(m_bot, a, pol_bot, 0.5);

    o.Q_top = topological_charge(m_top, a);
    o.Q_bot = topological_charge(m_bot, a);
    return o;
}

} // namespace sweep
} // namespace skyrmion
