#include "skyrmion/integrator.hpp"

#include "skyrmion/fields.hpp"
#include "skyrmion/lattice.hpp"

#include <cmath>
#include <stdexcept>

namespace skyrmion {

void normalize_inplace(Field3& m) {
    const int ny = m.ny, nx = m.nx;
    bool bad = false;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static) reduction(||:bad)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real mx = m(i, j, 0);
            const Real my = m(i, j, 1);
            const Real mz = m(i, j, 2);
            const Real n = std::sqrt(mx * mx + my * my + mz * mz);
            if (n == 0.0) bad = true;
            const Real inv = (n > 0.0) ? 1.0 / n : 0.0;
            m(i, j, 0) = mx * inv;
            m(i, j, 1) = my * inv;
            m(i, j, 2) = mz * inv;
        }
    }
    if (bad) {
        throw std::runtime_error(
            "normalize_inplace: zero-magnitude spin at a lattice site.");
    }
}

void llgs_rhs(const Field3& m, const Field3& H_eff, const Params& p, Real t,
              Field3& dmdt) {
    const Real gp = p.gamma_p;
    const Real alpha = p.alpha;
    const Real J_t = (*p.pulse)(t);
    const bool sot_active = (J_t != 0.0);
    const Real H_DL_t = sot_active ? p.DL_SOT * J_t : 0.0;
    const Real H_FL_t = sot_active ? p.FL_SOT * J_t : 0.0;
    const Real c_dl = H_DL_t + alpha * H_FL_t;
    const Real c_fl = H_FL_t - alpha * H_DL_t;
    const Real px = p.p_hat[0], py = p.p_hat[1], pz = p.p_hat[2];

    const bool tsh_active = (p.lambda_sq != 0.0) && (J_t != 0.0);
    const Real b_j = tsh_active ? p.mu_B_over_q_e * J_t * p.P / p.Ms : 0.0;
    const Real inv_2a = 1.0 / (2.0 * p.a);

    const int ny = m.ny, nx = m.nx;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        int im, ip;
        pbc_pm(i, ny, im, ip);
        for (int j = 0; j < nx; ++j) {
            int jm, jp;
            pbc_pm(j, nx, jm, jp);

            const Real mx = m(i, j, 0), my = m(i, j, 1), mz = m(i, j, 2);
            const Real Hx = H_eff(i, j, 0), Hy = H_eff(i, j, 1), Hz = H_eff(i, j, 2);

            // Precession: m x H
            const Real mxH_x = my * Hz - mz * Hy;
            const Real mxH_y = mz * Hx - mx * Hz;
            const Real mxH_z = mx * Hy - my * Hx;

            // Damping: m x (m x H)
            const Real mxmxH_x = my * mxH_z - mz * mxH_y;
            const Real mxmxH_y = mz * mxH_x - mx * mxH_z;
            const Real mxmxH_z = mx * mxH_y - my * mxH_x;

            Real dx = -gp * (mxH_x + alpha * mxmxH_x);
            Real dy = -gp * (mxH_y + alpha * mxmxH_y);
            Real dz = -gp * (mxH_z + alpha * mxmxH_z);

            if (sot_active) {
                // p_hat x m
                const Real pxm_x = py * mz - pz * my;
                const Real pxm_y = pz * mx - px * mz;
                const Real pxm_z = px * my - py * mx;
                Real norm = std::sqrt(pxm_x * pxm_x + pxm_y * pxm_y + pxm_z * pxm_z);
                if (norm < 1e-30) norm = 1e-30;
                const Real sx = pxm_x / norm;
                const Real sy = pxm_y / norm;
                const Real sz = pxm_z / norm;

                // m x s
                const Real mxs_x = my * sz - mz * sy;
                const Real mxs_y = mz * sx - mx * sz;
                const Real mxs_z = mx * sy - my * sx;

                dx += gp * (c_dl * mxs_x + c_fl * sx);
                dy += gp * (c_dl * mxs_y + c_fl * sy);
                dz += gp * (c_dl * mxs_z + c_fl * sz);
            }

            if (tsh_active) {
                // Central differences for gradients
                const Real dmx_dx = (m(i, jp, 0) - m(i, jm, 0)) * inv_2a;
                const Real dmy_dx = (m(i, jp, 1) - m(i, jm, 1)) * inv_2a;
                const Real dmz_dx = (m(i, jp, 2) - m(i, jm, 2)) * inv_2a;
                const Real dmx_dy = (m(ip, j, 0) - m(im, j, 0)) * inv_2a;
                const Real dmy_dy = (m(ip, j, 1) - m(im, j, 1)) * inv_2a;
                const Real dmz_dy = (m(ip, j, 2) - m(im, j, 2)) * inv_2a;
                // dm/dx x dm/dy
                const Real cx = dmy_dx * dmz_dy - dmz_dx * dmy_dy;
                const Real cy = dmz_dx * dmx_dy - dmx_dx * dmz_dy;
                const Real cz = dmx_dx * dmy_dy - dmy_dx * dmx_dy;
                const Real N_xy = mx * cx + my * cy + mz * cz;
                const Real coef = -b_j * p.lambda_sq * N_xy;
                dx += coef * dmx_dy;
                dy += coef * dmy_dy;
                dz += coef * dmz_dy;
            }

            dmdt(i, j, 0) = dx;
            dmdt(i, j, 1) = dy;
            dmdt(i, j, 2) = dz;
        }
    }
}

RHSLocalKeff::RHSLocalKeff(const Params& p, const std::uint8_t* mask)
    : p_(p), mask_(mask), H_top_(p.ny, p.nx), H_bot_(p.ny, p.nx) {}

void RHSLocalKeff::operator()(const Field3& m_top, const Field3& m_bot, Real t,
                              Field3& dmdt_top, Field3& dmdt_bot) {
    effective_field(m_top, m_bot, p_.C_ex, p_.C_dmi, p_.C_anis_top,
                    p_.H_ext, p_.H_RKKY, H_top_, mask_, /*free_y=*/false);
    effective_field(m_bot, m_top, p_.C_ex, p_.C_dmi, p_.C_anis_bot,
                    p_.H_ext, p_.H_RKKY, H_bot_, mask_, /*free_y=*/false);
    llgs_rhs(m_top, H_top_, p_, t, dmdt_top);
    llgs_rhs(m_bot, H_bot_, p_, t, dmdt_bot);
}

RHSDemag::RHSDemag(const Params& p, DemagState& demag,
                   const std::uint8_t* mask)
    : p_(p), demag_(demag), mask_(mask),
      H_top_(p.ny, p.nx), H_bot_(p.ny, p.nx) {}

void RHSDemag::operator()(const Field3& m_top, const Field3& m_bot, Real t,
                          Field3& dmdt_top, Field3& dmdt_bot) {
    effective_field_demag(m_top, m_bot, p_, demag_, H_top_, H_bot_, mask_);
    llgs_rhs(m_top, H_top_, p_, t, dmdt_top);
    llgs_rhs(m_bot, H_bot_, p_, t, dmdt_bot);
}

RHSSingleKeff::RHSSingleKeff(const Params& p, const std::uint8_t* mask)
    : p_(p), mask_(mask), H_(p.ny, p.nx) {}

void RHSSingleKeff::operator()(const Field3& m, Real t, Field3& dmdt) {
    effective_field(m, m, p_.C_ex, p_.C_dmi, p_.C_anis_top,
                    p_.H_ext, p_.H_RKKY, H_, mask_, /*free_y=*/false);
    llgs_rhs(m, H_, p_, t, dmdt);
}

namespace {

inline void axpby_normalize(const Field3& m, const Field3& kdt, Real coef,
                            Field3& out) {
    const int ny = m.ny, nx = m.nx;
    bool bad = false;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static) reduction(||:bad)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real x = m(i, j, 0) + coef * kdt(i, j, 0);
            const Real y = m(i, j, 1) + coef * kdt(i, j, 1);
            const Real z = m(i, j, 2) + coef * kdt(i, j, 2);
            const Real n = std::sqrt(x * x + y * y + z * z);
            if (n == 0.0) bad = true;
            const Real inv = (n > 0.0) ? 1.0 / n : 0.0;
            out(i, j, 0) = x * inv;
            out(i, j, 1) = y * inv;
            out(i, j, 2) = z * inv;
        }
    }
    if (bad) {
        throw std::runtime_error(
            "rk4_step: zero-magnitude spin during substage "
            "normalisation.");
    }
}

inline void combine_rk4_normalize(const Field3& m,
                                  const Field3& k1, const Field3& k2,
                                  const Field3& k3, const Field3& k4,
                                  Field3& out) {
    const int ny = m.ny, nx = m.nx;
    const Real one_sixth = 1.0 / 6.0;
    bool bad = false;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static) reduction(||:bad)
#endif
    for (int i = 0; i < ny; ++i) {
        for (int j = 0; j < nx; ++j) {
            const Real dx = (k1(i, j, 0) + 2.0 * k2(i, j, 0) + 2.0 * k3(i, j, 0) + k4(i, j, 0)) * one_sixth;
            const Real dy = (k1(i, j, 1) + 2.0 * k2(i, j, 1) + 2.0 * k3(i, j, 1) + k4(i, j, 1)) * one_sixth;
            const Real dz = (k1(i, j, 2) + 2.0 * k2(i, j, 2) + 2.0 * k3(i, j, 2) + k4(i, j, 2)) * one_sixth;
            const Real x = m(i, j, 0) + dx;
            const Real y = m(i, j, 1) + dy;
            const Real z = m(i, j, 2) + dz;
            const Real n = std::sqrt(x * x + y * y + z * z);
            if (n == 0.0) bad = true;
            const Real inv = (n > 0.0) ? 1.0 / n : 0.0;
            out(i, j, 0) = x * inv;
            out(i, j, 1) = y * inv;
            out(i, j, 2) = z * inv;
        }
    }
    if (bad) {
        throw std::runtime_error(
            "rk4_step: zero-magnitude spin during final "
            "normalisation.");
    }
}

inline void scale_inplace(Field3& f, Real s) {
    const std::size_t n = f.data.size();
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (std::size_t k = 0; k < n; ++k) f.data[k] *= s;
}

} // namespace

template <typename RHS>
void rk4_step(RHS& rhs, Field3& m_top, Field3& m_bot,
              Real t, Real dt, const Params& p) {
    Field3 k1t(p.ny, p.nx), k1b(p.ny, p.nx);
    Field3 k2t(p.ny, p.nx), k2b(p.ny, p.nx);
    Field3 k3t(p.ny, p.nx), k3b(p.ny, p.nx);
    Field3 k4t(p.ny, p.nx), k4b(p.ny, p.nx);
    Field3 mt2(p.ny, p.nx), mb2(p.ny, p.nx);

    rhs(m_top, m_bot, t, k1t, k1b);
    scale_inplace(k1t, dt); scale_inplace(k1b, dt);

    axpby_normalize(m_top, k1t, 0.5, mt2);
    axpby_normalize(m_bot, k1b, 0.5, mb2);
    rhs(mt2, mb2, t + 0.5 * dt, k2t, k2b);
    scale_inplace(k2t, dt); scale_inplace(k2b, dt);

    axpby_normalize(m_top, k2t, 0.5, mt2);
    axpby_normalize(m_bot, k2b, 0.5, mb2);
    rhs(mt2, mb2, t + 0.5 * dt, k3t, k3b);
    scale_inplace(k3t, dt); scale_inplace(k3b, dt);

    axpby_normalize(m_top, k3t, 1.0, mt2);
    axpby_normalize(m_bot, k3b, 1.0, mb2);
    rhs(mt2, mb2, t + dt, k4t, k4b);
    scale_inplace(k4t, dt); scale_inplace(k4b, dt);

    combine_rk4_normalize(m_top, k1t, k2t, k3t, k4t, m_top);
    combine_rk4_normalize(m_bot, k1b, k2b, k3b, k4b, m_bot);
}

template <typename RHS>
void rk4_step_single(RHS& rhs, Field3& m, Real t, Real dt, const Params& p) {
    Field3 k1(p.ny, p.nx), k2(p.ny, p.nx);
    Field3 k3(p.ny, p.nx), k4(p.ny, p.nx);
    Field3 m2(p.ny, p.nx);

    rhs(m, t, k1);
    scale_inplace(k1, dt);

    axpby_normalize(m, k1, 0.5, m2);
    rhs(m2, t + 0.5 * dt, k2);
    scale_inplace(k2, dt);

    axpby_normalize(m, k2, 0.5, m2);
    rhs(m2, t + 0.5 * dt, k3);
    scale_inplace(k3, dt);

    axpby_normalize(m, k3, 1.0, m2);
    rhs(m2, t + dt, k4);
    scale_inplace(k4, dt);

    combine_rk4_normalize(m, k1, k2, k3, k4, m);
}

// Explicit instantiations for the RHS types declared in this TU.
template void rk4_step<RHSLocalKeff>(RHSLocalKeff&, Field3&, Field3&,
                                     Real, Real, const Params&);
template void rk4_step<RHSDemag>(RHSDemag&, Field3&, Field3&,
                                 Real, Real, const Params&);
template void rk4_step_single<RHSSingleKeff>(RHSSingleKeff&, Field3&,
                                             Real, Real, const Params&);

} // namespace skyrmion
