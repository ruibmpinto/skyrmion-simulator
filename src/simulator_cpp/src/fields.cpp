#include "skyrmion/fields.hpp"

#include "skyrmion/lattice.hpp"

namespace skyrmion {

// Single fused loop computes:
//   H = exchange + DMI + anisotropy + Zeeman + RKKY
// with PBC-aware neighbour access. Avoids building four shifted copies
// of m, which the Python version uses for vectorization but is wasteful
// in C++.
void effective_field(const Field3& m, const Field3& m_other,
                     Real C_ex, Real C_dmi, Real C_anis,
                     const Vec3& H_ext, Real H_RKKY,
                     Field3& H) {
    const int ny = m.ny;
    const int nx = m.nx;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < ny; ++i) {
        int im, ip;
        pbc_pm(i, ny, im, ip);
        for (int j = 0; j < nx; ++j) {
            int jm, jp;
            pbc_pm(j, nx, jm, jp);

            const Real mx  = m(i, j, 0);
            const Real my  = m(i, j, 1);
            const Real mz  = m(i, j, 2);

            const Real mpx_x = m(i, jp, 0);  const Real mpx_y = m(i, jp, 1);  const Real mpx_z = m(i, jp, 2);
            const Real mmx_x = m(i, jm, 0);  const Real mmx_y = m(i, jm, 1);  const Real mmx_z = m(i, jm, 2);
            const Real mpy_x = m(ip, j, 0);  const Real mpy_y = m(ip, j, 1);  const Real mpy_z = m(ip, j, 2);
            const Real mmy_x = m(im, j, 0);  const Real mmy_y = m(im, j, 1);  const Real mmy_z = m(im, j, 2);

            // Exchange: C_ex * (sum(neighbours) - 4 m)
            const Real ex_x = C_ex * (mpx_x + mmx_x + mpy_x + mmy_x - 4.0 * mx);
            const Real ex_y = C_ex * (mpx_y + mmx_y + mpy_y + mmy_y - 4.0 * my);
            const Real ex_z = C_ex * (mpx_z + mmx_z + mpy_z + mmy_z - 4.0 * mz);

            // Interfacial Neel DMI:
            //   Hx =  C * (m+x_z - m-x_z)
            //   Hy =  C * (m+y_z - m-y_z)
            //   Hz = -C * ((m+x_x - m-x_x) + (m+y_y - m-y_y))
            const Real dmi_x =  C_dmi * (mpx_z - mmx_z);
            const Real dmi_y =  C_dmi * (mpy_z - mmy_z);
            const Real dmi_z = -C_dmi * ((mpx_x - mmx_x) + (mpy_y - mmy_y));

            // Perpendicular anisotropy: H_z = C_anis * m_z (others zero).
            const Real anis_z = C_anis * mz;

            // Zeeman: uniform field.
            // RKKY: -H_RKKY * m_other (antiparallel coupling).
            const Real mo_x = m_other(i, j, 0);
            const Real mo_y = m_other(i, j, 1);
            const Real mo_z = m_other(i, j, 2);

            H(i, j, 0) = ex_x + dmi_x + H_ext[0] - H_RKKY * mo_x;
            H(i, j, 1) = ex_y + dmi_y + H_ext[1] - H_RKKY * mo_y;
            H(i, j, 2) = ex_z + dmi_z + anis_z + H_ext[2] - H_RKKY * mo_z;
        }
    }
}

void effective_field_demag(const Field3& m_top, const Field3& m_bot,
                           const Params& p, DemagState& demag,
                           Field3& H_top, Field3& H_bot) {
    const BareAnis ba = bare_anis_prefactors(p);
    effective_field(m_top, m_bot, p.C_ex, p.C_dmi, ba.C_top,
                    p.H_ext, p.H_RKKY, H_top);
    effective_field(m_bot, m_top, p.C_ex, p.C_dmi, ba.C_bot,
                    p.H_ext, p.H_RKKY, H_bot);

    Field3 H_dem_top(p.ny, p.nx);
    Field3 H_dem_bot(p.ny, p.nx);
    demag.compute(m_top, m_bot, H_dem_top, H_dem_bot);

#ifdef SKYRMION_OPENMP
    #pragma omp parallel for schedule(static)
#endif
    for (int i = 0; i < p.ny; ++i) {
        for (int j = 0; j < p.nx; ++j) {
            for (int k = 0; k < 3; ++k) {
                H_top(i, j, k) += H_dem_top(i, j, k);
                H_bot(i, j, k) += H_dem_bot(i, j, k);
            }
        }
    }
}

} // namespace skyrmion
