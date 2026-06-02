#include "skyrmion/energy.hpp"

#include "skyrmion/fields.hpp"
#include "skyrmion/lattice.hpp"

namespace skyrmion {

Real total_energy(const Field3& m_top, const Field3& m_bot,
                  const Params& p, DemagState& demag) {
    const Real V_cell = p.t_Co * p.a * p.a;
    const Real Ms = p.Ms;
    const BareAnis ba = bare_anis_prefactors(p);

    // Internal (bilinear) field, bare-K anisotropy, both layers.
    Field3 H_top(p.ny, p.nx), H_bot(p.ny, p.nx);
    effective_field(m_top, m_bot, p.C_ex, p.C_dmi, ba.C_top,
                    Vec3{0, 0, 0}, p.H_RKKY, H_top, nullptr);
    effective_field(m_bot, m_top, p.C_ex, p.C_dmi, ba.C_bot,
                    Vec3{0, 0, 0}, p.H_RKKY, H_bot, nullptr);

    Field3 H_dem_top(p.ny, p.nx), H_dem_bot(p.ny, p.nx);
    demag.compute(m_top, m_bot, H_dem_top, H_dem_bot);

    Real dot_int = 0.0;
    Real dot_zee = 0.0;
#ifdef SKYRMION_OPENMP
    #pragma omp parallel for reduction(+:dot_int,dot_zee) schedule(static)
#endif
    for (int i = 0; i < p.ny; ++i) {
        for (int j = 0; j < p.nx; ++j) {
            for (int k = 0; k < 3; ++k) {
                const Real h_t = H_top(i, j, k) + H_dem_top(i, j, k);
                const Real h_b = H_bot(i, j, k) + H_dem_bot(i, j, k);
                dot_int += m_top(i, j, k) * h_t + m_bot(i, j, k) * h_b;
                dot_zee += (m_top(i, j, k) + m_bot(i, j, k)) * p.H_ext[k];
            }
        }
    }
    const Real E_internal = -0.5 * V_cell * Ms * dot_int;
    const Real E_zeeman   = -V_cell * Ms * dot_zee;
    return E_internal + E_zeeman;
}

} // namespace skyrmion
