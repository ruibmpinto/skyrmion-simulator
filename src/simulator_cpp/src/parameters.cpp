#include "skyrmion/parameters.hpp"

#include <cmath>
#include <cstdio>
#include <sstream>
#include <stdexcept>

namespace skyrmion {

void precompute(Params& p) {
    const Real a2 = p.a * p.a;
    p.C_ex  = 2.0 * p.A_ex / (p.Ms * a2);
    p.C_dmi = p.D / (p.Ms * p.a);

    const Real mu0_Ms = p.mu0 * p.Ms;
    p.C_anis_top = 2.0 * p.K_top / p.Ms - mu0_Ms;
    p.C_anis_bot = 2.0 * p.K_bot / p.Ms - mu0_Ms;
    p.gamma_p = p.gamma_ / (1.0 + p.alpha * p.alpha);
}

Params make_default_params() {
    Params p;
    precompute(p);
    return p;
}

BareAnis bare_anis_prefactors(const Params& p) {
    const Real inv_Ms = 1.0 / p.Ms;
    return BareAnis{2.0 * p.K_top * inv_Ms, 2.0 * p.K_bot * inv_Ms};
}

EffectiveAnis effective_anisotropy(const Params& p) {
    const Real half_mu0_Ms2 = 0.5 * p.mu0 * p.Ms * p.Ms;
    EffectiveAnis e;
    e.K_top = p.K_top;
    e.K_bot = p.K_bot;
    e.K_eff_top = p.K_top - half_mu0_Ms2;
    e.K_eff_bot = p.K_bot - half_mu0_Ms2;
    e.K_eff_avg = 0.5 * (e.K_eff_top + e.K_eff_bot);
    e.mu0_Ms2_over_2 = half_mu0_Ms2;
    return e;
}

Real critical_dmi(const Params& p) {
    const Real K_eff_avg = effective_anisotropy(p).K_eff_avg;
    if (K_eff_avg <= 0.0) {
        throw std::runtime_error("critical_dmi: K_eff_avg <= 0; undefined.");
    }
    return 4.0 * std::sqrt(p.A_ex * K_eff_avg) / 3.14159265358979323846;
}

Real pma_anisotropy_field(const Params& p) {
    const Real K_eff_avg = effective_anisotropy(p).K_eff_avg;
    if (K_eff_avg <= 0.0) {
        throw std::runtime_error("pma_anisotropy_field: K_eff_avg <= 0.");
    }
    return 2.0 * K_eff_avg / p.Ms;
}

static const char* demag_kind_str(DemagKind k) {
    switch (k) {
        case DemagKind::None:         return "none";
        case DemagKind::Slab:         return "slab";
        case DemagKind::Newell:       return "newell";
        case DemagKind::NewellFreeBC: return "newell_freebc";
        case DemagKind::Racetrack:    return "racetrack";
    }
    return "none";
}

std::string params_to_json(const Params& p) {
    std::ostringstream s;
    s.precision(17);
    s << "{";
    s << "\"nx\":" << p.nx << ",\"ny\":" << p.ny << ",\"a\":" << p.a;
    s << ",\"Ms\":" << p.Ms << ",\"A_ex\":" << p.A_ex << ",\"D\":" << p.D;
    s << ",\"K_top\":" << p.K_top << ",\"K_bot\":" << p.K_bot;
    s << ",\"alpha\":" << p.alpha << ",\"t_Co\":" << p.t_Co << ",\"d_Ru\":" << p.d_Ru;
    s << ",\"mu0\":" << p.mu0 << ",\"gamma\":" << p.gamma_;
    s << ",\"H_RKKY\":" << p.H_RKKY;
    s << ",\"H_ext\":[" << p.H_ext[0] << "," << p.H_ext[1] << "," << p.H_ext[2] << "]";
    s << ",\"DL_SOT\":" << p.DL_SOT << ",\"FL_SOT\":" << p.FL_SOT;
    s << ",\"J0\":" << (p.pulse ? (*p.pulse)(0.0) : 0.0);
    s << ",\"p_hat\":[" << p.p_hat[0] << "," << p.p_hat[1] << "," << p.p_hat[2] << "]";
    s << ",\"lambda_sq\":" << p.lambda_sq << ",\"P\":" << p.P;
    s << ",\"dt\":" << p.dt << ",\"n_relax\":" << p.n_relax
      << ",\"n_steps\":" << p.n_steps;
    s << ",\"skyrmion_R\":" << p.skyrmion_R
      << ",\"skyrmion_dw\":" << p.skyrmion_dw;
    s << ",\"dump_every_relax\":" << p.dump_every_relax
      << ",\"dump_every_drive\":" << p.dump_every_drive
      << ",\"max_dump_frames\":" << p.max_dump_frames;
    s << ",\"demag_kind\":\"" << demag_kind_str(p.demag_kind) << "\"";
    s << "}";
    return s.str();
}

std::string dmi_dir_tag(Real D) {
    // mJ/m^2 with trailing zeros stripped (%g), then '.' -> 'p'.
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", static_cast<double>(D) * 1e3);
    std::string tag = "D";
    for (const char* c = buf; *c; ++c) tag += (*c == '.') ? 'p' : *c;
    return tag;
}

} // namespace skyrmion
