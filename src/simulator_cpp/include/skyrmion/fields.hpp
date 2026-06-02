// Effective field contributions for the LLGS equation. All fields are
// in Tesla so the LLG carries gamma in rad/(s*T) and no explicit mu0.
//
// Two effective-field paths:
//   effective_field          : local-K_eff (uniform demag folded into K),
//                              no explicit FFT demag.
//   effective_field_demag    : bare-K anisotropy + explicit FFT demag.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {

// Local-K_eff path (matches Python rhs_local_keff). H is written, not added.
void effective_field(const Field3& m, const Field3& m_other,
                     Real C_ex, Real C_dmi, Real C_anis,
                     const Vec3& H_ext, Real H_RKKY,
                     Field3& H);

// Bare-K + demag pair path (matches Python effective_field_demag_pair).
void effective_field_demag(const Field3& m_top, const Field3& m_bot,
                           const Params& p, DemagState& demag,
                           Field3& H_top, Field3& H_bot);

} // namespace skyrmion
