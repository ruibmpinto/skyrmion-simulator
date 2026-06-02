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
//
// `mask` is a row-major (ny*nx) array of 0/1 flags marking the magnetic
// region for a free-boundary geometry (1 = inside). Pass `nullptr` for
// the periodic path, which is byte-identical to the unmasked code. With
// a mask, out-of-region neighbours are replaced by the Rohart-Thiaville
// / Bogdanov-Roesler ghost-cell extrapolation (xi_inv_a = C_dmi/C_ex
// when C_ex > 0, else 0) shared by the exchange and DMI stencils, and
// the field is zeroed outside the region. Matches Python
// fields.effective_field(mask=...).
void effective_field(const Field3& m, const Field3& m_other,
                     Real C_ex, Real C_dmi, Real C_anis,
                     const Vec3& H_ext, Real H_RKKY,
                     Field3& H, const std::uint8_t* mask);

// Bare-K + demag pair path (matches Python effective_field_demag_pair).
//
// With a non-null `mask`, the local terms use the free-boundary path
// above, the magnetisation is zeroed outside the region before the demag
// convolution, and the returned demag field is zeroed outside the region.
// Pass `nullptr` for the periodic path.
void effective_field_demag(const Field3& m_top, const Field3& m_bot,
                           const Params& p, DemagState& demag,
                           Field3& H_top, Field3& H_bot,
                           const std::uint8_t* mask);

} // namespace skyrmion
