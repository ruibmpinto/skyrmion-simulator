/// \file
/// Effective field contributions for the LLGS equation. All fields are
/// in Tesla so the LLG carries gamma in rad/(s*T) and no explicit mu0.
///
/// Two effective-field paths:
///   effective_field          : local-K_eff (uniform demag folded into K),
///                              no explicit FFT demag.
///   effective_field_demag    : bare-K anisotropy + explicit FFT demag.
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {

/// Local-K_eff path (matches Python rhs_local_keff). H is written, not added.
///
/// Assembles exchange, interfacial Neel DMI, perpendicular anisotropy,
/// Zeeman and RKKY in one fused nearest-neighbour loop.
///
/// `mask` is a row-major (ny*nx) array of 0/1 flags marking the magnetic
/// region for a free-boundary geometry (1 = inside). Pass `nullptr` for
/// the periodic path, which is byte-identical to the unmasked code. With
/// a mask, out-of-region neighbours are replaced by the Rohart-Thiaville
/// / Bogdanov-Roesler ghost-cell extrapolation (xi_inv_a = C_dmi/C_ex
/// when C_ex > 0, else 0) shared by the exchange and DMI stencils, and
/// the field is zeroed outside the region. Matches Python
/// fields.effective_field(mask=...).
///
/// `free_y` selects the racetrack boundary: periodic along x, free
/// top/bottom along y over the full box. The y-neighbour of row 0 (-y)
/// and of row ny-1 (+y) is taken as outside, so it uses the same R-T
/// ghost as a mask edge; x stays periodic and no cells are zeroed. This
/// is the exchange/DMI analogue of the Racetrack demag (free-y over
/// the full box). free_y composes with mask (a neighbour is "outside" if
/// the mask says so OR it is across the free y-edge).
/// \param m Magnetisation direction field of this layer.
/// \param m_other Magnetisation of the RKKY partner layer.
/// \param C_ex Exchange prefactor 2 A_ex / (Ms a^2), in Tesla.
/// \param C_dmi DMI prefactor D / (Ms a), in Tesla.
/// \param C_anis Perpendicular anisotropy prefactor, in Tesla.
/// \param H_ext Uniform external field, in Tesla.
/// \param H_RKKY Interlayer coupling field, in Tesla; enters as
///        -H_RKKY * m_other (antiparallel coupling).
/// \param H Effective field (T), shape (ny, nx, 3), overwritten.
/// \param mask Free-boundary region flags, or nullptr for periodic.
/// \param free_y True for the racetrack free top/bottom y-edges.
void effective_field(const Field3& m, const Field3& m_other,
                     Real C_ex, Real C_dmi, Real C_anis,
                     const Vec3& H_ext, Real H_RKKY,
                     Field3& H, const std::uint8_t* mask, bool free_y);

/// Bare-K + demag pair path (matches Python effective_field_demag_pair).
///
/// Uses bare_anis_prefactors so the slab demag is not double-counted,
/// selects free-y exchange/DMI when `p.demag_kind` is Racetrack, and
/// adds the FFT demag field of both layers on top.
///
/// With a non-null `mask`, the local terms use the free-boundary path
/// above, the magnetisation is zeroed outside the region before the demag
/// convolution, and the returned demag field is zeroed outside the region.
/// Pass `nullptr` for the periodic path.
/// \param m_top Top-layer magnetisation direction field.
/// \param m_bot Bottom-layer magnetisation direction field.
/// \param p Parameter set supplying the prefactors and demag_kind.
/// \param demag Demag state holding the kernel and FFT plans.
/// \param H_top Top-layer effective field (T), overwritten.
/// \param H_bot Bottom-layer effective field (T), overwritten.
/// \param mask Free-boundary region flags, or nullptr for periodic.
void effective_field_demag(const Field3& m_top, const Field3& m_bot,
                           const Params& p, DemagState& demag,
                           Field3& H_top, Field3& H_bot,
                           const std::uint8_t* mask);

} // namespace skyrmion
