/// \file
/// Initial spin configurations: single Neel skyrmion, uniform state,
/// SAF skyrmion pair (antiparallel cores).
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {

/// One Neel skyrmion on (ny, nx) lattice centered at the box middle.
/// polarity == +1: core at m_z = -1; polarity == -1: core at m_z = +1.
/// The wall follows the standard 360-degree profile
/// theta(r) = 2 atan(exp(-(r - R) / dw)) with outward radial in-plane
/// component (Neel chirality).
/// \param nx Number of cells along x.
/// \param ny Number of cells along y.
/// \param a Cell size in metres.
/// \param R Skyrmion radius in metres.
/// \param dw Domain-wall width in metres.
/// \param polarity Core sign, +1 or -1 as described above.
/// \return Unit magnetisation field of shape (ny, nx, 3).
/// \throws std::runtime_error If polarity is neither +1 nor -1.
Field3 skyrmion_profile(int nx, int ny, Real a, Real R, Real dw, int polarity);

/// Uniform state. `dir` must be a unit vector (will be normalized).
/// \param nx Number of cells along x.
/// \param ny Number of cells along y.
/// \param dir Direction of the uniform magnetisation.
/// \return Unit magnetisation field of shape (ny, nx, 3).
/// \throws std::runtime_error If `dir` has zero length.
Field3 uniform_state(int nx, int ny, const Vec3& dir);

/// SAF ground state: top layer core down, bottom layer core up.
struct SAFPair { Field3 m_top; Field3 m_bot; };
/// The bottom layer is the full flip -m_top, so both layers carry the
/// DMI-favoured chirality and the wall rings are antiparallel.
/// \param nx Number of cells along x.
/// \param ny Number of cells along y.
/// \param a Cell size in metres.
/// \param R Skyrmion radius in metres.
/// \param dw Domain-wall width in metres.
/// \return The two coupled layer configurations.
SAFPair saf_skyrmion(int nx, int ny, Real a, Real R, Real dw);

} // namespace skyrmion
