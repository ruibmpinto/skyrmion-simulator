// Initial spin configurations: single Neel skyrmion, uniform state,
// SAF skyrmion pair (antiparallel cores).
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {

// One Neel skyrmion on (ny, nx) lattice centered at the box middle.
// polarity == +1: core at m_z = -1; polarity == -1: core at m_z = +1.
Field3 skyrmion_profile(int nx, int ny, Real a, Real R, Real dw, int polarity);

// Uniform state. `dir` must be a unit vector (will be normalized).
Field3 uniform_state(int nx, int ny, const Vec3& dir);

// SAF ground state: top layer core down, bottom layer core up.
struct SAFPair { Field3 m_top; Field3 m_bot; };
SAFPair saf_skyrmion(int nx, int ny, Real a, Real R, Real dw);

} // namespace skyrmion
