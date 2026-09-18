/// \file
/// Split the magnetisation rate into translation and deformation.
///
/// The LLG damping dissipates power proportional to the integral of
/// |dm/dt|^2 over the film. A rigidly translating texture obeys
/// dm/dt = -(v . grad) m exactly, so projecting dm/dt onto the space
/// spanned by the two spatial-gradient fields grad_x m and grad_y m
/// recovers the collective velocity v and separates the rate into
///
///   translational : the part explained by rigid motion at velocity v,
///   deformation   : the orthogonal residual, i.e. shape change.
///
/// v is the least-squares minimiser of |dm/dt + v . grad m|^2, so the
/// two parts are orthogonal and their integrals sum to the total. The
/// ratio translational / total is dimensionless and independent of the
/// volume element and the LLG prefactor, so it answers "what fraction
/// of the dissipation moves the skyrmion rather than deforms it"
/// directly.
///
/// dm/dt must be the DETERMINISTIC rate (no thermal noise term): the
/// aim is coherent motion, and the random field would otherwise
/// dominate |dm/dt|^2 with jitter that is neither translation nor
/// deformation.
#pragma once

#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

/// Result of the translation / deformation partition of dm/dt.
struct DissipationSplit {
    /// Integrals of |.|^2 over the lattice, times the cell area a^2.
    /// The film thickness and the LLG damping prefactor are NOT
    /// applied, so these are proportional to, not equal to, dissipated
    /// power; the trans / (trans + def) ratio is prefactor-independent.
    double trans = 0.0;   ///< rigid-translation part
    double def = 0.0;     ///< deformation (residual) part
    double total = 0.0;   ///< int |dm/dt|^2 a^2, = trans + def to round-off
    /// Recovered collective velocity (m/s), a cross-check against the
    /// centre-of-mass tracking.
    double v_x = 0.0;
    double v_y = 0.0;     ///< Fitted collective velocity along y (m/s)
};

/// Partition `dmdt` against the spatial gradients of `m`.
///
/// `a` is the lattice constant (m). `free_y` selects the transverse
/// boundary of the gradient stencil: false wraps y periodically
/// (Newell), true takes one-sided differences at the y edges
/// (racetrack). x is always periodic. `m` and `dmdt` must share the
/// lattice shape.
/// \param m Magnetization field, shape (ny, nx, 3), |m| = 1.
/// \param dmdt Deterministic magnetization rate (1/s), same shape as
///        m; must exclude the thermal-noise contribution.
/// \param a Lattice constant (m); must be positive.
/// \param free_y True for one-sided y differences at the transverse
///        edges (racetrack); false for a periodic y stencil.
/// \return The two parts and their total, plus the fitted velocity.
///         With a vanishing normal-equation determinant (a uniform
///         field, no gradient) the velocity stays zero and the whole
///         total is reported as deformation.
/// \throws std::runtime_error if m and dmdt have different lattice
///         shapes, or if a is not positive.
DissipationSplit dissipation_partition(const Field3& m, const Field3& dmdt,
                                       Real a, bool free_y);

} // namespace stochastic
} // namespace skyrmion
