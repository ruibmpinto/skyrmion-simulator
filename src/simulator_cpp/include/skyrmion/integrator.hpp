/// \file
/// LLGS time integration. Mirrors src/skyrmion_simulator/simulator/integrator.py.
///
/// Two RHS variants:
///   RHSLocalKeff : local-K_eff effective field (no explicit demag).
///   RHSDemag     : bare-K effective field + FFT demag (slab kernel).
///
/// Both implement the protocol:
///   void operator()(const Field3& m_top, const Field3& m_bot,
///                   Real t, Field3& dmdt_top, Field3& dmdt_bot);
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/types.hpp"

#include <cstdint>
#include <memory>

namespace skyrmion {

/// Rescale every spin of `m` to unit length, in place.
/// \param m Magnetisation field, normalised in place.
/// \throws std::runtime_error If any site has zero magnitude.
void normalize_inplace(Field3& m);

/// dm/dt for a single layer from the explicit LLGS equation with SOT and
/// optional topological spin Hall torque. H_eff is provided by the caller.
/// `mask` (row-major ny*nx, 1 = inside, or nullptr = periodic): vacuum
/// sites get dm/dt = 0 exactly -- the SOT/TSH torques do not depend on
/// H_eff and would otherwise rotate vacuum spins. TSH + mask throws.
/// The instantaneous current is J(t) = (*p.pulse)(t); both SOT terms
/// scale with it, and the spin direction is p_hat x m (unnormalised
/// Slonczewski form).
/// \param m Magnetisation direction field of this layer.
/// \param H_eff Effective field of this layer, in Tesla.
/// \param p Parameter set: gamma_p, alpha, SOT and TSH coefficients.
/// \param t Time in seconds, passed to the current pulse.
/// \param dmdt Time derivative (1/s), shape (ny, nx, 3), overwritten.
/// \param mask Free-boundary region flags, or nullptr for periodic.
/// \throws std::runtime_error If p.lambda_sq != 0 and `mask` is
///         non-null: the TSH gradient stencil would read vacuum spins.
void llgs_rhs(const Field3& m, const Field3& H_eff, const Params& p, Real t,
              Field3& dmdt, const std::uint8_t* mask);

/// Two-layer RHS functor on the local-K_eff path (no explicit demag).
///
/// `mask` (row-major ny*nx, 1 = inside the magnetic region, or nullptr for
/// the periodic path) is forwarded to effective_field for free-boundary
/// geometries. Stored as a borrowed pointer; the array must outlive the
/// functor.
class RHSLocalKeff {
public:
    /// \param p Parameter set; stored by reference, must outlive this.
    /// \param mask Free-boundary region flags, or nullptr.
    RHSLocalKeff(const Params& p, const std::uint8_t* mask);
    /// Evaluates both layers' dm/dt at time `t`.
    /// \param m_top Top-layer magnetisation.
    /// \param m_bot Bottom-layer magnetisation.
    /// \param t Time in seconds.
    /// \param dmdt_top Top-layer derivative (1/s), overwritten.
    /// \param dmdt_bot Bottom-layer derivative (1/s), overwritten.
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    const std::uint8_t* mask_;
    Field3 H_top_;
    Field3 H_bot_;
};

/// Two-layer RHS functor on the bare-K + explicit FFT demag path.
/// `mask` is borrowed exactly as in RHSLocalKeff.
class RHSDemag {
public:
    /// \param p Parameter set; stored by reference, must outlive this.
    /// \param demag Demag state; stored by reference, must outlive this.
    /// \param mask Free-boundary region flags, or nullptr.
    RHSDemag(const Params& p, DemagState& demag, const std::uint8_t* mask);
    /// Evaluates both layers' dm/dt at time `t`.
    /// \param m_top Top-layer magnetisation.
    /// \param m_bot Bottom-layer magnetisation.
    /// \param t Time in seconds.
    /// \param dmdt_top Top-layer derivative (1/s), overwritten.
    /// \param dmdt_bot Bottom-layer derivative (1/s), overwritten.
    void operator()(const Field3& m_top, const Field3& m_bot, Real t,
                    Field3& dmdt_top, Field3& dmdt_bot);
private:
    const Params& p_;
    DemagState& demag_;
    const std::uint8_t* mask_;
    Field3 H_top_;
    Field3 H_bot_;
};

/// Single-layer local-K_eff RHS (matches Python relaxation._rhs_single).
/// A lone ferromagnet: the layer is its own RKKY partner (harmless when
/// the caller sets H_RKKY = 0). Implements the rk4_step_single protocol:
///   void operator()(const Field3& m, Real t, Field3& dmdt);
/// The top-layer anisotropy prefactor C_anis_top is used.
class RHSSingleKeff {
public:
    /// \param p Parameter set; stored by reference, must outlive this.
    /// \param mask Free-boundary region flags, or nullptr.
    RHSSingleKeff(const Params& p, const std::uint8_t* mask);
    /// Evaluates the layer's dm/dt at time `t`.
    /// \param m Magnetisation direction field.
    /// \param t Time in seconds.
    /// \param dmdt Time derivative (1/s), overwritten.
    void operator()(const Field3& m, Real t, Field3& dmdt);
private:
    const Params& p_;
    const std::uint8_t* mask_;
    Field3 H_;
};

/// One classical RK4 step, both layers in lockstep. Intermediate stages
/// are renormalised to keep |m| = 1.
/// Explicitly instantiated for RHSLocalKeff and RHSDemag.
/// \tparam RHS Callable following the two-layer RHS protocol.
/// \param rhs Right-hand side evaluated four times per step.
/// \param m_top Top-layer magnetisation, advanced in place.
/// \param m_bot Bottom-layer magnetisation, advanced in place.
/// \param t Time at the start of the step, in seconds.
/// \param dt Step size in seconds.
/// \param p Parameter set supplying the grid size.
/// \throws std::runtime_error If a spin reaches zero magnitude during
///         a stage or the final normalisation.
template <typename RHS>
void rk4_step(RHS& rhs, Field3& m_top, Field3& m_bot,
              Real t, Real dt, const Params& p);

/// One classical RK4 step for a single layer (matches Python
/// integrator.rk4_step_single). `rhs` is a callable
///   void operator()(const Field3& m, Real t, Field3& dmdt);
/// Intermediate stages are renormalised to keep |m| = 1.
/// Explicitly instantiated for RHSSingleKeff.
/// \tparam RHS Callable following the single-layer RHS protocol.
/// \param rhs Right-hand side evaluated four times per step.
/// \param m Magnetisation, advanced in place.
/// \param t Time at the start of the step, in seconds.
/// \param dt Step size in seconds.
/// \param p Parameter set supplying the grid size.
/// \throws std::runtime_error If a spin reaches zero magnitude during
///         a stage or the final normalisation.
template <typename RHS>
void rk4_step_single(RHS& rhs, Field3& m, Real t, Real dt, const Params& p);

} // namespace skyrmion
