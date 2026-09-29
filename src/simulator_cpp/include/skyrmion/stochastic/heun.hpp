/// \file
/// Stratonovich-Heun stepper for the stochastic SAF LLGS. Port of
/// src/skyrmion_simulator/stochastic_llgs/integrator_sllg.py::heun_stochastic_step.
///
/// A predictor-corrector RK2 (NOT rk4_step): assemble H from m only,
/// add the pre-sampled thermal field, evaluate llgs_rhs; the SAME
/// noise sample is reused in predictor and corrector (this is what
/// makes the scheme converge to the Stratonovich SDE). The predictor
/// is not renormalised; one end-of-step renorm restores |m| = 1 after
/// a loud norm-drift check.
///
/// Implements sweep::Stepper, so it also drops into sweep::run_trace /
/// sweep::relax unchanged. demag == nullptr selects the local-K_eff
/// field; non-null selects bare-K + FFT demag. sigma == 0 draws no
/// noise (the deterministic T = 0 limit, bit-parity with Python).
#pragma once

#include "skyrmion/demag.hpp"
#include "skyrmion/parameters.hpp"
#include "skyrmion/stochastic/rng.hpp"
#include "skyrmion/sweep/stepper.hpp"
#include "skyrmion/types.hpp"

namespace skyrmion {
namespace stochastic {

/// Stochastic Heun (Stratonovich) stepper for the SAF LLGS.
///
/// `mask` (row-major ny*nx, 1 = inside the magnetic region, or nullptr
/// for the periodic path) is forwarded to the field assembly for
/// free-boundary geometries; the noise sample is zeroed outside the
/// region so frozen cells do not random-walk. Single-layer mode is
/// selected by passing an empty `m_bot` (n_sites() == 0) to step(): a
/// lone ferromagnet is evolved with the top layer as its own RKKY
/// partner; it requires demag == nullptr (demag is a bilayer
/// coupling), else step() raises. Matches Python
/// integrator_sllg.heun_stochastic_step(mask=..., m_bot=None).
class HeunStochasticStepper : public sweep::Stepper {
public:
    /// Construct the stepper and size its scratch fields.
    /// \param p Parameter set; only p.ny and p.nx are read here, to
    ///        allocate the scratch fields.
    /// \param demag Demag state, or nullptr for the local-K_eff field
    ///        model. Borrowed, not owned; must outlive the stepper.
    /// \param rng Thermal-noise source, held by reference. Its
    ///        counter-based stream makes the draw independent of the
    ///        OpenMP thread count.
    /// \param sigma Thermal-field amplitude (T*sqrt(s), p.sigma_noise);
    ///        finite and >= 0, with 0 the deterministic T = 0 limit.
    /// \param tol_norm Maximum tolerated per-site |m| drift over one
    ///        step before step() raises; finite and > 0.
    /// \param mask Row-major ny*nx region mask, or nullptr for the
    ///        fully periodic path. Borrowed, not owned.
    /// \throws std::runtime_error if tol_norm is non-finite or
    ///         non-positive, or sigma non-finite or negative.
    HeunStochasticStepper(const Params& p, DemagState* demag,
                          ThermalRng& rng, Real sigma, Real tol_norm,
                          const std::uint8_t* mask);

    /// Advance both layers by one Stratonovich-Heun step.
    ///
    /// Draws the thermal field once per step (one draw per layer, none
    /// when sigma == 0) and reuses the SAME sample in predictor and
    /// corrector, which is what makes the scheme converge to the
    /// Stratonovich SDE. Renormalises |m| once at the end of the step.
    /// \param m_top Top-layer magnetization, updated in place.
    /// \param m_bot Bottom-layer magnetization, updated in place; an
    ///        empty field selects single-layer mode.
    /// \param t Current time (s), passed to llgs_rhs for the pulse.
    /// \param dt Step size (s); the noise is scaled by 1/sqrt(dt).
    /// \param p Parameter set used for the field assembly and the LLGS
    ///        right-hand side.
    /// \throws std::runtime_error in single-layer mode with a non-null
    ///         demag, or when the end-of-step norm drift exceeds
    ///         tol_norm.
    void step(Field3& m_top, Field3& m_bot,
              Real t, Real dt, Params& p) override;

    /// \return Largest per-site deviation of |m| from 1 seen in the
    ///         last step(), before the end-of-step renormalisation.
    Real last_norm_drift() const { return last_drift_; }

private:
    DemagState* demag_;
    ThermalRng& rng_;
    Real sigma_;
    Real tol_norm_;
    const std::uint8_t* mask_;
    Real last_drift_ = 0.0;
    // Scratch (sized at construction).
    Field3 H_top_, H_bot_;
    Field3 h_top_, h_bot_;
    Field3 f1_top_, f1_bot_, f2_top_, f2_bot_;
    Field3 mp_top_, mp_bot_;

    /// Assemble H_top_/H_bot_ from (m_top, m_bot) using the configured
    /// field model, then add the current noise sample h_top_/h_bot_.
    /// `single` evolves the top layer alone (m_bot ignored).
    ///
    /// The noise is added to the assembled H, never inside the field
    /// assembly, so the exchange/DMI difference operators act on m
    /// only.
    void field_plus_noise(const Field3& m_top, const Field3& m_bot,
                          Params& p, bool single);
};

} // namespace stochastic
} // namespace skyrmion
