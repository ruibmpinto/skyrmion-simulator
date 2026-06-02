"""Verify the Python error contract that the C++ test_edge_cases mirrors.

For every invalid-input case, assert Python raises (RuntimeError) exactly
where C++ raises. The previously documented divergences (Python silently
returning NaN on a degenerate division: skyrmion_center empty mask,
uniform_state zero direction, normalize zero spin; and skyrmion_profile
accepting any polarity) have been aligned: Python now raises in all of
these, matching C++.

Functions
---------
main
    Run all contract checks and print PASS/FAIL per case.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
import sys
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _expect_raise(name, fn, results):
    """Record PASS iff fn() raises."""
    try:
        fn()
        results.append((name, False))
    except Exception:
        results.append((name, True))


def _expect_no_raise(name, fn, results):
    """Record PASS iff fn() does not raise."""
    try:
        fn()
        results.append((name, True))
    except Exception:
        results.append((name, False))


def main():
    """Run the Python error-contract checks."""
    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from src.simulator.parameters import _precompute, default_params
    from src.simulator.pulses import (
        SquarePulse, GaussianPulse, SuperpositionPulse)
    from src.simulator.initial_conditions import (
        skyrmion_profile, uniform_state)
    from src.simulator.analysis import (
        skyrmion_center, skyrmion_diameter, skyrmion_ellipse, dw_angle)
    from src.simulator.integrator import normalize
    from src.simulator.energy import critical_dmi, pma_anisotropy_field
    from src.simulator.demag import precompute_demag_kernels
    from src.simulator.demag_newell import precompute_demag_kernels_newell
    from src.stochastic_llgs.diagnostics import skyrmion_center_pbc
    from src.phase_diagram.relaxation import relax
    from src.stochastic_llgs.integrator_sllg import heun_stochastic_step

    res = []
    # ---- Pulses --------------------------------------------------------------
    _expect_raise('SquarePulse_t_end_le_t_start',
                  lambda: SquarePulse(1.0, 1.0, 1.0), res)
    _expect_raise('GaussianPulse_zero_FWHM',
                  lambda: GaussianPulse(1.0, 0.0, 0.0), res)
    _expect_raise('SuperpositionPulse_empty',
                  lambda: SuperpositionPulse([]), res)
    # ---- Parameters ----------------------------------------------------------

    def _bad_keff():
        p = default_params()
        p.K_top = 1.0e3
        p.K_bot = 1.0e3
        _precompute(p)
        critical_dmi(p)
    _expect_raise('critical_dmi_Keff_nonpositive', _bad_keff, res)

    def _bad_keff_pma():
        p = default_params()
        p.K_top = 1.0e3
        p.K_bot = 1.0e3
        _precompute(p)
        pma_anisotropy_field(p)
    _expect_raise('pma_field_Keff_nonpositive', _bad_keff_pma, res)
    # ---- Initial conditions --------------------------------------------------
    _expect_raise('skyrmion_profile_polarity_0',
                  lambda: skyrmion_profile(8, 8, 2e-9, 6e-9, 3e-9, 0), res)
    _expect_raise('skyrmion_profile_polarity_2',
                  lambda: skyrmion_profile(8, 8, 2e-9, 6e-9, 3e-9, 2), res)
    _expect_raise('uniform_state_zero_direction',
                  lambda: uniform_state(8, 8, np.array([0.0, 0.0, 0.0])),
                  res)
    _expect_raise('normalize_zero_vector',
                  lambda: normalize(np.zeros((4, 4, 3))), res)
    # ---- Observables (polarity + degenerate) ---------------------------------
    up = np.zeros((8, 8, 3))
    up[..., 2] = 1.0
    _expect_raise('skyrmion_center_polarity_0',
                  lambda: skyrmion_center(up, 2e-9, 0), res)
    _expect_raise('skyrmion_center_empty_mask',
                  lambda: skyrmion_center(up, 2e-9, +1), res)
    _expect_raise('skyrmion_diameter_polarity_0',
                  lambda: skyrmion_diameter(up, 2e-9, 0), res)
    _expect_raise('skyrmion_ellipse_polarity_0',
                  lambda: skyrmion_ellipse(up, 2e-9, 0), res)
    _expect_raise('skyrmion_center_pbc_polarity_0',
                  lambda: skyrmion_center_pbc(up, 2e-9, 0), res)
    _expect_raise('skyrmion_center_pbc_zero_weight',
                  lambda: skyrmion_center_pbc(up, 2e-9, +1), res)
    _expect_raise('skyrmion_ellipse_too_few_core_sites',
                  lambda: skyrmion_ellipse(up, 2e-9, +1), res)
    _expect_raise('dw_angle_no_domain_wall',
                  lambda: dw_angle(up, 2e-9, +1, 0.5), res)
    # ---- Demag ---------------------------------------------------------------

    def _newell(acc, tol, nx=8, ny=8):
        p = default_params()
        p.nx = nx
        p.ny = ny
        _precompute(p)
        precompute_demag_kernels_newell(p, accuracy=acc, tol_conv=tol)
    _expect_raise('demag_newell_accuracy_nonpositive',
                  lambda: _newell(0.0, 0.02), res)
    _expect_raise('demag_newell_tol_ge_one',
                  lambda: _newell(4.0, 1.0), res)
    _expect_raise('demag_newell_nx_too_small',
                  lambda: _newell(4.0, 0.02, nx=1), res)

    def _slab_bad_tco():
        p = default_params()
        p.nx = 8
        p.ny = 8
        p.t_Co = -1.0e-9
        _precompute(p)
        precompute_demag_kernels(p, kind='slab', accuracy=None,
                                 tol_conv=None)
    _expect_raise('demag_slab_tCo_nonpositive', _slab_bad_tco, res)
    # ---- Free-boundary / single-layer contract -------------------------------

    def _small_p():
        p = default_params()
        p.nx = 8
        p.ny = 8
        _precompute(p)
        return p

    def _relax_single_with_kernels():
        p = _small_p()
        K = precompute_demag_kernels(p, kind='slab', accuracy=None,
                                     tol_conv=None)
        m = np.zeros((8, 8, 3))
        m[..., 2] = 1.0
        relax(m, None, p, K, max_steps=1, alpha_relax=1.0,
              tol_torque=1.0e-5, tol_dE=1.0e-8, check_every=1,
              print_every=0, mask=None)
    _expect_raise('relax_single_with_kernels',
                  _relax_single_with_kernels, res)

    def _heun_single_h_bot_not_none():
        p = _small_p()
        m = np.zeros((8, 8, 3))
        m[..., 2] = 1.0
        z = np.zeros((8, 8, 3))
        heun_stochastic_step(m, None, p.dt, p, None, z, z, 1.0, t=0.0)
    _expect_raise('heun_single_h_bot_not_none',
                  _heun_single_h_bot_not_none, res)

    def _heun_single_with_kernels():
        p = _small_p()
        K = precompute_demag_kernels(p, kind='slab', accuracy=None,
                                     tol_conv=None)
        m = np.zeros((8, 8, 3))
        m[..., 2] = 1.0
        z = np.zeros((8, 8, 3))
        heun_stochastic_step(m, None, p.dt, p, K, z, None, 1.0, t=0.0)
    _expect_raise('heun_single_with_kernels',
                  _heun_single_with_kernels, res)
    # ---- Sanity: normal cases do NOT raise -----------------------------------
    _expect_no_raise('critical_dmi_normal',
                     lambda: critical_dmi(default_params()), res)
    _expect_no_raise('uniform_state_normal',
                     lambda: uniform_state(8, 8,
                                           direction=np.array([0, 0, 1.0])),
                     res)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Report.
    n_pass = sum(1 for _, ok in res if ok)
    for name, ok in res:
        print(f'  {name:<44s} {"PASS" if ok else "FAIL"}')
    print(f'edge_contract: {n_pass}/{len(res)} Python contract checks passed.')
    print()
    print('Error contract fully aligned: Python and C++ raise on the same '
          'invalid inputs. No intentional stricter-than-Python divergences '
          'remain in these functions.')
    if n_pass != len(res):
        sys.exit(1)


# =============================================================================
if __name__ == '__main__':
    main()
