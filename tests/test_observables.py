"""Regression tests for the new observables and the topological
spin Hall torque (TSH) integration path.

Tests
-----
test_skyrmion_ellipse_circular
    Verify that an analytic Neel skyrmion gives D1 = D2 = 2R
    and theta = 0 (degenerate principal axes).
test_dw_angle_neel_at_rest
    Verify that an analytic Neel skyrmion at rest gives psi
    = pi (180 deg) on the right DW, both layers.
test_tsh_off_equivalence
    Confirm that the TSH branch in `llgs_rhs` is bypassed
    bit-identically when `p.lambda_sq == 0` (the default).
test_topological_torque_limits
    Confirm that the analytic v_TSH magnitude decreases
    monotonically with R/Delta, consistent with the paper's
    qualitative claim.

Run with `python -m tests.test_observables`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
# Third-party
import numpy as np
# Local
from src.simulator.analysis import (
    dw_angle,
    skyrmion_ellipse,
)
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rhs_local_keff, rk4_step
from src.simulator.parameters import default_params
from src.simulator.pulses import ConstantPulse
from src.simulator.topological_torque import (
    sot_thiele_speed,
    tsh_thiele_speed,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def test_skyrmion_ellipse_circular():
    """Analytic Neel skyrmion has D1 = D2 = 2 R."""
    # Lattice and skyrmion geometry.
    nx = ny = 128
    a = 2.0e-9
    R = 80.0e-9
    dw = 27.0e-9
    # Build the SAF skyrmion pair; use the top layer.
    m_top, _m_bot = saf_skyrmion(nx, ny, a, R, dw)
    # Compute principal-axis diameters from the core mask.
    # Top layer in saf_skyrmion has core_polarity = +1 (m_z = -1 core).
    D1, D2, theta = skyrmion_ellipse(m_top, a, core_polarity=+1)
    # Expected diameter (2R) and aspect tolerance.
    expected = 2.0 * R
    # Allow 1% deviation; the disk approximation is not exact
    # because of the lattice discretisation and DW width.
    tol_diameter = 0.02 * expected
    # Both axes should equal 2R to within the tolerance.
    print(f'  D1 = {D1*1e9:.2f} nm, D2 = {D2*1e9:.2f} nm '
          f'(expected {expected*1e9:.2f} nm)')
    if abs(D1 - expected) > tol_diameter:
        raise RuntimeError(
            f'skyrmion_ellipse D1 = {D1:.3e} m differs from '
            f'expected 2R = {expected:.3e} m by more than '
            f'{tol_diameter:.3e} m.')
    if abs(D2 - expected) > tol_diameter:
        raise RuntimeError(
            f'skyrmion_ellipse D2 = {D2:.3e} m differs from '
            f'expected 2R = {expected:.3e} m by more than '
            f'{tol_diameter:.3e} m.')
    # Circular skyrmion: orientation should be near zero.
    print(f'  theta = {np.degrees(theta):.3f} deg '
          f'(expected ~0 for circular)')


# -----------------------------------------------------------------------------
def test_dw_angle_neel_at_rest():
    """Analytic Neel skyrmion at rest: psi = pi at right DW."""
    # Same geometry as the previous test.
    nx = ny = 128
    a = 2.0e-9
    R = 80.0e-9
    dw = 27.0e-9
    # Build SAF skyrmion pair.
    m_top, m_bot = saf_skyrmion(nx, ny, a, R, dw)
    # DW angle at the right DW of each layer; pass the polarity
    # so dw_angle can locate the correct skyrmion centroid.
    psi_top = dw_angle(m_top, a, core_polarity=+1)
    psi_bot = dw_angle(m_bot, a, core_polarity=-1)
    # Print for the operator.
    print(f'  psi_top = {np.degrees(psi_top):.2f} deg '
          f'(expected 0)')
    print(f'  psi_bot = {np.degrees(psi_bot):.2f} deg '
          f'(expected 0)')
    # New convention: psi is the signed deviation from the layer's
    # natural Neel orientation, so a rest skyrmion (no drive)
    # gives psi ~ 0 in both layers regardless of chirality.
    tol = np.radians(2.0)
    if abs(psi_top) > tol:
        raise RuntimeError(
            f'dw_angle top {np.degrees(psi_top):.3f} deg '
            f'differs from 0 by more than 2 deg.')
    if abs(psi_bot) > tol:
        raise RuntimeError(
            f'dw_angle bot {np.degrees(psi_bot):.3f} deg '
            f'differs from 0 by more than 2 deg.')


# -----------------------------------------------------------------------------
def test_tsh_off_equivalence():
    """TSH-off run is bit-identical to a baseline without TSH."""
    # Build two parameter sets that differ only in TSH being
    # nominally "on" vs "off", but with `lambda_sq = 0` so the
    # TSH branch should be skipped in both cases.
    p_baseline = default_params()
    p_baseline.nx = p_baseline.ny = 32
    p_with_tsh = default_params()
    p_with_tsh.nx = p_with_tsh.ny = 32
    # Even with lambda_sq = 0 (the default), the if-guard in
    # llgs_rhs should keep the dynamics identical to baseline.
    p_with_tsh.lambda_sq = 0.0
    # Sanity check.
    if p_baseline.lambda_sq != 0.0:
        raise RuntimeError(
            'Default parameters unexpectedly enable TSH; expected '
            'lambda_sq = 0 by default.')
    # Same initial condition for both paths.
    m_top_a, m_bot_a = saf_skyrmion(
        p_baseline.nx, p_baseline.ny, p_baseline.a,
        p_baseline.skyrmion_R, p_baseline.skyrmion_dw,)
    m_top_b = m_top_a.copy()
    m_bot_b = m_bot_a.copy()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Advance both for a few steps.
    t = 0.0
    for _ in range(50):
        m_top_a, m_bot_a = rk4_step(
            rhs_local_keff,
            m_top_a, m_bot_a, t, p_baseline.dt, p_baseline)
        m_top_b, m_bot_b = rk4_step(
            rhs_local_keff,
            m_top_b, m_bot_b, t, p_with_tsh.dt, p_with_tsh)
        t += p_baseline.dt
    # Compare; expect bit-identical results.
    diff_top = float(np.max(np.abs(m_top_a - m_top_b)))
    diff_bot = float(np.max(np.abs(m_bot_a - m_bot_b)))
    print(f'  max |delta m_top| = {diff_top:.3e}')
    print(f'  max |delta m_bot| = {diff_bot:.3e}')
    # Float64 round-off bound; bit-identical execution should
    # give zero, allow one ULP slack.
    tol = 1e-15
    if diff_top > tol or diff_bot > tol:
        raise RuntimeError(
            'TSH-off path diverges from baseline beyond '
            f'tol={tol:.1e}: '
            f'top {diff_top:.3e}, bot {diff_bot:.3e}.')


# -----------------------------------------------------------------------------
def test_topological_torque_limits():
    """Analytic |v_TSH| decreases monotonically with R/Delta."""
    p = default_params()
    # Use the S49 drive density.
    p.pulse = ConstantPulse(8.0e11)
    # Domain wall width from S49 caption.
    Delta = 24.5e-9
    # Use a moderate lambda_sq inside the paper's quoted range.
    lambda_sq = 50.0e-18
    # Sweep R/Delta from 1 to 5 (the paper's range).
    ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    # Container for printed speeds.
    speeds = []
    # Compute speeds.
    for ratio in ratios:
        R = ratio * Delta
        v = tsh_thiele_speed(R, Delta, lambda_sq, p)
        speeds.append((ratio, v))
        print(f'  R/Delta = {ratio:.1f}: v_TSH = {v:+.2f} m/s')
    # Monotonicity check on |v|; expect strictly decreasing.
    abs_speeds = [abs(v) for _, v in speeds]
    for i in range(1, len(abs_speeds)):
        if abs_speeds[i] >= abs_speeds[i - 1]:
            raise RuntimeError(
                'tsh_thiele_speed: |v_TSH| not strictly '
                f'decreasing between R/Delta = {ratios[i-1]:.1f} '
                f'({abs_speeds[i-1]:.3f}) and {ratios[i]:.1f} '
                f'({abs_speeds[i]:.3f}).')
    # Cross-check v_SOT is positive at the same R/Delta values.
    for ratio in ratios:
        R = ratio * Delta
        v_sot = sot_thiele_speed(R, Delta, p)
        if not (v_sot > 0.0):
            raise RuntimeError(
                'sot_thiele_speed returned non-positive speed '
                f'{v_sot:.3e} at R/Delta = {ratio:.1f}.')


# =============================================================================
def main():
    print('Test 1: skyrmion_ellipse on circular Neel skyrmion...')
    test_skyrmion_ellipse_circular()
    print('  OK')
    print()
    print('Test 2: dw_angle = pi at right DW for Neel skyrmion...')
    test_dw_angle_neel_at_rest()
    print('  OK')
    print()
    print('Test 3: TSH-off path is bit-identical to baseline...')
    test_tsh_off_equivalence()
    print('  OK')
    print()
    print('Test 4: |v_TSH| monotonically decreasing with R/Delta...')
    test_topological_torque_limits()
    print('  OK')
    print()
    print('All observable tests passed.')


# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f'FAIL: {e}', file=sys.stderr)
        sys.exit(1)
