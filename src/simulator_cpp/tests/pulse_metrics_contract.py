"""Checks for the pulse charge and action metrics.

Verifies, for every primitive shape, that the Simpson quadrature of
int J dt and int J^2 dt reproduces the closed form, and that the
triangular pulse delivers the same charge and the same action wherever
its peak sits -- the property that lets the three asymmetries be
compared with charge and dissipated energy held fixed.

Run with:
    python src/simulator_cpp/tests/pulse_metrics_contract.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
import sys
# Third-party
import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
sys.path.insert(0, str(_REPO))

# Local (after path setup)
from skyrmion_simulator.simulator.pulses import (
    ConstantPulse, GaussianPulse, HalfSinePulse, SquarePulse,
    TrianglePulse,
)
from skyrmion_simulator.simulator.pulse_metrics import (
    analytic_action, analytic_charge, pulse_action, pulse_charge,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _check(name, ok, detail, results):
    """Record one check.

    Parameters
    ----------
    name : str
        Check label.
    ok : bool
        Whether the check passed.
    detail : str
        Extra context printed alongside the verdict.
    results : list
        Accumulator of (name, ok) pairs.
    """
    results.append((name, bool(ok)))
    print(f'  {name:52s} {"PASS" if ok else "FAIL"}  {detail}')


# -----------------------------------------------------------------------------
def _expect_raise(name, thunk, results):
    """Record whether a call raises RuntimeError as required.

    Parameters
    ----------
    name : str
        Check label.
    thunk : callable
        Zero-argument callable expected to raise.
    results : list
        Accumulator of (name, ok) pairs.
    """
    try:
        thunk()
    except RuntimeError:
        _check(name, True, 'raised', results)
        return
    _check(name, False, 'did NOT raise', results)


# -----------------------------------------------------------------------------
def main():
    """Run the pulse-metric checks and report."""
    # =========================== User Configuration =========================
    j0 = 4.0e11               # A/m^2, peak amplitude
    t_start = 0.0             # s, window start
    t_end = 1.0e-9            # s, window end
    n_sample = 20001          # quadrature samples
    rel_tol = 1.0e-9          # allowed |numeric/analytic - 1|
    # ======================= End User Configuration =========================
    t_mid = 0.5*(t_start + t_end)
    shapes = {
        'square': SquarePulse(j0, t_start, t_end),
        'triangle_sharprise': TrianglePulse(j0, t_start, t_start, t_end),
        'triangle_sharpfall': TrianglePulse(j0, t_start, t_end, t_end),
        'triangle_symmetric': TrianglePulse(j0, t_start, t_mid, t_end),
        'half_sine': HalfSinePulse(j0, t_start, t_end),
    }
    results = []
    print('quadrature vs closed form:')
    for name, pulse in shapes.items():
        q_num = pulse_charge(pulse, t_start, t_end, n_sample)
        q_ana = analytic_charge(pulse)
        s_num = pulse_action(pulse, t_start, t_end, n_sample)
        s_ana = analytic_action(pulse)
        _check(f'{name}_charge',
               abs(q_num/q_ana - 1.0) < rel_tol,
               f'num/ana = {q_num/q_ana:.12f}', results)
        _check(f'{name}_action',
               abs(s_num/s_ana - 1.0) < rel_tol,
               f'num/ana = {s_num/s_ana:.12f}', results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # The Gaussian's closed forms are over the whole real line, so the
    # window must be wide enough for the tails to be negligible.
    gauss = GaussianPulse(j0, t_mid, 2.0e-10)
    lo = t_mid - 12.0*gauss.sigma
    hi = t_mid + 12.0*gauss.sigma
    q_num = pulse_charge(gauss, lo, hi, n_sample)
    s_num = pulse_action(gauss, lo, hi, n_sample)
    _check('gaussian_charge',
           abs(q_num/analytic_charge(gauss) - 1.0) < rel_tol,
           f'num/ana = {q_num/analytic_charge(gauss):.12f}', results)
    _check('gaussian_action',
           abs(s_num/analytic_action(gauss) - 1.0) < rel_tol,
           f'num/ana = {s_num/analytic_action(gauss):.12f}', results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Triangle invariance: charge and action must not depend on t_peak,
    # which is what makes the three asymmetries directly comparable.
    print('triangle charge/action independent of peak placement:')
    peaks = np.linspace(t_start, t_end, 9)
    charges = [analytic_charge(TrianglePulse(j0, t_start, tp, t_end))
               for tp in peaks]
    actions = [analytic_action(TrianglePulse(j0, t_start, tp, t_end))
               for tp in peaks]
    num_charges = [
        pulse_charge(TrianglePulse(j0, t_start, tp, t_end),
                     t_start, t_end, n_sample) for tp in peaks]
    _check('triangle_charge_peak_invariant_analytic',
           np.ptp(charges) == 0.0,
           f'spread = {np.ptp(charges):.3e}', results)
    _check('triangle_action_peak_invariant_analytic',
           np.ptp(actions) == 0.0,
           f'spread = {np.ptp(actions):.3e}', results)
    _check('triangle_charge_peak_invariant_quadrature',
           np.ptp(num_charges)/np.mean(num_charges) < 1.0e-9,
           f'rel spread = '
           f'{np.ptp(num_charges)/np.mean(num_charges):.3e}', results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shapes at equal peak and duration rank as square > half-sine >
    # triangle in delivered charge; a regression here means a shape was
    # mis-implemented.
    ordered = (analytic_charge(shapes['square'])
               > analytic_charge(shapes['half_sine'])
               > analytic_charge(shapes['triangle_symmetric']))
    _check('charge_ordering_square_gt_sine_gt_triangle', ordered,
           f"{analytic_charge(shapes['square']):.3e} > "
           f"{analytic_charge(shapes['half_sine']):.3e} > "
           f"{analytic_charge(shapes['triangle_symmetric']):.3e}",
           results)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Error contracts.
    print('error contracts:')
    _expect_raise('constant_charge_needs_window',
                  lambda: analytic_charge(ConstantPulse(j0)), results)
    _expect_raise('constant_action_needs_window',
                  lambda: analytic_action(ConstantPulse(j0)), results)
    _expect_raise('charge_inverted_window',
                  lambda: pulse_charge(shapes['square'], t_end,
                                       t_start, n_sample), results)
    _expect_raise('charge_too_few_samples',
                  lambda: pulse_charge(shapes['square'], t_start,
                                       t_end, 2), results)
    _expect_raise('charge_even_sample_count',
                  lambda: pulse_charge(shapes['square'], t_start,
                                       t_end, 2000), results)
    n_pass = sum(1 for _n, ok in results if ok)
    print(f'pulse_metrics_contract: {n_pass}/{len(results)} checks '
          f'passed.')
    if n_pass != len(results):
        raise RuntimeError(
            f'pulse_metrics_contract: {len(results) - n_pass} '
            f'check(s) failed.')


# =============================================================================
if __name__ == '__main__':
    main()
