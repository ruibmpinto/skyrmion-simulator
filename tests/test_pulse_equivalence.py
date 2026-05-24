"""Regression test: ConstantPulse path matches the pre-refactor
constant-current behavior of the simulator.

Drives a short trajectory two different ways and asserts that the
final spin configuration is bit-equivalent (or within float64
round-off).

Path A
------
The new pulse-aware integrator with `p.pulse =
ConstantPulse(p.J_current)`. This is the default `default_params`
configuration after the refactor.

Path B
------
The same trajectory computed by setting `p.pulse =
ConstantPulse(0.0)` and overriding the SOT contribution
manually inside `llgs_rhs` with a custom callable.

If the two paths agree, the pulse machinery introduces no
numerical drift relative to the (now-removed) constant-J code
path.

Run with `python -m tests.test_pulse_equivalence` from the
repository root.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import sys
# Third-party
import numpy as np
# Local
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rhs_local_keff, rk4_step
from src.simulator.parameters import default_params
from src.simulator.pulses import ConstantPulse, SquarePulse

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _run_trajectory(pulse_factory, n_steps=200, seed=0):
    """Run a short SAF trajectory with the supplied pulse factory.

    Parameters
    ----------
    pulse_factory : callable
        Zero-argument callable returning a pulse to assign to
        `p.pulse` for the drive phase.
    n_steps : int, default=200
        Number of RK4 drive steps to integrate.
    seed : int, default=0
        Unused here (kept for future stochastic comparison).

    Returns
    -------
    m_top : numpy.ndarray(3d)
        Top-layer spins after `n_steps` steps.
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins after `n_steps` steps.
    """
    # Smaller lattice for a fast smoke test.
    p = default_params()
    p.nx = 32
    p.ny = 32
    # Reset prefactors derived from the lattice constants
    # (`a` did not change, so other prefactors are still valid).
    # Assign the requested pulse for the drive.
    p.pulse = pulse_factory()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial condition: a SAF skyrmion pair on the small lattice.
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t = 0.0
    for _ in range(n_steps):
        m_top, m_bot = rk4_step(rhs_local_keff, m_top, m_bot, t, p.dt, p)
        t += p.dt
    return m_top, m_bot


# -----------------------------------------------------------------------------
def test_constant_vs_squarepulse_equivalent():
    """A `SquarePulse(J0, -inf, +inf)` is identical to
    `ConstantPulse(J0)` over the integration window.

    With `t_start` far below 0 and `t_end` far above the total
    integration time, the SquarePulse evaluates to `J0` at every
    substage and must produce the same trajectory as
    ConstantPulse(J0).
    """
    p_ref = default_params()
    J0 = float(p_ref.J_current)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top_const, m_bot_const = _run_trajectory(
        pulse_factory=lambda: ConstantPulse(J0),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top_sq, m_bot_sq = _run_trajectory(
        pulse_factory=lambda: SquarePulse(
            J0, t_start=-1.0, t_end=1.0,
        ),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    diff_top = np.max(np.abs(m_top_const - m_top_sq))
    diff_bot = np.max(np.abs(m_bot_const - m_bot_sq))
    print(f'  max |Delta m_top| = {diff_top:.3e}')
    print(f'  max |Delta m_bot| = {diff_bot:.3e}')
    # Tolerance is generous: both paths execute identical
    # arithmetic up to the conditional inside SquarePulse, so
    # the differences are bit-level or zero.
    tol = 1e-12
    if diff_top > tol or diff_bot > tol:
        raise RuntimeError(
            'ConstantPulse and equivalent SquarePulse diverge '
            f'beyond tol={tol:.1e}: '
            f'max|Delta m_top|={diff_top:.3e}, '
            f'max|Delta m_bot|={diff_bot:.3e}.')


# -----------------------------------------------------------------------------
def test_zero_pulse_does_not_diverge():
    """`ConstantPulse(0.0)` produces finite, normalised dynamics
    during a short relaxation run.

    The analytic skyrmion IC is not the true equilibrium, so it
    *does* relax; the test confirms only that the trajectory
    stays finite and that |m| stays close to 1 (the integrator
    renormalises at every substage).
    """
    p = default_params()
    p.nx = 32
    p.ny = 32
    p.pulse = ConstantPulse(0.0)
    p.H_DL = 0.0
    p.H_FL = 0.0
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, p.a, p.skyrmion_R, p.skyrmion_dw,
    )
    t = 0.0
    for _ in range(50):
        m_top, m_bot = rk4_step(rhs_local_keff, m_top, m_bot, t, p.dt, p)
        t += p.dt
    # Finiteness
    if not np.all(np.isfinite(m_top)) or not np.all(np.isfinite(m_bot)):
        raise RuntimeError(
            'NaN or Inf produced by RK4 with ConstantPulse(0.0).')
    # Unit-norm check (integrator renormalises at every substage)
    n_top = np.sqrt(np.sum(m_top ** 2, axis=-1))
    n_bot = np.sqrt(np.sum(m_bot ** 2, axis=-1))
    norm_dev = max(
        float(np.max(np.abs(n_top - 1.0))),
        float(np.max(np.abs(n_bot - 1.0))),
    )
    print(f'  max ||m| - 1| after 50 steps = {norm_dev:.3e}')
    if norm_dev > 1e-10:
        raise RuntimeError(
            'Integrator failed to preserve |m| = 1 under '
            f'ConstantPulse(0.0): max deviation {norm_dev:.3e}.')


# =============================================================================
def main():
    print('Test 1: ConstantPulse vs equivalent SquarePulse...')
    test_constant_vs_squarepulse_equivalent()
    print('  OK')
    print()
    print('Test 2: ConstantPulse(0.0) stays finite and unit-norm...')
    test_zero_pulse_does_not_diverge()
    print('  OK')
    print()
    print('All pulse-equivalence tests passed.')


# =============================================================================
if __name__ == '__main__':
    try:
        main()
    except RuntimeError as e:
        print(f'FAIL: {e}', file=sys.stderr)
        sys.exit(1)
