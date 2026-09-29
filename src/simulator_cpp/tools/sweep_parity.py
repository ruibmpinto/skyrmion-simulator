"""Compare the C++ sweep trace (/tmp/cpp_sweep_parity.npz) against the
Python run_one driver at the identical small-lattice K_eff config.

Functions
---------
main
    Run the Python driver and diff every observable array.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import pathlib
import sys
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def main():
    """Run Python run_one and compare to the C++ trace."""
    cpp_npz = '/tmp/cpp_sweep_parity.npz'
    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from src.simulator.parameters import _precompute, default_params
    from src.simulator.pulses import SquarePulse
    from src.simulator.initial_conditions import saf_skyrmion
    from src.orchestrator.driver import run_one
    from src.orchestrator.integrators import step_deterministic
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Identical config to tools/sweep_parity.cpp.
    p = default_params()
    p.nx = 64
    p.ny = 64
    p.a = 2.0e-9
    p.dt = 5.0e-14
    p.D = 0.85e-3
    p.skyrmion_R = 25.0e-9
    p.skyrmion_dw = 10.0e-9
    _precompute(p)
    J0 = 1.0e11
    t_pulse = 1.0e-10
    n_relax = int(math.ceil(5.0e-12 / p.dt))
    n_drive = int(math.ceil(1.5e-10 / p.dt))
    sample_every = int(math.ceil(5.0e-12 / p.dt))
    pulse = SquarePulse(J0=J0, t_start=0.0, t_end=t_pulse)
    step = step_deterministic()

    def ic_factory(p):
        return saf_skyrmion(
            p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)

    trace_py = run_one(
        p=p, pulse=pulse, n_relax=n_relax, n_drive=n_drive,
        sample_every=sample_every, step_drive=step, step_relax=step,
        ic_factory=ic_factory, record_snapshot_at=None, print_every=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    cpp = np.load(cpp_npz)
    keys = ['t', 'cx_top', 'cy_top', 'cx_bot', 'cy_bot',
            'd_top', 'd_bot', 'D1_top', 'D2_top', 'theta_top',
            'D1_bot', 'D2_bot', 'theta_bot', 'psi_top', 'psi_bot',
            'Q_top', 'Q_bot']
    print(f'{"key":<12s} {"n":>4s} {"max|diff|":>12s} '
          f'{"max|py|":>12s} {"status":>6s}')
    max_rel = 0.0
    for k in keys:
        a = np.asarray(cpp[k], dtype=float)
        b = np.asarray(trace_py[k], dtype=float)
        if a.shape != b.shape:
            print(f'{k:<12s} SHAPE MISMATCH {a.shape} vs {b.shape}')
            max_rel = float('inf')
            continue
        diff = np.abs(a - b)
        denom = max(np.max(np.abs(b)), 1e-30)
        rel = float(diff.max() / denom)
        status = 'PASS' if (diff.max() <= 1e-12 + 1e-10 * denom) else 'FAIL'
        if rel > max_rel and status == 'FAIL':
            max_rel = rel
        print(f'{k:<12s} {a.size:>4d} {diff.max():>12.3e} '
              f'{np.max(np.abs(b)):>12.3e} {status:>6s}')
    print()
    if max_rel == 0.0:
        print('sweep_parity: all observable arrays match (atol+rtol).')
    else:
        print(f'sweep_parity: MISMATCH, worst rel={max_rel:.3e}')


# =============================================================================
if __name__ == '__main__':
    main()
