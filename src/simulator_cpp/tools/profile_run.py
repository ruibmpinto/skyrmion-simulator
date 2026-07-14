"""Micro-benchmark: time RK4 LLGS steps on a 64x64 lattice with the
Python simulator, for the local-K_eff and slab FFT demag field models.
Mirrors tools/profile_run.cpp so the two integrators are compared on
the identical workload. Integration only; no I/O, no observables.

Functions
---------
main
    Run both field-model benchmarks and print us/step.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
import sys
import time
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


def main():
    """Time Python RK4 steps for K_eff and slab demag at 64x64."""
    repo = pathlib.Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repo))
    from src.simulator.parameters import _precompute, default_params
    from src.simulator.pulses import ConstantPulse
    from src.simulator.initial_conditions import saf_skyrmion
    from src.simulator.integrator import (
        rhs_demag, rhs_local_keff, rk4_step)
    from src.simulator.demag import precompute_demag_kernels
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nx = 64
    ny = 64
    n_warmup = 50
    n_steps = 2000
    p = default_params()
    p.nx = nx
    p.ny = ny
    p.dt = 5.0e-14
    p.J_current = 4.0e11
    p.pulse = ConstantPulse(p.J_current)
    _precompute(p)
    m_top0, m_bot0 = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw)
    print(f'Python profile: {nx}x{ny} lattice, {n_steps} steps '
          f'(after {n_warmup} warmup)')

    def bench(label, rhs):
        m_top = m_top0.copy()
        m_bot = m_bot0.copy()
        t = 0.0
        for _ in range(n_warmup):
            m_top, m_bot = rk4_step(rhs, m_top, m_bot, t, p.dt, p)
            t += p.dt
        t0 = time.perf_counter()
        for _ in range(n_steps):
            m_top, m_bot = rk4_step(rhs, m_top, m_bot, t, p.dt, p)
            t += p.dt
        sec = time.perf_counter() - t0
        print(f'  {label:<6s} : {sec / n_steps * 1e6:8.2f} us/step  '
              f'({n_steps / sec:6.1f} steps/s, {sec:.3f} s total)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    bench('keff', rhs_local_keff)
    kernels = precompute_demag_kernels(
        p, kind='slab', accuracy=None, tol_conv=None)
    bench('slab', rhs_demag(kernels))


# =============================================================================
if __name__ == '__main__':
    main()
