"""Measure peak RSS for a single phase-diagram task.

Runs a short relaxation at the lattice size set in the
User Configuration block below and prints peak RSS in MiB.
Use the reported value to size sbatch --mem-per-cpu.

Usage
-----
    PYTHONPATH=. python scripts/measure_ram.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import resource
import sys
import time
# Third-party
import numpy as np
# Local
from src.phase_diagram.classifier import classify
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.relaxation import relax
from src.simulator.demag import precompute_demag_kernels
from src.simulator.initial_conditions import saf_skyrmion

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _rss_mib():
    """Return current peak RSS in MiB.

    Returns
    -------
    rss : float
        Peak resident set size in MiB. macOS reports
        `ru_maxrss` in bytes, Linux in kilobytes.
    """
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform == 'darwin':
        return r / (1024.0 * 1024.0)
    return r / 1024.0


def main():
    """Run one relaxation and report peak RSS + wall time."""
    # ================ User Configuration ================
    nx = 128
    ny = 128
    max_steps = 2000
    alpha_relax = 1.0
    tol_torque = 1e-4
    tol_dE = 1e-7
    # ============ End User Configuration =================
    rss_before = _rss_mib()
    t0 = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # One representative phase-diagram task: build params and
    # kernels, seed a SAF skyrmion, relax, then classify.
    p = make_params(
        nx=nx, ny=ny,
        D=1.0e-3,
        H_ext=np.array([0.0, 0.0, 0.0]),
        J_current=0.0,
    )
    kernels = precompute_demag_kernels(
        p, kind='slab', accuracy=None, tol_conv=None)
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, a=p.a, R=p.skyrmion_R, dw=p.skyrmion_dw,
    )
    m_top, m_bot, converged, n_steps, E, tau = relax(
        m_top, m_bot, p, kernels,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
    )
    label, _ = classify(m_top, m_bot, p)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Report peak RSS and wall time to size sbatch memory.
    rss_after = _rss_mib()
    dt = time.time() - t0
    print(
        f'nx={nx} ny={ny} max_steps={max_steps}\n'
        f'  rss_before={rss_before:.1f} MiB\n'
        f'  rss_peak  ={rss_after:.1f} MiB\n'
        f'  delta     ={rss_after - rss_before:.1f} MiB\n'
        f'  wall      ={dt:.2f} s\n'
        f'  n_steps   ={n_steps}  converged={converged}\n'
        f'  label     ={label}'
    )


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
