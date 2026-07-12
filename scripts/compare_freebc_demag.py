"""Cross-validate the C++ free-BC (zero-padded) Newell demag vs Python.

Loads the dump from the C++ `dump_freebc_demag` tool (kernel spectra +
the seeded SAF field + the C++ demag output), recomputes the SAME with the
Python `newell_freebc` path on the EXACT C++ input field, and reports the
per-function differences:
  1. k-space kernel spectra (a few representative components),
  2. the demag field H_top/H_bot.

Run (after building/running dump_freebc_demag):
    python -m scripts.compare_freebc_demag
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
from types import SimpleNamespace
# Third-party
import numpy as np
# Local
from src.phase_diagram.params_helper import make_params
from src.simulator.demag import demag_field, precompute_demag_kernels
from src.simulator.fields import effective_field_demag_pair


def _rel(cpp, py):
    """Max-abs difference relative to the Python field scale."""
    cpp = np.asarray(cpp, dtype=float)
    py = np.asarray(py, dtype=float)
    scale = max(float(np.max(np.abs(py))), 1e-30)
    return float(np.max(np.abs(cpp - py))) / scale

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _max_abs(a, b):
    """Max absolute difference between two arrays."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def main():
    in_path = ('output/stochastic_llgs/validation/demag_freebc_cpp.npz')
    if not os.path.isfile(in_path):
        raise RuntimeError(
            f'compare_freebc_demag: {in_path!r} not found; build and run '
            f'src/simulator_cpp/build/dump_freebc_demag first.')
    z = np.load(in_path)
    nx = int(z['nx'])
    ny = int(z['ny'])
    a = float(z['a'])
    # Match the C++ tool's box; everything else from default params.
    p = make_params(nx=nx, ny=ny)
    p.a = a
    m_top = np.asarray(z['m_top'], dtype=float)
    m_bot = np.asarray(z['m_bot'], dtype=float)
    # Python free-BC and periodic fields on the EXACT C++ input.
    k_free = precompute_demag_kernels(
        p, kind='newell_freebc', accuracy=8.0, tol_conv=0.05)
    k_per = precompute_demag_kernels(
        p, kind='newell', accuracy=8.0, tol_conv=0.05)
    k_y = precompute_demag_kernels(
        p, kind='racetrack', accuracy=8.0, tol_conv=0.05)
    Htf_py, Hbf_py = demag_field(m_top, m_bot, k_free)
    Htp_py, Hbp_py = demag_field(m_top, m_bot, k_per)
    Hty_py, Hby_py = demag_field(m_top, m_bot, k_y)
    # Complete racetrack effective field (free-y demag + free-y
    # exchange/DMI) on the same input, kind='racetrack'.
    Hft_py, Hfb_py = effective_field_demag_pair(
        m_top, m_bot, p, k_y, mask=None)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # C++ vs Python, free-BC and periodic. Both sides now use the closed-
    # form Newell-Williams-Dunlop tensor (C++ DemagMethod::Closed default,
    # Python _METHOD='closed'), so the only residual is FFT round-off
    # (~1e-10). The periodic difference is the same-grid baseline.
    df_t = _rel(z['H_top'], Htf_py)
    df_b = _rel(z['H_bot'], Hbf_py)
    dp_t = _rel(z['H_top_per'], Htp_py)
    dp_b = _rel(z['H_bot_per'], Hbp_py)
    dy_t = _rel(z['H_top_y'], Hty_py)
    dy_b = _rel(z['H_bot_y'], Hby_py)
    # Full racetrack field: free-y demag + free-y exchange/DMI.
    fy_t = _rel(z['Hfull_top_y'], Hft_py)
    fy_b = _rel(z['Hfull_bot_y'], Hfb_py)
    print('=== demag field C++ vs Python (rel max-abs) ===')
    print(f'  periodic    : H_top={dp_t:.2e}  H_bot={dp_b:.2e}')
    print(f'  free-BC     : H_top={df_t:.2e}  H_bot={df_b:.2e}')
    print(f'  free-BC (y) : H_top={dy_t:.2e}  H_bot={dy_b:.2e}')
    print('=== full racetrack field (demag + exchange/DMI free-y) ===')
    print(f'  freebc_y all: H_top={fy_t:.2e}  H_bot={fy_b:.2e}')
    # Also: how much each boundary differs from periodic (the physics:
    # isolated vs image; nonzero at this tiny box).
    fp_t = _rel(z['H_top'], z['H_top_per'])
    yp_t = _rel(z['H_top_y'], z['H_top_per'])
    print(f'  freebc-vs-periodic   (C++, physics): H_top={fp_t:.2e}')
    print(f'  freebc_y-vs-periodic (C++, physics): H_top={yp_t:.2e}')
    # Pass: closed-form parity to FFT round-off (all three boundaries).
    tol = 1.0e-9
    ok = (df_t <= tol and df_b <= tol and dp_t <= tol and dp_b <= tol
          and dy_t <= tol and dy_b <= tol
          and fy_t <= tol and fy_b <= tol)
    print(f'FREEBC PORT {"PASS" if ok else "FAIL"}: closed-form C++/Python '
          f'agreement (freebc {max(df_t, df_b):.2e}, periodic '
          f'{max(dp_t, dp_b):.2e}, freebc_y {max(dy_t, dy_b):.2e}, '
          f'racetrack-full {max(fy_t, fy_b):.2e}) vs tol {tol:.1e}')
    sys.exit(0 if ok else 1)


# =============================================================================
if __name__ == '__main__':
    main()
