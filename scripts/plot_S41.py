"""Plotter for Pham et al. (2024) Figure S41.

Reads the trace produced by `scripts/sweep_S41_v_time.py`,
computes the instantaneous and average skyrmion velocity, and
emits one PNG to `output/figures_S41_S49/S41_v_t.png`.

Logic
-----
- Load the NPZ trace from `output/sweeps_S41_S49/S41/run.npz`.
- The skyrmion centre is sampled every few ps; the
  instantaneous velocity follows from centred finite
  differences on `(cx_top, cy_top)` after masking out the
  relaxation tail (steps outside the pulse window).
- The average velocity is `(x(t_end_pulse) - x(0)) / t_pulse`
  using the centroid samples closest to the pulse edges.
- One panel: |v(t)| in m/s vs time in ps with a dashed
  horizontal line at v_avg.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.sweeps.io import load_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults.
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # I/O configuration (edit here). The sweep encodes the
    # demag kind and DMI value in the filename; keep this
    # path in sync with `sweep_S41_v_time.py:out_path`.
    demag_kind = 'newell'
    D = 0.85e-3
    _D_tag = f'D{int(round(D*1e5)):03d}e-3'
    in_path = (
        f'output/sweeps_S41_S49/S41/'
        f'run_{demag_kind}_{_D_tag}.npz')
    out_dir = 'output/figures_S41_S49'
    out_path = os.path.join(
        out_dir,
        f'S41_v_t_{demag_kind}_{_D_tag}.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the trace + metadata.
    trace, metadata = load_trace(in_path)
    # Time stamps in seconds and ps.
    t = trace['t']
    t_ps = t * 1e12
    # Centroid coordinates (m) for the top layer; bottom layer
    # tracks the same in-plane motion by SAF symmetry. The
    # centroid is PBC-wrapped (modulo L = nx * a), so we
    # unwrap the stream before finite-differencing.
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    # Lattice constant from default_params (every sweep uses
    # the same `p.a = 2 nm`).
    from src.simulator.parameters import default_params
    a_default = float(default_params().a)
    L_x = nx * a_default
    L_y = ny * a_default
    cx_w = trace['cx_top']
    cy_w = trace['cy_top']
    cx, cy = unwrap_trajectory(cx_w, cy_w, L_x=L_x, L_y=L_y)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Instantaneous velocity from centred finite differences on
    # the unwrapped centroids.
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    v_mag = np.sqrt(vx * vx + vy * vy)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Average velocity over the pulse window.
    t_pulse = float(metadata['t_pulse'])
    # Index of the sample closest to the pulse start (t = 0).
    i_start = int(np.argmin(np.abs(t - 0.0)))
    # Index of the sample closest to the pulse end.
    i_end = int(np.argmin(np.abs(t - t_pulse)))
    # Net displacement during the pulse window.
    dx = cx[i_end] - cx[i_start]
    dy = cy[i_end] - cy[i_start]
    v_avg = float(np.sqrt(dx * dx + dy * dy) / t_pulse)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure (single panel, square plotting area).
    fig, ax = plt.subplots()
    # Instantaneous-velocity trace, solid line.
    ax.plot(t_ps, v_mag, label=r'$|v_{\mathrm{inst}}(t)|$')
    # Average-velocity reference, dashed horizontal.
    ax.axhline(
        v_avg,
        color='k', linestyle='--',
        label=fr'$v_{{\mathrm{{avg}}}} = {v_avg:.1f}$ m/s')
    # Pulse window shading for clarity.
    ax.axvspan(0.0, t_pulse * 1e12, color='0.92', zorder=0,
               label=f'pulse ({t_pulse*1e9:.1f} ns)')
    # Labels and title.
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.set_title(
        f'S41: $J = {metadata["J0"]:.0e}$ A/m$^2$')
    # Tight axis bounds; small y-padding for the v_avg label.
    ax.set_xlim(t_ps[0], t_ps[-1])
    ymax = float(np.max(v_mag)) * 1.1
    ax.set_ylim(0.0, max(ymax, v_avg * 1.4))
    ax.legend(loc='upper right', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist.
    os.makedirs(out_dir, exist_ok=True)
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    print(f'  v_avg = {v_avg:.1f} m/s (pulse {t_pulse*1e9:.1f} ns)')
    print(f'  max |v_inst| = {float(np.max(v_mag)):.1f} m/s')


# =============================================================================
if __name__ == '__main__':
    main()
