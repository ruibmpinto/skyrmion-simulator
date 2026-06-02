"""Plotter for Pham et al. (2024) Figure S46 (both panels).

S46_A_vD_HRKKY.png
    v_avg and max(d_top) vs H_RKKY at fixed J = 8.9e11,
    FWHM = 500 ps.
S46_B_vD_FWHM.png
    v_avg and max(d_top) vs FWHM at fixed J = 8.9e11,
    H_RKKY = 205 mT.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.simulator.parameters import default_params
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


def _vavg(trace, metadata):
    """Net displacement over +/- 3 sigma window / FWHM with
    the wrapped centroids unwrapped first."""
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    cx, cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
    )
    FWHM = float(metadata['FWHM'])
    tail = (float(metadata['tail_sigmas'])
            * float(metadata['sigma']))
    t_lo = float(metadata['t_center']) - tail
    t_hi = float(metadata['t_center']) + tail
    i_lo = int(np.argmin(np.abs(t - t_lo)))
    i_hi = int(np.argmin(np.abs(t - t_hi)))
    dx = cx[i_hi] - cx[i_lo]
    dy = cy[i_hi] - cy[i_lo]
    return float(np.sqrt(dx * dx + dy * dy) / FWHM)


def _load_sweep(in_dir, key):
    """Load every NPZ in `in_dir`, sorted by metadata[key]."""
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    items = []
    for path in paths:
        trace, metadata = load_trace(path)
        items.append((metadata, trace))
    items.sort(key=lambda mt: float(mt[0][key]))
    return items


def _plot_panel(items, x_key, x_label, x_scale,
                title, out_path):
    """Generic two-axis plotter: v_avg on left, d_max on right."""
    # X axis values (with caller-supplied unit scaling).
    x_arr = np.array([float(m[x_key]) for m, _ in items]) * x_scale
    # v_avg per trace (m/s).
    v_arr = np.array([_vavg(t, m) for m, t in items])
    # Peak top-layer diameter per trace (m).
    d_arr = np.array(
        [float(np.max(t['d_top'])) for _, t in items])
    # Twin axes: velocity on the left, diameter on the right.
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # Velocity points + line.
    ax.plot(x_arr, v_arr, 'o-', color='C0',
            label=r'$v_{\mathrm{avg}}$')
    # Diameter points + line in nm.
    ax2.plot(x_arr, d_arr * 1e9, '^-', color='C3',
             label=r'max $d_{\mathrm{top}}$')
    # Axis decoration; left/right axes colour-coded.
    ax.set_xlabel(x_label)
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)', color='C0')
    ax2.set_ylabel(r'max diameter (nm)', color='C3')
    ax.tick_params(axis='y', labelcolor='C0')
    ax2.tick_params(axis='y', labelcolor='C3')
    # Merge handles from both axes into one legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    # Console summary, one line per swept point.
    for m, t in items:
        v = _vavg(t, m)
        d = float(np.max(t['d_top']))
        print(f'  {x_key} = {float(m[x_key]) * x_scale:.2f}: '
              f'v_avg = {v:>5.0f} m/s, '
              f'max d_top = {d*1e9:.0f} nm')


def main():
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    # Panel A: H_RKKY sweep.
    in_dir_a = 'output/sweeps_S41_S49/S46a'
    if os.path.isdir(in_dir_a):
        items_a = _load_sweep(in_dir_a, key='H_RKKY')
        _plot_panel(
            items=items_a,
            x_key='H_RKKY',
            x_label=r'$H_{\mathrm{RKKY}}$ (mT)',
            x_scale=1.0e3,
            title='S46(a): velocity vs RKKY field',
            out_path=os.path.join(
                out_dir, 'S46_A_vD_HRKKY.png'))
    else:
        print(f'Skipping panel (a): {in_dir_a} not found.')
    # Panel B: FWHM sweep.
    in_dir_b = 'output/sweeps_S41_S49/S46b'
    if os.path.isdir(in_dir_b):
        items_b = _load_sweep(in_dir_b, key='FWHM')
        _plot_panel(
            items=items_b,
            x_key='FWHM',
            x_label=r'FWHM (ps)',
            x_scale=1.0e12,
            title='S46(b): velocity vs pulse width',
            out_path=os.path.join(
                out_dir, 'S46_B_vD_FWHM.png'))
    else:
        print(f'Skipping panel (b): {in_dir_b} not found.')


# =============================================================================
if __name__ == '__main__':
    main()
