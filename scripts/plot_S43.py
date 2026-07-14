"""Plotter for Pham et al. (2024) Figure S43 (both panels).

Reads `output/sweeps_S41_S49/S43a/` and `S43b/`, computes
v_avg for every trace, and emits two PNGs to
`output/figures_S41_S49/`:

S43a_v_FWHM.png
    v_avg vs FWHM, one line per fixed J.
S43b_v_J.png
    v_avg vs J, one line per fixed D.

v_avg is taken as the net x-y displacement between the two
sampled times nearest to (t_center +/- tail_sigmas * sigma),
divided by FWHM (matching the paper's Delta x / FWHM
definition).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
from collections import defaultdict
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.simulator.parameters import default_params
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.orchestrator.io import load_trace

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


def _vavg_from_trace(trace, metadata):
    """Net (delta x, delta y) over the +/- 3 sigma window, divided
    by FWHM; matches the paper's Delta x / FWHM definition.

    Unwraps the PBC-wrapped centroids before differencing so a
    skyrmion that crosses the box boundary still yields the
    physical displacement.
    """
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    cx, cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
    periodic_y=True)
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
    """Load NPZs and bucket them by metadata[key] -> list of
    (metadata, trace)."""
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    bucket = defaultdict(list)
    for path in paths:
        trace, metadata = load_trace(path)
        bucket[float(metadata[key])].append((metadata, trace))
    return bucket


def _plot_panel_a(in_dir, out_path):
    """v_avg vs FWHM, one line per J."""
    # Bucket by J0; within each bucket sort by FWHM.
    bucket = _load_sweep(in_dir, key='J0')
    # New figure with a secondary-axis-free layout.
    fig, ax = plt.subplots()
    # Sequential colormap for the J levels.
    cmap = plt.get_cmap('viridis')
    # Sweep over each fixed J value in ascending order.
    J_keys = sorted(bucket.keys())
    for J0 in J_keys:
        # Sort the FWHM samples within this J bucket.
        items = sorted(bucket[J0],
                       key=lambda mt: float(mt[0]['FWHM']))
        # Stack the FWHM (s) and v_avg (m/s) arrays for the plot.
        FWHM_arr = np.array(
            [float(m['FWHM']) for m, _ in items])
        v_arr = np.array(
            [_vavg_from_trace(t, m) for m, t in items])
        # Colour by J relative to its position in the J sweep.
        c = cmap(J_keys.index(J0) / max(len(J_keys) - 1, 1))
        # FWHM is converted to ps for the x-axis.
        ax.plot(FWHM_arr * 1e12, v_arr, 'o-', color=c,
                label=fr'$J = {J0:.1e}$ A/m$^2$')
    # Axis decoration.
    ax.set_xlabel(r'FWHM (ps)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    # Persist.
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    # Console summary, one block per J.
    for J0 in J_keys:
        items = sorted(bucket[J0],
                       key=lambda mt: float(mt[0]['FWHM']))
        print(f'  J = {J0:.2e}:')
        for m, t in items:
            v = _vavg_from_trace(t, m)
            print(f'    FWHM = {float(m["FWHM"])*1e12:>5.0f} ps -> '
                  f'v_avg = {v:>6.1f} m/s')


def _plot_panel_b(in_dir, out_path):
    """v_avg vs J, one line per D."""
    # Bucket by DMI strength; each D bucket holds the J sweep.
    bucket = _load_sweep(in_dir, key='D')
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    # Drop D = 0.60 mJ/m^2: sweep is incomplete (J >= 6e11
    # traces missing on disk).
    D_keys = sorted(d for d in bucket.keys()
                    if not np.isclose(d, 0.60e-3))
    for D in D_keys:
        # Sort the J samples within this D bucket.
        items = sorted(bucket[D],
                       key=lambda mt: float(mt[0]['J0']))
        # Stack the J (A/m^2) and v_avg (m/s) arrays.
        J_arr = np.array(
            [float(m['J0']) for m, _ in items])
        v_arr = np.array(
            [_vavg_from_trace(t, m) for m, t in items])
        # Colour by D's position in the plotted list.
        c = cmap(D_keys.index(D) / max(len(D_keys) - 1, 1))
        # Plot J in units of 10^11 A/m^2.
        ax.plot(J_arr / 1e11, v_arr, 'o-', color=c,
                label=fr'$D = {D*1e3:.2f}$ mJ/m$^2$')
    # Axis decoration.
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    # Console summary, one block per D (including dropped D).
    for D in sorted(bucket.keys()):
        items = sorted(bucket[D],
                       key=lambda mt: float(mt[0]['J0']))
        print(f'  D = {D*1e3:.2f} mJ/m^2:')
        for m, t in items:
            v = _vavg_from_trace(t, m)
            print(f'    J = {float(m["J0"]):.2e} -> '
                  f'v_avg = {v:>6.1f} m/s')


# Render whichever of the two S43 sweeps exist on disk.
def main():
    """Render panel (a) v_avg vs FWHM and panel (b) v_avg vs J from
    the S43a and S43b sweeps, skipping whichever sweep is absent."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # I/O configuration.
    in_dir_a = 'output/sweeps_S41_S49/S43a'
    in_dir_b = 'output/sweeps_S41_S49/S43b'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel (a): only render if the sweep exists.
    if os.path.isdir(in_dir_a):
        _plot_panel_a(
            in_dir=in_dir_a,
            out_path=os.path.join(out_dir, 'S43a_v_FWHM.png'))
    else:
        print(f'Skipping panel (a): {in_dir_a} not found.')
    # Panel (b): same.
    if os.path.isdir(in_dir_b):
        _plot_panel_b(
            in_dir=in_dir_b,
            out_path=os.path.join(out_dir, 'S43b_v_J.png'))
    else:
        print(f'Skipping panel (b): {in_dir_b} not found.')


# =============================================================================
if __name__ == '__main__':
    main()
