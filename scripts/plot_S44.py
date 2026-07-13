"""Plotter for Pham et al. (2024) Figure S44.

Three panels, all read from `output/sweeps_S41_S49/S44/`:

S44_A_d_t.png
    Skyrmion diameter (top layer) vs time during the 500 ps
    pulse at J = 8.9e11 A/m^2. The pulse profile is drawn on a
    secondary y-axis for context.
S44_B_vD_J.png
    v_avg, max(D1_top), min(D2_top) vs current density J.
S44_C_psi_J.png
    DW magnetization angle psi at the time of maximum diameter
    for top and bottom layers, vs J.
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


def _vavg(trace, metadata):
    """Net (delta x, delta y) over the +/- 3 sigma window /
    FWHM, with the wrapped centroids unwrapped first."""
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


def _load_panel_A(in_dir):
    """Load panel A trace. Accepts both legacy `panelA.npz`
    and the new D-tagged `panelA_D{X}e-3.npz`; picks the last
    match alphabetically (D-tagged sorts after the bare name)."""
    paths = sorted(glob.glob(os.path.join(in_dir, 'panelA*.npz')))
    if not paths:
        raise RuntimeError(
            f'plot_S44: no panelA*.npz trace found in {in_dir}.')
    return load_trace(paths[-1])


def _load_panel_BC(in_dir):
    """Load every `J_*.npz`, sorted by J0. Raises if empty."""
    paths = sorted(glob.glob(os.path.join(in_dir, 'J_*.npz')))
    if not paths:
        raise RuntimeError(
            f'plot_S44: no J_*.npz traces found in {in_dir}.')
    out = []
    for path in paths:
        trace, metadata = load_trace(path)
        out.append((metadata, trace))
    out.sort(key=lambda mt: float(mt[0]['J0']))
    return out


def _plot_A(trace, metadata, out_path):
    """Diameter vs time + pulse envelope on right axis."""
    # Time axis (s) and top-layer diameter (m) per sample.
    t = trace['t']
    d_top = trace['d_top']
    # Reconstruct the Gaussian pulse from its metadata.
    sigma = float(metadata['sigma'])
    t_center = float(metadata['t_center'])
    J0 = float(metadata['J0'])
    z = (t - t_center) / sigma
    J_t = J0 * np.exp(-0.5 * z * z)
    # Twin-axis figure: diameter (left) + J(t) (right).
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # Diameter trace, plotted in nm vs ps.
    ax.plot(t * 1e12, d_top * 1e9, color='C0',
            label='$d_{\\mathrm{top}}$')
    # Drive pulse profile in units of 10^11 A/m^2.
    ax2.plot(t * 1e12, J_t / 1e11, color='violet',
             linestyle='--',
             label='$J / 10^{11}$')
    # Axis decoration; pulse axis labelled in violet.
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'diameter (nm)')
    ax2.set_ylabel(r'$J$ ($10^{11}$ A/m$^2$)', color='violet')
    ax2.tick_params(axis='y', labelcolor='violet')
    fwhm_ps = float(metadata['FWHM']) * 1e12
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_B(items, out_path):
    """v_avg, max(D1_top), min(D2_top) vs J."""
    # Reduce each (metadata, trace) into one scalar per quantity.
    J_arr = np.array([float(m['J0']) for m, _ in items])
    v_arr = np.array([_vavg(t, m) for m, t in items])
    # Peak major-axis size (D1) and trough minor-axis size (D2).
    D1_max = np.array(
        [float(np.max(t['D1_top'])) for _, t in items])
    D2_min = np.array(
        [float(np.min(t['D2_top'])) for _, t in items])
    # Twin axes: velocity (left) + axis sizes (right).
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    # Velocity in m/s vs J in 10^11 A/m^2.
    ax.plot(J_arr / 1e11, v_arr, 'o-', color='C0',
            label=r'$v_{\mathrm{avg}}$')
    # Major and minor axes in nm on the right axis.
    ax2.plot(J_arr / 1e11, D1_max * 1e9, '^-', color='C3',
             label=r'max $D_1$ (nm)')
    ax2.plot(J_arr / 1e11, D2_min * 1e9, 's-', color='C2',
             label=r'min $D_2$ (nm)')
    # Axis decoration; left axis colour-tied to the velocity.
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)', color='C0')
    ax2.set_ylabel(r'axis size (nm)')
    ax.tick_params(axis='y', labelcolor='C0')
    # Combined legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _unwrap_to_zero_rest(psi_seq):
    """Unwrap a psi(J) sequence stored under the OLD dw_angle
    convention (atan2(my, -mx)) into a signed deviation from the
    layer's natural Neel orientation, matching the new convention.

    Strategy: np.unwrap removes spurious +-pi jumps, then we
    subtract the small-J rest value modulo pi so the series
    starts at 0 in either layer."""
    psi = np.unwrap(np.asarray(psi_seq))
    # Rest value is the smallest-J point; reduce mod pi so we
    # measure deviation from the layer's natural +-x_hat branch.
    rest = ((psi[0] + np.pi / 2.0) % np.pi) - np.pi / 2.0
    return psi - (psi[0] - rest)


def _plot_C(items, out_path):
    """psi at maximum-diameter time, top vs bottom layers."""
    # Per-trace J0 (drive level).
    J_arr = np.array([float(m['J0']) for m, _ in items])
    # Sample psi at the time of peak top-layer diameter.
    psi_top_raw = []
    psi_bot_raw = []
    for m, t in items:
        i_peak = int(np.argmax(t['d_top']))
        psi_top_raw.append(float(t['psi_top'][i_peak]))
        psi_bot_raw.append(float(t['psi_bot'][i_peak]))
    # Sort by J before unwrapping so np.unwrap follows monotone J.
    order = np.argsort(J_arr)
    J_arr = J_arr[order]
    # Convention conversion + radians -> degrees.
    psi_top = np.degrees(_unwrap_to_zero_rest(
        np.array(psi_top_raw)[order]))
    psi_bot = np.degrees(_unwrap_to_zero_rest(
        np.array(psi_bot_raw)[order]))
    # Top and bottom DW angles on a shared axis.
    fig, ax = plt.subplots()
    ax.plot(J_arr / 1e11, psi_top, 's-', color='C3',
            label=r'top, $\psi$')
    ax.plot(J_arr / 1e11, psi_bot, 'o-', color='k',
            label=r'bot, $\psi$')
    # Zero reference line (natural Neel orientation).
    ax.axhline(0.0, color='0.7', linestyle=':')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$\psi$ (deg, signed deviation)')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


# Render all three S44 panels from the shared S44 sweep.
def main():
    in_dir = 'output/sweeps_S41_S49/S44'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    # Panel A.
    trace_A, meta_A = _load_panel_A(in_dir)
    _plot_A(trace_A, meta_A,
            out_path=os.path.join(out_dir, 'S44_A_d_t.png'))
    # Panels B and C share the same J sweep.
    items_BC = _load_panel_BC(in_dir)
    _plot_B(items_BC,
            out_path=os.path.join(out_dir, 'S44_B_vD_J.png'))
    _plot_C(items_BC,
            out_path=os.path.join(out_dir, 'S44_C_psi_J.png'))


# =============================================================================
if __name__ == '__main__':
    main()
