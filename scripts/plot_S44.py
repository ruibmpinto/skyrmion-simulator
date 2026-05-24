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
    """Net (delta x, delta y) over the +/- 3 sigma window /
    FWHM, with the wrapped centroids unwrapped first."""
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


def _load_panel_A(in_dir):
    """Load `panelA.npz`. Raises if missing."""
    path = os.path.join(in_dir, 'panelA.npz')
    if not os.path.exists(path):
        raise RuntimeError(
            f'plot_S44: panel A trace not found at {path}.')
    return load_trace(path)


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
    t = trace['t']
    d_top = trace['d_top']
    sigma = float(metadata['sigma'])
    t_center = float(metadata['t_center'])
    J0 = float(metadata['J0'])
    z = (t - t_center) / sigma
    J_t = J0 * np.exp(-0.5 * z * z)
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(t * 1e12, d_top * 1e9, color='C0',
            label='$d_{\\mathrm{top}}$')
    ax2.plot(t * 1e12, J_t / 1e11, color='violet',
             linestyle='--',
             label='$J / 10^{11}$')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'diameter (nm)')
    ax2.set_ylabel(r'$J$ ($10^{11}$ A/m$^2$)', color='violet')
    ax2.tick_params(axis='y', labelcolor='violet')
    fwhm_ps = float(metadata['FWHM']) * 1e12
    ax.set_title(
        f'S44(a): d(t) at J={J0:.1e}, FWHM={fwhm_ps:.0f} ps')
    ax.set_box_aspect(1)
    ax.legend(loc='upper left', frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_B(items, out_path):
    """v_avg, max(D1_top), min(D2_top) vs J."""
    J_arr = np.array([float(m['J0']) for m, _ in items])
    v_arr = np.array([_vavg(t, m) for m, t in items])
    D1_max = np.array(
        [float(np.max(t['D1_top'])) for _, t in items])
    D2_min = np.array(
        [float(np.min(t['D2_top'])) for _, t in items])
    fig, ax = plt.subplots()
    ax2 = ax.twinx()
    ax.plot(J_arr / 1e11, v_arr, 'o-', color='C0',
            label=r'$v_{\mathrm{avg}}$')
    ax2.plot(J_arr / 1e11, D1_max * 1e9, '^-', color='C3',
             label=r'max $D_1$ (nm)')
    ax2.plot(J_arr / 1e11, D2_min * 1e9, 's-', color='C2',
             label=r'min $D_2$ (nm)')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)', color='C0')
    ax2.set_ylabel(r'axis size (nm)')
    ax.tick_params(axis='y', labelcolor='C0')
    ax.set_title('S44(b): velocity and ellipse axes vs J')
    # Combined legend.
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2,
              loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_C(items, out_path):
    """psi at maximum-diameter time, top vs bottom layers."""
    J_arr = np.array([float(m['J0']) for m, _ in items])
    psi_top = []
    psi_bot = []
    for m, t in items:
        i_peak = int(np.argmax(t['d_top']))
        psi_top.append(float(np.degrees(t['psi_top'][i_peak])))
        psi_bot.append(float(np.degrees(t['psi_bot'][i_peak])))
    fig, ax = plt.subplots()
    ax.plot(J_arr / 1e11, psi_top, 's-', color='C3',
            label=r'top, $\psi$')
    ax.plot(J_arr / 1e11, psi_bot, 'o-', color='k',
            label=r'bot, $\psi$')
    ax.axhline(180.0, color='0.7', linestyle=':')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$\psi$ (deg)')
    ax.set_title('S44(c): DW magnetization angle vs J')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


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
