"""Plotter for Pham et al. (2024) Figure S42.

Two panels:
(a) Instantaneous skyrmion velocity vs time, one trace per
    sweep point, the Gaussian pulse shape superimposed on
    a secondary y-axis.
(b) Maximum instantaneous velocity and average velocity
    (Delta x / FWHM) versus current density J.

Reads every NPZ in `output/sweeps_S41_S49/S42/`, identifies
each by its `J0` metadata key, and writes two PNGs into
`output/figures_S41_S49/`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import math
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


def _load_sweep(in_dir):
    """Load every NPZ in `in_dir`, sorted by `J0` metadata.

    Returns a list of `(metadata, trace)` tuples.
    """
    # Sort by file path for a deterministic order, then re-sort
    # by J0 once metadata is loaded.
    paths = sorted(glob.glob(os.path.join(in_dir, 'J_*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    sweep = []
    for path in paths:
        trace, metadata = load_trace(path)
        sweep.append((metadata, trace))
    # Sort once by J0 so figures render in ascending current.
    sweep.sort(key=lambda x: float(x[0]['J0']))
    return sweep


def _gaussian_J(t, J0, t_center, sigma):
    """Evaluate the Gaussian pulse at every t for overlay."""
    z = (t - t_center) / sigma
    return J0 * np.exp(-0.5 * z * z)


def _unwrapped(trace, metadata):
    """Unwrap the PBC-wrapped centroid stream for differencing."""
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    return unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
    periodic_y=True)


def _v_inst(t, cx, cy):
    """Instantaneous speed from centred finite differences."""
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    return np.sqrt(vx * vx + vy * vy)


# Build both S42 panels: v_inst(t) traces and v_max/v_avg vs J.
def main():
    """Load the S42 J sweep and write the v_inst(t) traces figure and
    the v_max / v_avg vs J summary figure."""
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # I/O configuration.
    in_dir = 'output/sweeps_S41_S49/S42'
    out_dir = 'output/figures_S41_S49'
    out_traces = os.path.join(out_dir, 'S42_v_t.png')
    out_summary = os.path.join(out_dir, 'S42_vmax_vavg.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load sweep.
    sweep = _load_sweep(in_dir)
    print(f'S42: loaded {len(sweep)} traces from {in_dir}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel (a): v_inst(t) traces, Gaussian on secondary axis.
    fig_a, ax_a = plt.subplots()
    # Secondary axis for the pulse shape.
    ax_a2 = ax_a.twinx()
    cmap = plt.get_cmap('viridis')
    # Build a colour table proportional to J for clear ordering.
    J_array = np.array([float(m['J0']) for m, _ in sweep])
    J_min = float(J_array.min())
    J_max = float(J_array.max())
    norm_J = (J_array - J_min) / max(J_max - J_min, 1e-30)
    # Reuse the first trace's pulse parameters to draw the
    # Gaussian envelope on the right axis. The same t_center
    # and sigma are used for every J0 in this sweep.
    metadata0, trace0 = sweep[0]
    t_center = float(metadata0['t_center'])
    sigma = float(metadata0['sigma'])
    # Time grid in ps for plotting.
    t_ps = trace0['t'] * 1e12
    # Pulse shape evaluated on the trace's time grid for the
    # largest J in the sweep (matches the paper's right-y-axis
    # convention).
    pulse_envelope = _gaussian_J(trace0['t'], J_max, t_center, sigma)
    ax_a2.plot(
        t_ps, pulse_envelope * 1e-11,
        color='violet', linestyle='--',
        label=f'$J/10^{{11}}$ (max trace)',)
    ax_a2.set_ylabel(r'$J$ ($10^{11}$ A/m$^2$)', color='violet')
    ax_a2.tick_params(axis='y', labelcolor='violet')
    # Velocity traces (unwrap centroids before differencing).
    for (metadata, trace), nv in zip(sweep, norm_J):
        cx, cy = _unwrapped(trace, metadata)
        v = _v_inst(trace['t'], cx, cy)
        ax_a.plot(
            trace['t'] * 1e12, v,
            color=cmap(nv),
            label=fr'$J = {float(metadata["J0"]):.1e}$ A/m$^2$',)
    ax_a.set_xlabel(r'$t$ (ps)')
    ax_a.set_ylabel(r'$|v_{\mathrm{inst}}|$ (m/s)')
    # Extend xlim past the trace's end so the upper-right
    # legend has space to the right of the velocity peaks
    # instead of overlapping them.
    _span = t_ps[-1] - t_ps[0]
    ax_a.set_xlim(t_ps[0], t_ps[-1] + 0.55 * _span)
    ax_a2.set_xlim(t_ps[0], t_ps[-1] + 0.55 * _span)
    ax_a.legend(loc='upper right', frameon=False, fontsize=10)
    ax_a.set_box_aspect(1)
    fig_a.tight_layout()
    # Persist panel (a).
    os.makedirs(out_dir, exist_ok=True)
    fig_a.savefig(out_traces)
    print(f'Saved {out_traces}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel (b): v_max and v_avg vs J.
    v_max = []
    v_avg = []
    for metadata, trace in sweep:
        t = trace['t']
        cx, cy = _unwrapped(trace, metadata)
        v = _v_inst(t, cx, cy)
        v_max.append(float(np.max(v)))
        # v_avg = (delta x at the pulse edges) / FWHM, where
        # the pulse edges are the +/-3 sigma window stored in
        # the trace metadata.
        FWHM = float(metadata['FWHM'])
        # Indices at +/- (tail_sigmas * sigma) around t_center.
        tail = float(metadata['tail_sigmas']) * float(metadata['sigma'])
        t_lo = float(metadata['t_center']) - tail
        t_hi = float(metadata['t_center']) + tail
        i_lo = int(np.argmin(np.abs(t - t_lo)))
        i_hi = int(np.argmin(np.abs(t - t_hi)))
        dx = cx[i_hi] - cx[i_lo]
        dy = cy[i_hi] - cy[i_lo]
        v_avg.append(float(np.sqrt(dx * dx + dy * dy) / FWHM))
    # Analytical Thiele model under a Gaussian pulse, rigid
    # skyrmion (no deformation). Instantaneous response gives
    #   v(t) = coef * J(t),
    #   coef = pi * DL_SOT * R * gamma /
    #          (2 * alpha * (R/Delta + Delta/R)).
    # Hence
    #   v_max^an  = coef * J0,
    #   v_avg^an  = (1/FWHM) * integral_{-tail*sigma}^{tail*sigma}
    #               coef * J0 * exp(-z^2/2) sigma dz
    #             = coef * J0 * sigma * sqrt(2 pi)
    #               * erf(tail / sqrt(2)) / FWHM.
    p = default_params()
    R = float(p.skyrmion_R)
    Delta = float(p.skyrmion_dw)
    coef = (np.pi * p.DL_SOT * R * p.gamma
            / (2.0 * p.alpha * (R / Delta + Delta / R)))
    # Pulse shape parameters are identical across the sweep, so
    # read them from the first metadata block.
    FWHM_a = float(metadata0['FWHM'])
    sigma_a = float(metadata0['sigma'])
    tail_a = float(metadata0['tail_sigmas'])
    avg_ratio = (sigma_a * np.sqrt(2.0 * np.pi)
                 * math.erf(tail_a / np.sqrt(2.0)) / FWHM_a)
    J_grid = np.linspace(0.0, float(J_array.max()), 200)
    v_max_analytic = coef * J_grid
    v_avg_analytic = coef * J_grid * avg_ratio
    # Plot.
    fig_b, ax_b = plt.subplots()
    ax_b.plot(J_array / 1e11, v_max, 'o-',
              label=r'$v_{\mathrm{max}}$ (sim.)', color='C0')
    ax_b.plot(J_array / 1e11, v_avg, 's-',
              label=r'$v_{\mathrm{avg}}$ (sim.)', color='C3')
    ax_b.plot(J_grid / 1e11, v_max_analytic, '--',
              color='C0',
              label=r'$v_{\mathrm{max}}$ (Thiele)')
    ax_b.plot(J_grid / 1e11, v_avg_analytic, '--',
              color='C3',
              label=r'$v_{\mathrm{avg}}$ (Thiele)')
    ax_b.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax_b.set_ylabel(r'$v$ (m/s)')
    ax_b.legend(loc='lower right', frameon=False, fontsize=11)
    ax_b.set_box_aspect(1)
    fig_b.tight_layout()
    fig_b.savefig(out_summary)
    print(f'Saved {out_summary}')
    # Console summary.
    for J0, vm, va in zip(J_array, v_max, v_avg):
        print(f'  J = {J0:.2e}: v_max = {vm:6.1f}, v_avg = {va:6.1f} m/s')


# =============================================================================
if __name__ == '__main__':
    main()
