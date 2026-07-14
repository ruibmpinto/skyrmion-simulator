"""Plotter for Pham et al. (2024) Figure S48.

Reads `output/sweeps_S41_S49/S48/*.npz` and emits:

S48_A_vt_rise.png
    Skyrmion velocity v(t) zoom on the rising edge of the
    2 ns square pulse, one trace per H_RKKY.
S48_B_vt_fall.png
    Same but on the falling edge.
S48_D_vt_full.png
    Complete v(t) trace for every H_RKKY with the DC pulse
    window shaded.
S48_C_invtau.png
    Inverse time constant 1/tau_d from a damped-sinusoid fit
    of the rising and falling edges, averaged over the
    {top, bot} layers and the {rise, fall} edges (up to 4
    values per H_RKKY) with the standard deviation as the
    error bar.
S48_E_freq.png
    Inertial oscillation frequency f from the same fit, with
    std-dev error bars.

The fits use scipy.optimize.curve_fit when available;
without scipy the fit is skipped and 1/tau_d and f are
reported as NaN.
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

try:
    from scipy.optimize import curve_fit
    _HAVE_SCIPY = True
except Exception:
    _HAVE_SCIPY = False

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


def _v_inst(t, cx, cy):
    """Centred-difference velocity components and magnitude."""
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    return np.sqrt(vx * vx + vy * vy), vx, vy


def _load_sweep(in_dir):
    """Collect every NPZ in the sweep directory, sorted by
    H_RKKY."""
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    items = []
    for path in paths:
        trace, metadata = load_trace(path)
        items.append((metadata, trace))
    # Sort by RKKY field for deterministic colour ordering.
    items.sort(key=lambda mt: float(mt[0]['H_RKKY']))
    return items


def _damped_sinusoid(tt, A, B, tau_d, f, phi, t0):
    """v(t) = A + B exp(-(t-t0)/tau_d) [cos(phi) - cos(2 pi f (t-t0) + phi)]

    Boundary conditions:
      v(t0) = A   (no transient yet)
      v(t -> inf) = A   (steady state)
    The envelope `B exp(-(t-t0)/tau_d)` controls the amplitude of
    the inertial oscillation; `f` is its frequency. This shape
    captures the overshoot-and-ringing seen in the SAF rise and
    fall transients without forcing a pure exponential."""
    dt = tt - t0
    env = np.exp(-dt / tau_d)
    return A + B * env * (np.cos(phi) - np.cos(2.0 * np.pi * f * dt + phi))


def _fit_damped_sinusoid(t, y, mode):
    """Fit the damped sinusoid above to either the rising or
    falling edge of v(t) and return (f_Hz, tau_d_s).

    `mode` is kept for symmetry with the old fit API but only
    influences initial guesses (rise: A = asymptotic plateau;
    fall: A = late-time plateau, which should be ~0).
    Returns (nan, nan) on fit failure or fewer than 5 samples."""
    if y.size < 5 or not _HAVE_SCIPY:
        return float('nan'), float('nan')
    t0 = float(t[0])
    span = float(t[-1] - t[0])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial guesses.
    # Late-time mean is a robust estimate of A (the plateau).
    A0 = float(np.mean(y[-max(3, len(y) // 5):]))
    # First-peak amplitude is a rough scale for B.
    B0 = float(np.max(np.abs(y - A0)))
    if B0 == 0.0:
        B0 = 1.0
    # Decay time: half the fit window is a neutral starting point.
    tau0 = 0.5 * span
    # Frequency: try peak-to-peak spacing if visible, else 5 GHz.
    centred = y - A0
    sign_changes = np.where(np.diff(np.sign(centred)) != 0)[0]
    if len(sign_changes) >= 2:
        period = 2.0 * (t[sign_changes[1]] - t[sign_changes[0]])
        f0 = 1.0 / period if period > 0 else 5.0e9
    else:
        f0 = 5.0e9
    phi0 = 0.0
    p0 = (A0, B0, tau0, f0, phi0)
    # Loose physical bounds.
    bounds = (
        [-1.0e3, -1.0e3, 1.0e-12, 1.0e8, -2.0 * np.pi],
        [+1.0e3, +1.0e3, 1.0e-7, 1.0e11, +2.0 * np.pi],
    )

    def wrapped(tt, A, B, tau_d, f, phi):
        return _damped_sinusoid(tt, A, B, tau_d, f, phi, t0)

    try:
        popt, _ = curve_fit(
            wrapped, t, y, p0=p0, bounds=bounds, maxfev=20000)
        _A, _B, tau_d, f_Hz, _phi = popt
        return float(abs(f_Hz)), float(abs(tau_d))
    except Exception:
        return float('nan'), float('nan')


def _per_layer_speed(trace, metadata, layer):
    """Speed (m/s) from finite differences for `layer` ('top'
    or 'bot'), with PBC unwrap applied to the centroid stream."""
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    cx, cy = unwrap_trajectory(
        trace[f'cx_{layer}'], trace[f'cy_{layer}'],
        L_x=nx * a, L_y=ny * a,
    periodic_y=True)
    v, _vx, _vy = _v_inst(t, cx, cy)
    return t, v


def _plot_edges(items, edge, out_path):
    """Plot the velocity transient on one edge (rise or fall),
    one trace per H_RKKY."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    H_keys = [float(m['H_RKKY']) for m, _ in items]
    H_min = float(min(H_keys))
    H_max = float(max(H_keys))
    for k, (metadata, trace) in enumerate(items):
        t, v = _per_layer_speed(trace, metadata, 'top')
        t_pulse = float(metadata['t_pulse'])
        if edge == 'rise':
            window = (t >= 0.0) & (t <= 0.2e-9)
        else:
            window = (t >= t_pulse) & (t <= t_pulse + 0.3e-9)
        if not np.any(window):
            continue
        c = cmap(
            (float(metadata['H_RKKY']) - H_min)
            / max(H_max - H_min, 1e-30))
        ax.plot(
            (t[window] - (0.0 if edge == 'rise' else t_pulse))
            * 1e12,
            v[window],
            color=c,
            label=f'{float(metadata["H_RKKY"])*1e3:.0f} mT')
    ax.set_xlabel(
        r'$t - t_{\mathrm{rise}}$ (ps)' if edge == 'rise'
        else r'$t - t_{\mathrm{fall}}$ (ps)')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.legend(loc='best', frameon=False, fontsize=9, ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_full(items, out_path):
    """Plot the complete v(t) trace for every H_RKKY, with the
    DC pulse window shaded. Shows the full rise, plateau, and
    fall in one panel rather than the zoomed-in transients of
    panels A and B."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    H_keys = [float(m['H_RKKY']) for m, _ in items]
    H_min = float(min(H_keys))
    H_max = float(max(H_keys))
    # Shade the pulse window; uses the first trace's t_pulse
    # (uniform across the sweep).
    t_pulse = float(items[0][0]['t_pulse'])
    ax.axvspan(
        0.0, t_pulse * 1e12,
        color='violet', alpha=0.15,
        label=f'pulse on (0--{t_pulse*1e9:.1f} ns)')
    for metadata, trace in items:
        t, v = _per_layer_speed(trace, metadata, 'top')
        c = cmap(
            (float(metadata['H_RKKY']) - H_min)
            / max(H_max - H_min, 1e-30))
        ax.plot(t * 1e12, v, color=c,
                label=f'{float(metadata["H_RKKY"])*1e3:.0f} mT')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.legend(loc='best', frameon=False, fontsize=9, ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _collect_fits(items):
    """For every H_RKKY, fit the damped sinusoid to each of the
    4 edges (top/bot x rise/fall) and return arrays
    (H_mT, mean(1/tau), std(1/tau), mean(f_GHz), std(f_GHz))."""
    H_arr, mean_invtau, std_invtau = [], [], []
    mean_f, std_f = [], []
    for metadata, trace in items:
        t_pulse = float(metadata['t_pulse'])
        taus, freqs = [], []
        for layer in ('top', 'bot'):
            t, v = _per_layer_speed(trace, metadata, layer)
            # Rise window: 0 to 500 ps captures overshoot + at
            # least one full damped cycle; fall window:
            # t_pulse to t_pulse+500 ps. Wider window improves
            # frequency localisation in the fit.
            wr = (t >= 0.0) & (t <= 0.5e-9)
            if np.sum(wr) >= 5:
                f_r, tau_r = _fit_damped_sinusoid(
                    t=t[wr], y=v[wr], mode='rise')
                if np.isfinite(tau_r) and np.isfinite(f_r):
                    taus.append(tau_r)
                    freqs.append(f_r)
            wf = (t >= t_pulse) & (t <= t_pulse + 0.5e-9)
            if np.sum(wf) >= 5:
                f_f, tau_f = _fit_damped_sinusoid(
                    t=t[wf], y=v[wf], mode='fall')
                if np.isfinite(tau_f) and np.isfinite(f_f):
                    taus.append(tau_f)
                    freqs.append(f_f)
        if not taus:
            continue
        H_arr.append(float(metadata['H_RKKY']) * 1e3)
        inv_tau = 1.0 / np.array(taus)
        f_arr = np.array(freqs) * 1e-9
        mean_invtau.append(float(np.mean(inv_tau)))
        std_invtau.append(float(np.std(inv_tau)))
        mean_f.append(float(np.mean(f_arr)))
        std_f.append(float(np.std(f_arr)))
        print(f'  H_RKKY = {H_arr[-1]:>4.0f} mT: '
              f'1/tau = {mean_invtau[-1]:.2e} +/- '
              f'{std_invtau[-1]:.2e} 1/s, '
              f'f = {mean_f[-1]:.2f} +/- {std_f[-1]:.2f} GHz '
              f'(n={len(taus)})')
    return (np.array(H_arr), np.array(mean_invtau),
            np.array(std_invtau), np.array(mean_f),
            np.array(std_f))


def _plot_invtau(H_arr, mean_invtau, std_invtau, out_path):
    """Inverse damping time with std-dev error bars."""
    fig, ax = plt.subplots()
    ax.errorbar(H_arr, mean_invtau, yerr=std_invtau,
                fmt='o-', color='C0', capsize=4)
    ax.set_xlabel(r'$H_{\mathrm{RKKY}}$ (mT)')
    ax.set_ylabel(r'$1/\tau_d$ (1/s)')
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_freq(H_arr, mean_f, std_f, out_path):
    """Inertial oscillation frequency with std-dev error bars."""
    fig, ax = plt.subplots()
    ax.errorbar(H_arr, mean_f, yerr=std_f,
                fmt='s-', color='C3', capsize=4)
    ax.set_xlabel(r'$H_{\mathrm{RKKY}}$ (mT)')
    ax.set_ylabel(r'inertial $f$ (GHz)')
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def main():
    """Load the S48 sweep and write the rise/fall/full v(t) panels
    plus the fitted 1/tau and inertial-frequency panels."""
    in_dir = 'output/sweeps_S41_S49/S48'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    items = _load_sweep(in_dir)
    print(f'S48: loaded {len(items)} traces')
    _plot_edges(
        items, edge='rise',
        out_path=os.path.join(out_dir, 'S48_A_vt_rise.png'))
    _plot_edges(
        items, edge='fall',
        out_path=os.path.join(out_dir, 'S48_B_vt_fall.png'))
    _plot_full(
        items,
        out_path=os.path.join(out_dir, 'S48_D_vt_full.png'))
    # Damped-sinusoid fits: collect once, plot both panels.
    H_arr, mean_invtau, std_invtau, mean_f, std_f = _collect_fits(items)
    _plot_invtau(
        H_arr, mean_invtau, std_invtau,
        out_path=os.path.join(out_dir, 'S48_C_invtau.png'))
    _plot_freq(
        H_arr, mean_f, std_f,
        out_path=os.path.join(out_dir, 'S48_E_freq.png'))


# =============================================================================
if __name__ == '__main__':
    main()
