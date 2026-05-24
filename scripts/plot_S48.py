"""Plotter for Pham et al. (2024) Figure S48.

Reads `output/sweeps_S41_S49/S48/HRKKY_*.npz` and emits:

S48_A_vt_rise.png
    Skyrmion velocity v(t) zoom on the rising edge of the
    2 ns square pulse, one trace per H_RKKY.
S48_B_vt_fall.png
    Same but on the falling edge.
S48_C_invtau.png
    Inverse time constant 1/tau extracted by exponential fit
    of the rising and falling edges, averaged over the
    {top, bot} layers and the {rise, fall} edges (4 values
    per H_RKKY) plotted with the standard deviation as the
    error bar.

The exponential fits use scipy.optimize.curve_fit when
available; otherwise a closed-form 3-point estimate is used.
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
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    return np.sqrt(vx * vx + vy * vy), vx, vy


def _load_sweep(in_dir):
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    items = []
    for path in paths:
        trace, metadata = load_trace(path)
        items.append((metadata, trace))
    items.sort(key=lambda mt: float(mt[0]['H_RKKY']))
    return items


def _fit_exponential(t, y, mode):
    """Fit either a rising or falling exponential and return tau.

    For mode == 'rise':  y(t) = A * (1 - exp(-(t - t0) / tau))
    For mode == 'fall':  y(t) = A * exp(-(t - t0) / tau)

    `t0` is the first sample, `A` is the asymptote (for rise)
    or the initial value (for fall). Returns NaN on fit failure.
    """
    if y.size < 3:
        return float('nan')
    t0 = float(t[0])
    if mode == 'rise':
        def model(tt, A, tau):
            return A * (1.0 - np.exp(-(tt - t0) / tau))
        p0 = (float(np.max(y)), float((t[-1] - t[0]) / 3.0))
    elif mode == 'fall':
        def model(tt, A, tau):
            return A * np.exp(-(tt - t0) / tau)
        p0 = (float(y[0]), float((t[-1] - t[0]) / 3.0))
    else:
        raise RuntimeError(
            f'_fit_exponential: unknown mode {mode!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if _HAVE_SCIPY:
        try:
            popt, _ = curve_fit(model, t, y, p0=p0, maxfev=5000)
            return float(abs(popt[1]))
        except Exception:
            return float('nan')
    # Fallback: 1/e crossing estimate.
    if mode == 'rise':
        asymp = float(np.max(y))
        target = (1.0 - 1.0 / np.e) * asymp
    else:
        target = float(y[0]) / np.e
    above = (y >= target) if mode == 'fall' else (y <= target)
    idx = np.argmax(~above) if mode == 'rise' else np.argmax(above)
    if idx == 0:
        return float('nan')
    return float(abs(t[idx] - t0))


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
    )
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
    ax.set_title(
        f'S48({"a" if edge == "rise" else "b"}): '
        f'{edge} transient')
    ax.legend(loc='best', frameon=False, fontsize=9, ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_invtau(items, out_path):
    """1/tau vs H_RKKY with error bars over the 4 fits."""
    H_arr = []
    inv_tau_mean = []
    inv_tau_std = []
    for metadata, trace in items:
        t_pulse = float(metadata['t_pulse'])
        taus = []
        for layer in ('top', 'bot'):
            t, v = _per_layer_speed(trace, metadata, layer)
            # Rise window: 0 to 200 ps.
            wr = (t >= 0.0) & (t <= 0.2e-9)
            if np.sum(wr) >= 3:
                tau_r = _fit_exponential(
                    t=t[wr], y=v[wr], mode='rise')
                if np.isfinite(tau_r):
                    taus.append(tau_r)
            # Fall window: t_pulse to t_pulse + 300 ps.
            wf = (t >= t_pulse) & (t <= t_pulse + 0.3e-9)
            if np.sum(wf) >= 3:
                tau_f = _fit_exponential(
                    t=t[wf], y=v[wf], mode='fall')
                if np.isfinite(tau_f):
                    taus.append(tau_f)
        if not taus:
            continue
        H_arr.append(float(metadata['H_RKKY']) * 1e3)
        inv_tau = 1.0 / np.array(taus)
        inv_tau_mean.append(float(np.mean(inv_tau)))
        inv_tau_std.append(float(np.std(inv_tau)))
        print(f'  H_RKKY = {float(metadata["H_RKKY"])*1e3:>4.0f} mT: '
              f'1/tau = {np.mean(inv_tau):.2e} +/- '
              f'{np.std(inv_tau):.2e} 1/s (n={len(taus)})')
    fig, ax = plt.subplots()
    ax.errorbar(H_arr, inv_tau_mean, yerr=inv_tau_std,
                fmt='o-', color='C0', capsize=4)
    ax.set_xlabel(r'$H_{\mathrm{RKKY}}$ (mT)')
    ax.set_ylabel(r'$1/\tau$ (1/s)')
    ax.set_title(r'S48(c): inverse time constant vs $H_{\mathrm{RKKY}}$')
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def main():
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
    _plot_invtau(
        items,
        out_path=os.path.join(out_dir, 'S48_C_invtau.png'))


# =============================================================================
if __name__ == '__main__':
    main()
