"""Plot and analyse the perturbation-recovery sweep.

For each `breathing_*.npz` produced by `sweep_breathing.py`:
  1. Subtract the equilibrium diameter d_eq (read from metadata).
  2. Fit d_top(t) - d_eq to a damped sinusoid
         A * exp(-t / tau) * cos(2*pi*f*t + phi)
     via scipy.optimize.curve_fit. Initial guesses come from
     the first zero-crossing (period -> f) and amplitude decay
     between two early peaks (tau).
  3. Produce a 3-panel figure:
        - all d(t) traces overlaid (colour by D).
        - extracted f(D).
        - extracted tau(D), log-scale on y.
     Add a vertical line for the analytical D_c.

Reads
-----
output/sweeps_S41_S49/S41_breathing/breathing_*.npz

Writes
------
output/figures_S41_S49/breathing_diag.png
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import json
import math
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
# Local
from src.simulator.parameters import default_params

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _model(t, A, tau, f, phi, c):
    """Damped sinusoid + DC offset.  c absorbs residual drift."""
    return A * np.exp(-t / tau) * np.cos(2.0 * math.pi * f * t
                                         + phi) + c


def _initial_guess(t, y):
    """Rough (A, tau, f, phi, c) from the first peak/zero."""
    # DC offset from late-time mean (last 20% of samples).
    n = len(t)
    c0 = float(np.mean(y[int(0.8 * n):]))
    y_centred = y - c0
    A0 = float(np.max(np.abs(y_centred)))
    # Period estimate: distance between first and second sign
    # change of y - c0, doubled.
    s = np.sign(y_centred)
    zero_cross = np.where(np.diff(s) != 0)[0]
    if len(zero_cross) >= 2:
        T = 2.0 * (t[zero_cross[1]] - t[zero_cross[0]])
        f0 = 1.0 / T if T > 0 else 1.0e10
    else:
        # Fall back: assume 5 ns period if no oscillation
        # visible (very soft mode).
        f0 = 2.0e8
    # Decay: assume amplitude drops by factor e over ~half the
    # sampled window.
    tau0 = 0.5 * (t[-1] - t[0])
    phi0 = 0.0
    return [A0, tau0, f0, phi0, c0]


# Load the 10 ns breathing runs, D-sorted, as plain dict records.
def _load(in_dir):
    # Only the 10 ns variants; the duration tag is encoded in the
    # filename (`..._10ns_...`) so a glob suffices and avoids
    # opening the NPZ just to read metadata.
    paths = sorted(glob.glob(
        os.path.join(in_dir, 'breathing_*_10ns_*.npz')))
    if not paths:
        raise RuntimeError(
            f'plot_breathing: no breathing_*_10ns_*.npz in '
            f'{in_dir}.')
    records = []
    for p in paths:
        z = np.load(p, allow_pickle=True)
        md = json.loads(z['_metadata'].item())
        records.append({
            'D_mJ': float(md['D']) * 1e3,
            'd_eq_nm': float(md['d_eq']) * 1e9,
            't_ps': np.asarray(z['t']) * 1e12,
            'd_top_nm': np.asarray(z['d_top']) * 1e9,
        })
    records.sort(key=lambda r: r['D_mJ'])
    return records


# Analytical critical DMI D_c = (4/pi) sqrt(A * K_eff), in mJ/m^2.
def _analytical_Dc():
    p = default_params()
    K_bar = 0.5 * (float(p.K_top) + float(p.K_bot))
    K_eff = K_bar - 0.5 * float(p.mu0) * float(p.Ms) ** 2
    if K_eff <= 0.0:
        return float('nan')
    return (4.0 / math.pi) * math.sqrt(float(p.A_ex) * K_eff) * 1e3


def _fit_one(t_ps, d_nm, d_eq_nm):
    """Return (f_GHz, tau_ps, success_flag)."""
    y = d_nm - d_eq_nm
    t_s = t_ps * 1e-12
    p0 = _initial_guess(t_s, y)
    try:
        # Bound tau and f to physical ranges.
        bounds = (
            [-50.0, 1.0e-12, 1.0e7, -math.pi, -5.0],
            [+50.0, 1.0e-7, 1.0e11, +math.pi, +5.0])
        popt, _ = curve_fit(_model, t_s, y, p0=p0,
                            bounds=bounds, maxfev=20000)
        _A, tau_s, f_Hz, _phi, _c = popt
        return float(f_Hz * 1e-9), float(tau_s * 1e12), True
    except Exception as e:
        print(f'  fit failed: {e}', flush=True)
        return float('nan'), float('nan'), False


# Build the 3-panel breathing/drift diagnostic figure.
def main():
    """Load the breathing sweep, extract per-D drift metrics, and
    write the three-panel perturbation-recovery figure."""
    in_dir = 'output/sweeps_S41_S49/S41_breathing'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'breathing_diag.png')
    Dc_mJ = _analytical_Dc()
    records = _load(in_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Drift diagnostics. The breathing-mode period turns out to
    # exceed the 10 ns free-evolution window across the swept D
    # range, so d(t) - d_eq is monotone over the window rather
    # than oscillatory. We report two model-free drift metrics:
    #
    #   drift_total = d(t_end) - d(t = 0)        [nm]
    #   drift_rate  = drift_total / (t_end - t0) [m/s]
    #
    # A sign change in drift_total marks the D at which the
    # K_eff relax landed closest to the true (paper-alpha)
    # equilibrium; the magnitude rises away from that crossing
    # and is largest where the soft mode is weakest.
    drift_nm = []
    rate_mps = []
    for r in records:
        d = r['d_top_nm']
        t_s = r['t_ps'] * 1e-12
        dnm = float(d[-1] - d[0])
        drift_nm.append(dnm)
        rate_mps.append(dnm * 1e-9 / (t_s[-1] - t_s[0]))
        print(f'  D = {r["D_mJ"]:.3f} mJ/m^2: '
              f'd_eq={r["d_eq_nm"]:.1f} nm, '
              f'drift={dnm:+6.1f} nm, '
              f'rate={rate_mps[-1]:+6.1f} m/s',
              flush=True)
    D_arr = np.array([r['D_mJ'] for r in records])
    drift_arr = np.array(drift_nm)
    rate_arr = np.array(rate_mps)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Three-panel figure.
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    cmap = plt.get_cmap('viridis')
    n = len(records)
    # Panel A: d(t) overlays.
    for k, r in enumerate(records):
        c = cmap(k / max(n - 1, 1))
        axes[0].plot(r['t_ps'], r['d_top_nm'] - r['d_eq_nm'],
                     color=c, lw=1.0,
                     label=f'D = {r["D_mJ"]:.2f}')
    axes[0].axhline(0, color='0.7', ls=':')
    axes[0].set_xlabel('$t$ (ps)')
    axes[0].set_ylabel(r'$d_\mathrm{top}(t) - d_\mathrm{eq}$ (nm)')
    axes[0].set_title('(a) Breathing recovery')
    axes[0].legend(loc='best', frameon=False, fontsize=8)
    axes[0].set_box_aspect(1)
    # Panel B: total drift vs D.
    axes[1].plot(D_arr, drift_arr, 'o-', color='C0')
    axes[1].axhline(0, color='0.7', ls=':')
    if not math.isnan(Dc_mJ):
        axes[1].axvline(Dc_mJ, ls='--', color='0.4',
                        label=f'$D_c$ = {Dc_mJ:.3f}')
        axes[1].legend(loc='best', frameon=False)
    axes[1].set_xlabel('$D$ (mJ/m$^2$)')
    axes[1].set_ylabel(r'$d(t_\mathrm{end}) - d(0)$ (nm)')
    axes[1].set_title('(b) Total drift over free window')
    axes[1].set_box_aspect(1)
    # Panel C: drift rate vs D.
    axes[2].plot(D_arr, rate_arr, 's-', color='C3')
    axes[2].axhline(0, color='0.7', ls=':')
    if not math.isnan(Dc_mJ):
        axes[2].axvline(Dc_mJ, ls='--', color='0.4')
    axes[2].set_xlabel('$D$ (mJ/m$^2$)')
    axes[2].set_ylabel(r'$\dot d$ (m/s)')
    axes[2].set_title('(c) Drift rate vs $D$')
    axes[2].set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'Saved {out_path}', flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
