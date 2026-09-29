"""Arrhenius analysis: tau(T) at j = 0 (Q4 intrinsic).

Consumes
    output/stochastic_llgs/scan_arrhenius/aggregate.npz
(written by `scripts.aggregate_sllg scan_arrhenius`). Fits
`log(tau) = log(tau_0) + dE / (k_B T)` and writes the
Arrhenius plot.

Run with:
    python -m scripts.analyze_sllg_scan_arrhenius
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import math
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults (S4X style).
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


def main():
    """Load the Arrhenius aggregate, fit log(tau) against 1/T, and
    write the Arrhenius figure and the fit NPZ."""
    # =========================== User Configuration =========================
    in_npz        = (
        'output/stochastic_llgs/scan_arrhenius/aggregate.npz')
    fig_dir       = 'output/figures_sllg'
    fig_path      = os.path.join(fig_dir, 'scan_arrhenius.png')
    fit_npz       = (
        'output/stochastic_llgs/scan_arrhenius/fit.npz')
    # Minimum flips per T-cell to include it in the fit.
    min_flips_fit = 5
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.isfile(in_npz):
        raise RuntimeError(
            f'{in_npz!r} not found. Run '
            f'`python -m scripts.aggregate_sllg '
            f'scan_arrhenius` first.'
        )
    d = np.load(in_npz, allow_pickle=True)
    T_arr = d['T_arr']
    tau = d['tau_mle']
    n_flipped = d['n_flipped']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Arrhenius fit: only use cells with enough flips for a
    # reliable tau, then fit log(tau) vs 1/T for dE and tau_0.
    fit_mask = n_flipped >= min_flips_fit
    k_B = 1.380649e-23
    if int(fit_mask.sum()) >= 2:
        inv_T = 1.0 / T_arr[fit_mask]
        log_tau = np.log(tau[fit_mask])
        slope, intercept = np.polyfit(inv_T, log_tau, 1)
        dE = float(slope * k_B)
        tau_0 = float(math.exp(intercept))
        print(
            f'Arrhenius fit on {int(fit_mask.sum())} cells: '
            f'dE = {dE/1.602e-19*1e3:.3f} meV  '
            f'tau_0 = {tau_0:.3e} s'
        )
    else:
        slope = float('nan')
        intercept = float('nan')
        dE = float('nan')
        tau_0 = float('nan')
        print(
            f'WARN: only {int(fit_mask.sum())} cells with '
            f'>= {min_flips_fit} flips; fit skipped. '
            f'Increase the highest T_sub or n_drive in '
            f'scan_arrhenius.py.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist the fit parameters for the report/downstream use.
    np.savez_compressed(
        fit_npz,
        T_arr=T_arr, tau=tau,
        n_flipped=n_flipped, fit_mask=fit_mask,
        slope=slope, intercept=intercept,
        dE_joule=dE,
        dE_meV=dE / 1.602e-19 * 1e3,
        tau_0=tau_0, min_flips_fit=int(min_flips_fit),
    )
    print(f'Saved {fit_npz}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Arrhenius plot: tau vs 1000/T, fit cells vs excluded
    # cells, with the fitted line overlaid when available.
    fig, ax = plt.subplots()
    if np.any(fit_mask):
        ax.semilogy(
            1000.0 / T_arr[fit_mask], tau[fit_mask],
            'o', label='MLE (fit cells)',
        )
    if np.any(~fit_mask):
        ax.semilogy(
            1000.0 / T_arr[~fit_mask], tau[~fit_mask],
            'x', label=r'$< 5$ flips (excluded)',
            color='gray',
        )
    if np.isfinite(slope):
        Tfit = np.linspace(T_arr.min(), T_arr.max(), 200)
        tau_fit = np.exp(intercept + slope / Tfit)
        ax.semilogy(
            1000.0 / Tfit, tau_fit, '-',
            label=(
                rf'fit: $\Delta E={dE/1.602e-19*1e3:.2f}$ meV, '
                rf'$\tau_0={tau_0:.2e}$ s'
            ),
        )
    ax.set_xlabel(r'$1000 / T$ (K$^{-1}$)')
    ax.set_ylabel(r'$\tau$ (MLE) (s)')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f'Saved {fig_path}')


# =============================================================================
if __name__ == '__main__':
    main()
