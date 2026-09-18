"""Lattice-resolution comparison of the hk36 track-width case at cell
sizes a = 2 nm and a = 3 nm.

The two runs share the same physical racetrack (about 1400 x 1000 nm)
and the same continuum parameters (D, K_top, M_s), differing only in the
discretization: 700 x 500 cells at a = 2 nm versus 467 x 333 cells at
a = 3 nm. This script overlays their survival (stability) maps and their
transport observables (drift speed and skyrmion Hall angle) on the shared
temperature x current grid, to separate discretization artifacts from
physical behavior.

Functions
---------
main
    Load both aggregates and emit the side-by-side comparison figure.

Notes
-----
Reads only the aggregate.npz of each case (no large dumps). Survival is
P_surv from the classifier; transport cells are masked to P_surv >= 0.5.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Pinto (rui_pinto@brown.edu)'
__credits__ = ['Rui Pinto', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _survival_map(ax, ts, js, p_surv, title):
    """Draw one survival (P_surv) heatmap on `ax`.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    ts : numpy.ndarray(1d)
        Temperatures (K), rows.
    js : numpy.ndarray(1d)
        Current densities (A/m^2), columns.
    p_surv : numpy.ndarray(2d)
        Survival probability, shape (len(ts), len(js)).
    title : str
        Panel title.

    Returns
    -------
    im : matplotlib.image.AxesImage
        The drawn image (for a shared colorbar).
    """
    im = ax.imshow(p_surv, origin='lower', aspect='auto', vmin=0.0,
                   vmax=1.0, cmap='viridis')
    ax.set_xticks(range(len(js)))
    ax.set_xticklabels(['%.1f' % (j / 1e11) for j in js])
    ax.set_yticks(range(len(ts)))
    ax.set_yticklabels(['%.0f' % t for t in ts])
    ax.set_xlabel(r'$J$ ($10^{11}$ A m$^{-2}$)', fontsize=14)
    ax.set_ylabel(r'$T$ (K)', fontsize=14)
    ax.set_title(title, fontsize=14)
    for i in range(len(ts)):
        for k in range(len(js)):
            ax.text(k, i, '%.2f' % p_surv[i, k], ha='center',
                    va='center', color='w', fontsize=7)
    return im
# -----------------------------------------------------------------------------
def _transport_overlay(ax, js, y_a2, y_a3, mask_a2, mask_a3, ylabel,
                       rows, ts):
    """Overlay a transport observable vs current for both lattices.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    js : numpy.ndarray(1d)
        Current densities (A/m^2).
    y_a2, y_a3 : numpy.ndarray(2d)
        Observable on the (T, J) grid for a = 2 nm and a = 3 nm.
    mask_a2, mask_a3 : numpy.ndarray(2d, bool)
        Surviving-cell mask (P_surv >= 0.5) for each lattice.
    ylabel : str
        y-axis label.
    rows : tuple[int]
        Temperature-row indices to plot.
    ts : numpy.ndarray(1d)
        Temperatures (K), for the legend.
    """
    colors = plt.cm.plasma(np.linspace(0.15, 0.75, len(rows)))
    j_scaled = js / 1e11
    for c, i in zip(colors, rows):
        ya = np.where(mask_a2[i], y_a2[i], np.nan)
        yb = np.where(mask_a3[i], y_a3[i], np.nan)
        ax.plot(j_scaled, ya, '-o', color=c,
                label=r'$a=2$ nm, $T=%.0f$ K' % ts[i])
        ax.plot(j_scaled, yb, '--s', color=c,
                label=r'$a=3$ nm, $T=%.0f$ K' % ts[i])
    ax.set_xlabel(r'$J$ ($10^{11}$ A m$^{-2}$)', fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=16, framealpha=0.9)
# -----------------------------------------------------------------------------
def main():
    """Emit the a = 2 nm vs a = 3 nm lattice-comparison figure."""
    # =========================== User Configuration =========================
    campaign_root = ('/Volumes/T7/skyrmion_simulator/output/'
                     'stochastic_llgs/scan_track_width/campaign')
    tag_a2 = 'hk36_D0p72_700x500'
    tag_a3 = 'hk36_D0p72_467x333_a3nm'
    out_path = ('output/figures_sllg/track_width/'
                'hk36_a3_vs_a2_comparison.png')
    overlay_rows = (0, 2)      # T = 10 K and T = 100 K
    # ======================= End User Configuration =========================
    a2 = np.load(os.path.join(campaign_root, tag_a2, 'aggregate.npz'))
    a3 = np.load(os.path.join(campaign_root, tag_a3, 'aggregate.npz'))
    ts = a2['Ts']
    js = a2['Js']
    if not (np.allclose(ts, a3['Ts']) and np.allclose(js, a3['Js'])):
        raise RuntimeError('a2 and a3 grids differ; cannot overlay.')
    mask_a2 = a2['P_surv'] >= 0.5
    mask_a3 = a3['P_surv'] >= 0.5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(2, 2, figsize=(11, 9),
                             constrained_layout=True)
    im = _survival_map(
        axes[0, 0], ts, js, a2['P_surv'],
        r'Survival $P_{\mathrm{surv}}$, $a=2$ nm (700$\times$500)')
    _survival_map(
        axes[0, 1], ts, js, a3['P_surv'],
        r'Survival $P_{\mathrm{surv}}$, $a=3$ nm (467$\times$333)')
    cbar = fig.colorbar(im, ax=axes[0, :].tolist(), shrink=0.85,
                        location='right')
    cbar.set_label(r'$P_{\mathrm{surv}}$', fontsize=14)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    _transport_overlay(
        axes[1, 0], js, a2['v_mean'], a3['v_mean'], mask_a2, mask_a3,
        r'$\langle v \rangle$ (m s$^{-1}$)', overlay_rows, ts)
    _transport_overlay(
        axes[1, 1], js, a2['theta_mean'], a3['theta_mean'], mask_a2,
        mask_a3, r'$\theta_{\mathrm{H}}$ (deg)', overlay_rows, ts)
    axes[1, 1].axhline(0.0, color='k', lw=0.8, ls=':')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('wrote %s' % out_path)
# =============================================================================
if __name__ == '__main__':
    main()
