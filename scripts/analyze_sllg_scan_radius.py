"""(D, H_z) radius analysis: velocity, Hall angle, and
inter-layer (top-bottom) coupling as a function of skyrmion
size (Q3).

Consumes
    output/stochastic_llgs/scan_radius/aggregate.npz
(written by `scripts.aggregate_sllg scan_radius`). Produces a
four-panel figure:

    (a) <v_top> vs <d_top>          - velocity-vs-size
    (b) <theta_H_top> vs <d_top>    - Hall angle-vs-size
    (c) <|r_top - r_bot|> vs <d_top> - inter-layer offset
    (d) <Q_top + Q_bot> vs <d_top>   - compensation residual

Panels (c, d) probe the inter-layer Q3 (the SAF "pair" =
top-Co + bot-Co coupled by RKKY + demag).

Run with:
    python -m scripts.analyze_sllg_scan_radius
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
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['lines.linewidth'] = 1.5


def main():
    """Load the (D, H_z) radius aggregate and write the four-panel
    velocity / Hall-angle / inter-layer figure versus skyrmion
    size."""
    # =========================== User Configuration =========================
    in_npz   = 'output/stochastic_llgs/scan_radius/aggregate.npz'
    fig_dir  = 'output/figures_sllg'
    fig_path = os.path.join(fig_dir, 'scan_radius.png')
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.isfile(in_npz):
        raise RuntimeError(
            f'{in_npz!r} not found. Run '
            f'`python -m scripts.aggregate_sllg scan_radius` '
            f'first.'
        )
    d = np.load(in_npz)
    D_arr = d['D_arr']
    Hz_arr = d['Hz_arr']
    p_surv = d['p_surv']
    # Top-layer drift / shape
    d_mean = d['d_mean']
    d_se = d['d_se']
    v_mean = d['v_mean']
    v_se = d['v_se']
    th_mean = d['theta_mean']
    th_se = d['theta_se']
    # Inter-layer diagnostics
    offset_mean = d['offset_mean']
    offset_se = d['offset_se']
    q_total_mean = d['q_total_mean']
    q_total_se = d['q_total_se']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Four-panel figure: velocity, Hall angle, inter-layer
    # offset, and compensation residual, all vs skyrmion size.
    fig, axes = plt.subplots(2, 2)
    xlbl = r'$\langle d_\mathrm{top} \rangle$ (nm)'
    # Velocity vs size.
    axes[0, 0].errorbar(d_mean * 1e9, v_mean,
                        xerr=d_se * 1e9, yerr=v_se,
                        fmt='o', capsize=3)
    axes[0, 0].set_xlabel(xlbl)
    axes[0, 0].set_ylabel(
        r'$\langle v_\mathrm{top} \rangle$ (m/s)')
    axes[0, 0].set_title('Velocity vs skyrmion size')
    axes[0, 0].set_box_aspect(1)
    # Hall angle vs size.
    axes[0, 1].errorbar(d_mean * 1e9, th_mean,
                        xerr=d_se * 1e9, yerr=th_se,
                        fmt='o', capsize=3)
    axes[0, 1].set_xlabel(xlbl)
    axes[0, 1].set_ylabel(
        r'$\langle \theta_H \rangle$ (deg)')
    axes[0, 1].set_title('Hall angle vs size')
    axes[0, 1].set_box_aspect(1)
    # Inter-layer center offset vs size.
    axes[1, 0].errorbar(d_mean * 1e9, offset_mean * 1e9,
                        xerr=d_se * 1e9, yerr=offset_se * 1e9,
                        fmt='o', capsize=3, color='C2')
    axes[1, 0].set_xlabel(xlbl)
    axes[1, 0].set_ylabel(
        r'$\langle |\, r_\mathrm{top} - r_\mathrm{bot}\,| '
        r'\rangle$ (nm)')
    axes[1, 0].set_title(
        'Top-bot lateral offset (inter-layer drag)')
    axes[1, 0].set_box_aspect(1)
    # Compensation residual vs size.
    axes[1, 1].errorbar(d_mean * 1e9, q_total_mean,
                        xerr=d_se * 1e9, yerr=q_total_se,
                        fmt='o', capsize=3, color='C3')
    axes[1, 1].axhline(0.0, color='k')
    axes[1, 1].set_xlabel(xlbl)
    axes[1, 1].set_ylabel(
        r'$\langle Q_\mathrm{top} + Q_\mathrm{bot} \rangle$')
    axes[1, 1].set_title('Compensation residual')
    axes[1, 1].set_box_aspect(1)
    # Annotate cells with (D, H_z, P_surv) in the v panel.
    for D, Hz, dm, ps in zip(D_arr, Hz_arr, d_mean, p_surv):
        axes[0, 0].annotate(
            (
                rf'$D={D*1e3:.2f}$' '\n'
                rf'$H_z={Hz:+.2f}$' '\n'
                rf'$P={ps:.2f}$'
            ),
            xy=(dm * 1e9, 0.0),
            xytext=(0, -3), textcoords='offset points',
            fontsize=12, ha='center', va='top', color='gray',
        )
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f'Saved {fig_path}')


# =============================================================================
if __name__ == '__main__':
    main()
