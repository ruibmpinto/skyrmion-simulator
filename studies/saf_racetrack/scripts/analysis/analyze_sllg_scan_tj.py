"""(T_sub, j) analysis: drift velocity, Hall angle, and
survival across the temperature--current grid (Q1, Q2,
Q4 driven branch).

Consumes
    output/stochastic_llgs/scan_tj/aggregate.npz
(written by
`studies.saf_racetrack.scripts.analysis.aggregate_sllg scan_tj`). Produces four
heatmaps and a one-line Pareto front (max <v> subject to
P_surv >= threshold).

Run with:
    python -m studies.saf_racetrack.scripts.analysis.analyze_sllg_scan_tj
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
plt.rcParams['figure.figsize'] = (12, 12)
plt.rcParams['lines.linewidth'] = 1.5


def main():
    """Load the (T_sub, j) aggregate, write the four heatmaps and the
    survival-constrained Pareto-front figure, and save the Pareto
    NPZ."""
    # =========================== User Configuration =========================
    in_npz      = 'output/stochastic_llgs/scan_tj/aggregate.npz'
    fig_dir     = 'output/figures_sllg'
    fig_path    = os.path.join(fig_dir, 'scan_tj.png')
    pareto_npz  = 'output/stochastic_llgs/scan_tj/pareto.npz'
    # Survival threshold a cell must clear to enter the Pareto front.
    p_surv_min  = 0.8
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.isfile(in_npz):
        raise RuntimeError(
            f'{in_npz!r} not found. Run '
            f'`python -m '
            f'studies.saf_racetrack.scripts.analysis.aggregate_sllg '
            f'scan_tj` '
            f'first.'
        )
    d = np.load(in_npz)
    Ts = d['Ts']
    Js = d['Js']
    P_surv = d['P_surv']
    v_mean = d['v_mean']
    th_mean = d['theta_mean']
    sy_mean = d['sigma_y_mean']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pareto front: at each T_sub, the j-cell with max <v>
    # subject to P_surv >= threshold.
    pareto_T = []
    pareto_j = []
    pareto_v = []
    for i, T_sub in enumerate(Ts):
        mask = P_surv[i, :] >= p_surv_min
        if not np.any(mask):
            continue
        v_row = v_mean[i, :].copy()
        v_row[~mask] = -np.inf
        k_star = int(np.nanargmax(v_row))
        if not np.isfinite(v_row[k_star]):
            continue
        pareto_T.append(float(T_sub))
        pareto_j.append(float(Js[k_star]))
        pareto_v.append(float(v_row[k_star]))
    pareto_T = np.array(pareto_T, dtype=float)
    pareto_j = np.array(pareto_j, dtype=float)
    pareto_v = np.array(pareto_v, dtype=float)
    print(
        f'Pareto front (max <v> at P_surv >= '
        f'{p_surv_min:.2f}):'
    )
    for T_sub, j, v in zip(pareto_T, pareto_j, pareto_v):
        print(
            f'  T_sub={T_sub:5.1f} K  j={j:.2e} A/m^2  '
            f'<v>={v:7.1f} m/s'
        )
    np.savez_compressed(
        pareto_npz,
        pareto_T=pareto_T, pareto_j=pareto_j,
        pareto_v=pareto_v, p_surv_min=float(p_surv_min),
    )
    print(f'Saved {pareto_npz}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Four heatmaps over the (T_sub, log j) grid; the Pareto
    # front is overlaid on the velocity panel below.
    fig, axes = plt.subplots(2, 2)
    panels = [
        (axes[0, 0], P_surv,
         r'$P_\mathrm{surv}$', 'viridis'),
        (axes[0, 1], v_mean,
         r'$\langle v \rangle$ (m/s)', 'plasma'),
        (axes[1, 0], th_mean,
         r'$\langle \theta_H \rangle$ (deg)', 'coolwarm'),
        (axes[1, 1], sy_mean * 1e9,
         r'$\sigma_y$ (nm)', 'magma'),
    ]
    # Half-step padding so imshow pixels are centered on the
    # grid values (T linear, j logarithmic on the x-axis).
    dT = (Ts[1] - Ts[0]) if Ts.size >= 2 else 1.0
    dlj = (
        (math.log10(Js[-1]) - math.log10(Js[0]))
        / max(1, Js.size - 1)
    )
    extent = (
        math.log10(Js[0]) - 0.5 * dlj,
        math.log10(Js[-1]) + 0.5 * dlj,
        Ts[0] - 0.5 * dT, Ts[-1] + 0.5 * dT,
    )
    for ax, data, title, cmap in panels:
        im = ax.imshow(
            data, origin='lower', aspect='auto',
            cmap=cmap, extent=extent,
        )
        ax.set_xlabel(r'$\log_{10}(j)$ (A/m$^2$)')
        ax.set_ylabel(r'$T_\mathrm{sub}$ (K)')
        ax.set_title(title)
        ax.set_box_aspect(1)
        fig.colorbar(im, ax=ax)
        # Overlay Pareto front on the velocity panel.
        if r'\langle v \rangle' in title and pareto_T.size:
            ax.plot(
                np.log10(pareto_j), pareto_T,
                'o-', color='cyan', markersize=6,
                label=(
                    r'Pareto ($P_\mathrm{surv} \geq '
                    rf'{p_surv_min:.2f}$)'
                ),
            )
            ax.legend(loc='best', frameon=False)
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f'Saved {fig_path}')


# =============================================================================
if __name__ == '__main__':
    main()
