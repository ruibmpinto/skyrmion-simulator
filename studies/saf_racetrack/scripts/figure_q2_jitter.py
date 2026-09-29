"""Q2 directional-control figures: sigma_y vs j at T=300 K, and
a schematic of the skyrmion jitter envelope at different
temperatures at fixed j.

Reads the scan_tj aggregate.npz directly (mean), plus the per-
trajectory NPZs (for SE bars on the first panel). Writes:

    output/figures_sllg/q2_jitter_vs_j.png
    output/figures_sllg/q2_jitter_circles.png

Run with:
    python -m studies.saf_racetrack.scripts.figure_q2_jitter
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


def _load_cell(in_dir, T, j):
    """Return list of per-trajectory NPZs for the (T, j) cell."""
    pat = os.path.join(
        in_dir, f'T{T:05.1f}_j{j:.2e}_ens*.npz')
    return sorted(glob.glob(pat))


def main():
    """Load the scan_tj aggregate and per-trajectory NPZs and write
    the sigma_y-vs-j jitter figure and the jitter-envelope
    schematics."""
    # =========================== User Configuration =========================
    in_dir   = 'output/stochastic_llgs/scan_tj'
    fig_dir  = 'output/figures_sllg'
    T_target = 300.0
    j_list   = [1.0e11, 2.0e11, 4.0e11, 8.0e11, 1.6e12]
    # Circle-band schematic: same Pareto-optimal drive,
    # vary T to show how the jitter envelope grows.
    j_circles = 4.0e11
    # Skyrmion radius (m -> nm) and box geometry.
    R_sk_nm = 93.25
    nx = 256
    a_m = 2.0e-9
    L_nm = nx * a_m * 1e9
    out_jitter      = os.path.join(
        fig_dir, 'q2_jitter_vs_j.png')
    out_circles     = os.path.join(
        fig_dir, 'q2_jitter_circles.png')
    out_circles_j   = os.path.join(
        fig_dir, 'q2_jitter_circles_vs_j.png')
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 1: sigma_y vs j at T = T_target, with ensemble error bars.
    j_arr = np.array(j_list, dtype=float)
    sy_mean = np.full(j_arr.size, np.nan)
    sy_se = np.full(j_arr.size, np.nan)
    n_used = np.zeros(j_arr.size, dtype=int)
    for k, j in enumerate(j_arr):
        files = _load_cell(in_dir, T_target, j)
        sy_list = []
        for fp in files:
            d = np.load(fp, allow_pickle=True)
            if not bool(d['alive_at_end']):
                continue
            sy = float(d['sigma_y'])
            if np.isfinite(sy):
                sy_list.append(sy)
        if sy_list:
            arr = np.array(sy_list, dtype=float)
            sy_mean[k] = float(arr.mean())
            n_used[k] = arr.size
            if arr.size > 1:
                sy_se[k] = float(
                    arr.std(ddof=1) / math.sqrt(arr.size))
        print(
            f'  j={j:.2e}  n_alive={n_used[k]:2d}  '
            f'<sigma_y>={sy_mean[k]*1e9:6.1f} nm  '
            f'SE={sy_se[k]*1e9 if np.isfinite(sy_se[k]) else 0:.1f}')
    # Plot.
    fig1, ax1 = plt.subplots()
    valid = np.isfinite(sy_mean)
    ax1.errorbar(
        j_arr[valid], sy_mean[valid] * 1e9,
        yerr=np.where(np.isfinite(sy_se[valid]),
                      sy_se[valid] * 1e9, 0.0),
        fmt='o', capsize=4, markersize=7,
        color='C0',
    )
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel(r'$j$ (A/m$^2$)')
    ax1.set_ylabel(r'$\sigma_y$ (nm)')
    ax1.set_title(
        rf'Transverse jitter at $T_\mathrm{{sub}}={T_target:.0f}$ K')
    # Reference lines: box half-width and inter-skyrmion separation.
    ax1.axhline(L_nm / 2, color='gray', linestyle='--',
                label=r'box half-width (256 nm)')
    ax1.axhline(200.0, color='k', linestyle=':',
                label=r'inter-skyrmion sep. (200 nm)')
    # Annotate each point with n_alive.
    for k, j in enumerate(j_arr):
        if np.isfinite(sy_mean[k]):
            ax1.annotate(
                rf'$n={n_used[k]}$',
                xy=(j, sy_mean[k] * 1e9),
                xytext=(7, 5), textcoords='offset points',
                fontsize=12, color='gray',
            )
    ax1.legend(loc='best', frameon=False)
    ax1.set_box_aspect(1)
    ax1.set_ylim([None, 600])
    fig1.tight_layout()
    fig1.savefig(out_jitter)
    print(f'Saved {out_jitter}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 2: schematic skyrmion box (no background fill) with
    # the nominal skyrmion outline at the center and concentric
    # 1-sigma jitter circles per temperature at fixed j.
    # Read sigma_y(T) at j_circles from the aggregate.
    agg = np.load(os.path.join(in_dir, 'aggregate.npz'))
    Ts_all = agg['Ts']
    Js_all = agg['Js']
    sy_grid = agg['sigma_y_mean']
    j_idx = int(np.argmin(np.abs(Js_all - j_circles)))
    sy_T_m = sy_grid[:, j_idx]
    # Keep only T values with finite sigma_y (surviving cells).
    mask = np.isfinite(sy_T_m)
    Ts_keep = Ts_all[mask]
    sy_keep_nm = sy_T_m[mask] * 1e9
    print(
        f'Circle panel: j={j_circles:.2e}, '
        f'using {Ts_keep.size} T values '
        f'({Ts_keep[0]:.0f}-{Ts_keep[-1]:.0f} K).')
    # Plot: square box, skyrmion at center, concentric sigma_y
    # circles colored by T (viridis low->high).
    fig2, ax2 = plt.subplots()
    cmap = plt.get_cmap('viridis')
    norm = plt.Normalize(vmin=float(Ts_keep.min()),
                         vmax=float(Ts_keep.max()))
    # Box centered on origin to make the circles symmetric.
    half = L_nm / 2.0
    ax2.set_xlim(-half, half)
    ax2.set_ylim(-half, half)
    # Box outline (no background fill).
    ax2.add_patch(plt.Rectangle(
        (-half, -half), L_nm, L_nm,
        fill=False, edgecolor='black'))
    # Nominal skyrmion at origin: dashed circle.
    ax2.add_patch(plt.Circle(
        (0.0, 0.0), R_sk_nm,
        fill=False, edgecolor='black', linestyle='--'))
    # ax2.text(
    #     0.0, 0.0, 'skyrmion',
    #     ha='center', va='center',
    #     fontsize=12, color='black')
    # 1-sigma jitter circles, one per T, plotted with the
    # skyrmion radius added so the circle marks where the
    # core boundary can wander to under 1 sigma_y of jitter.
    for T_K, sy_nm in zip(Ts_keep, sy_keep_nm):
        ax2.add_patch(plt.Circle(
            (0.0, 0.0), R_sk_nm + sy_nm,
            fill=False, edgecolor=cmap(norm(T_K))))
    # Strip axes; only the colorbar label remains.
    ax2.set_xticks([])
    ax2.set_yticks([])
    for spine in ax2.spines.values():
        spine.set_visible(False)
    # Colorbar mapping T -> color.
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cb = fig2.colorbar(sm, ax=ax2, fraction=0.046,
                       pad=0.04)
    cb.set_label(r'$T_\mathrm{sub}$ (K)')
    cb.set_ticks(list(Ts_keep))
    cb.set_ticklabels([f'{int(T)}' for T in Ts_keep])
    # ax2.set_xlabel(r'$x$ (nm)')
    # ax2.set_ylabel(r'$y$ (nm)')
    # ax2.set_title(
    #     rf'Jitter envelope: $j={j_circles:.1e}$ A/m$^2$, '
    #     rf'$R_\mathrm{{sk}}+\sigma_y(T)$')
    ax2.set_box_aspect(1)
    fig2.tight_layout()
    fig2.savefig(out_circles)
    print(f'Saved {out_circles}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 3: same envelope at fixed T = T_target, varying j.
    T_idx = int(np.argmin(np.abs(Ts_all - T_target)))
    sy_j_m = sy_grid[T_idx, :]
    j_mask = np.isfinite(sy_j_m)
    Js_keep = Js_all[j_mask]
    sy_j_nm = sy_j_m[j_mask] * 1e9
    print(
        f'Circle-vs-j panel: T={T_target:.0f} K, '
        f'using {Js_keep.size} j values '
        f'({Js_keep[0]:.2e}-{Js_keep[-1]:.2e} A/m^2).')
    fig3, ax3 = plt.subplots()
    cmap_j = plt.get_cmap('plasma')
    # Log-normalise the current colormap so the spread visually
    # tracks log_10(j) rather than linear j (the operating range
    # spans more than one decade).
    norm_j = plt.matplotlib.colors.LogNorm(
        vmin=float(Js_keep.min()),
        vmax=float(Js_keep.max()))
    ax3.set_xlim(-half, half)
    ax3.set_ylim(-half, half)
    ax3.add_patch(plt.Rectangle(
        (-half, -half), L_nm, L_nm,
        fill=False, edgecolor='black'))
    ax3.add_patch(plt.Circle(
        (0.0, 0.0), R_sk_nm,
        fill=False, edgecolor='black', linestyle='--'))
    # ax3.text(
    #     0.0, 0.0, 'skyrmion',
    #     ha='center', va='center',
    #     fontsize=12, color='black')
    for j_val, sy_nm in zip(Js_keep, sy_j_nm):
        ax3.add_patch(plt.Circle(
            (0.0, 0.0), R_sk_nm + sy_nm,
            fill=False, edgecolor=cmap_j(norm_j(j_val))))
    # Strip axes; only the colorbar label remains.
    ax3.set_xticks([])
    ax3.set_yticks([])
    for spine in ax3.spines.values():
        spine.set_visible(False)
    sm_j = plt.cm.ScalarMappable(cmap=cmap_j, norm=norm_j)
    sm_j.set_array([])
    cb3 = fig3.colorbar(sm_j, ax=ax3, fraction=0.046,
                        pad=0.04)
    cb3.set_label(r'$j$ (A/m$^2$)')
    # ax3.set_xlabel(r'$x$ (nm)')
    # ax3.set_ylabel(r'$y$ (nm)')
    # ax3.set_title(
    #     rf'Jitter envelope: $T_\mathrm{{sub}}'
    #     rf'={T_target:.0f}$ K, '
    #     rf'$R_\mathrm{{sk}}+\sigma_y(j)$')
    ax3.set_box_aspect(1)
    fig3.tight_layout()
    fig3.savefig(out_circles_j)
    print(f'Saved {out_circles_j}')


# =============================================================================
if __name__ == '__main__':
    main()
