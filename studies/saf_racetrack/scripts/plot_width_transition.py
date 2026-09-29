"""Analysis of the intermediate-temperature width sweep (T6).

The square pulse is driven at the intermediate temperatures
{55,65,75,85,95} K -- which bracket the 50->100 K transition seen in the
main width study -- at two fixed currents (the 100 K cap 3e11 and its
half 1e11), over pulse durations {100..700} ps. Temperature is the only
variable across the two currents, so the plots below locate the
temperature at which widening the pulse breaks the skyrmion.

Each cell is an ensemble of 25 members on the temperature x duration
grid, read straight from the trajectory dumps (the pulse-study
convention; no aggregate step). Three views, mirroring the campaign
sections:

width_trans_stability_<box>.png      dominant stability class per cell
width_trans_config_j<J>_<box>.png    ens0 final m_z per cell
width_trans_survival_hall_<box>.png  P_surv and Hall angle per cell

No argparse; configure via the variables at the top of main().

Run with:
    python -m studies.saf_racetrack.scripts.plot_width_transition

Functions
---------
main
    Load the sweep and write the stability, configuration and
    survival/Hall figures.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
# Local
from studies.saf_racetrack.scripts.plot_pulse_ranking import _classify, \
    _decode_config

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _localized(code, metrics):
    """True if the final frame is an x-localized moving skyrmion.

    Parameters
    ----------
    code : str
        Classifier verdict.
    metrics : dict
        Classifier metrics (uses 'dx_lx').

    Returns
    -------
    keep : bool
        Compact or elongated, or a periodic-seam skyrmion with
        D_x / L_x < 0.5.
    """
    return (code in ('S', 'E')
            or (code == 'P' and metrics['dx_lx'] < 0.5))
# -------------------------------------------------------------------------


def _cell(width_dir, t_sub, peak_j, width_ps, classes):
    """Ensemble stability, survival and Hall angle of one cell.

    Parameters
    ----------
    width_dir : str
        Dataset directory for the box.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).
    width_ps : int
        Pulse duration (ps).
    classes : list[str]
        Ordered class codes, for the dominant-class index.

    Returns
    -------
    stats : {dict, None}
        'dom' (index into classes of the most common code), 'p_surv'
        (localized fraction), 'hall' (mean Hall angle over localized
        survivors, deg; NaN if none), 'n_total'. None if the cell is
        absent.
    """
    paths = sorted(glob.glob(os.path.join(
        width_dir, 'square_tp%04d_T%05.1f_j%.2e_ens*.npz'
        % (width_ps, t_sub, peak_j))))
    if not paths:
        return None
    counts = {c: 0 for c in classes}
    n_loc = 0
    halls = []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        code, metrics = _classify(npz, config)
        if code in counts:
            counts[code] += 1
        if _localized(code, metrics):
            n_loc += 1
            halls.append(float(np.asarray(npz['hall_deg']).ravel()[-1]))
    dom = int(np.argmax([counts[c] for c in classes]))
    return {
        'dom': dom, 'p_surv': n_loc / len(paths),
        'hall': float(np.mean(halls)) if halls else float('nan'),
        'n_total': len(paths),
    }
# -------------------------------------------------------------------------


def _grid(width_dir, temperatures, peak_j, widths, classes):
    """Per-cell stats on the temperature x duration grid.

    Parameters
    ----------
    width_dir : str
        Dataset directory for the box.
    temperatures : list[float]
        Substrate temperatures (K).
    peak_j : float
        Peak current density (A/m^2).
    widths : list[int]
        Pulse durations (ps).
    classes : list[str]
        Ordered class codes.

    Returns
    -------
    dom, p_surv, hall : numpy.ndarray(2d)
        Dominant-class index, survival fraction, mean Hall angle, each
        shaped (n_T, n_width); missing cells are NaN (or -1 for dom).
    """
    nt, nw = len(temperatures), len(widths)
    dom = np.full((nt, nw), -1, dtype=int)
    p_surv = np.full((nt, nw), np.nan)
    hall = np.full((nt, nw), np.nan)
    for i, t_sub in enumerate(temperatures):
        for k, width_ps in enumerate(widths):
            stats = _cell(width_dir, t_sub, peak_j, width_ps, classes)
            if stats is None:
                continue
            dom[i, k] = stats['dom']
            p_surv[i, k] = stats['p_surv']
            hall[i, k] = stats['hall']
    return dom, p_surv, hall
# -------------------------------------------------------------------------


def _plot_stability(box_tag, grids, temperatures, currents, widths,
                    classes, colors, out_path):
    """Dominant stability class on the T x duration grid per current.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    grids : dict
        peak_j -> (dom, p_surv, hall) from `_grid`.
    temperatures : list[float]
        Substrate temperatures (K).
    currents : list[float]
        Peak currents, one panel each (A/m^2).
    widths : list[int]
        Pulse durations (ps).
    classes : list[str]
        Ordered class codes.
    colors : list[str]
        Colour per class.
    out_path : str
        Destination PNG.
    """
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(-0.5, len(classes) + 0.5),
                                cmap.N)
    fig, axes = plt.subplots(1, len(currents),
                             figsize=(5.2 * len(currents), 4.8),
                             squeeze=False)
    for col, peak_j in enumerate(currents):
        ax = axes[0][col]
        dom = np.ma.masked_less(grids[peak_j][0], 0)
        ax.imshow(dom, origin='lower', cmap=cmap, norm=norm,
                  aspect='auto')
        ax.set_xticks(range(len(widths)))
        ax.set_xticklabels(widths)
        ax.set_yticks(range(len(temperatures)))
        ax.set_yticklabels(['%.0f' % t for t in temperatures])
        ax.set_xlabel(r'$t_{\mathrm{pulse}}$ (ps)')
        if col == 0:
            ax.set_ylabel(r'$T$ (K)')
        ax.set_title(r'$J = %.1f\times10^{11}$ A/m$^2$'
                     % (peak_j * 1e-11))
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i])
               for i in range(len(classes))]
    axes[0][-1].legend(handles, classes, title='class',
                       loc='center left', bbox_to_anchor=(1.02, 0.5),
                       frameon=False)
    fig.suptitle('%s -- dominant stability class (square, '
                 'intermediate T)' % box_tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_survival_hall(box_tag, grids, temperatures, currents, widths,
                        out_path):
    """Survival fraction and Hall angle on the T x duration grid.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    grids : dict
        peak_j -> (dom, p_surv, hall) from `_grid`.
    temperatures : list[float]
        Substrate temperatures (K).
    currents : list[float]
        Peak currents, one column each (A/m^2).
    widths : list[int]
        Pulse durations (ps).
    out_path : str
        Destination PNG.
    """
    hmax = np.nanmax([np.nanmax(np.abs(grids[j][2])) for j in currents])
    hmax = hmax if np.isfinite(hmax) and hmax > 0 else 1.0
    fig, axes = plt.subplots(2, len(currents),
                             figsize=(5.2 * len(currents), 8.4),
                             squeeze=False)
    for col, peak_j in enumerate(currents):
        _, p_surv, hall = grids[peak_j]
        im0 = axes[0][col].imshow(p_surv, origin='lower', cmap='viridis',
                                  vmin=0.0, vmax=1.0, aspect='auto')
        im1 = axes[1][col].imshow(hall, origin='lower', cmap='RdBu_r',
                                  vmin=-hmax, vmax=hmax, aspect='auto')
        for row, im in ((0, im0), (1, im1)):
            ax = axes[row][col]
            ax.set_xticks(range(len(widths)))
            ax.set_xticklabels(widths)
            ax.set_yticks(range(len(temperatures)))
            ax.set_yticklabels(['%.0f' % t for t in temperatures])
            ax.set_xlabel(r'$t_{\mathrm{pulse}}$ (ps)')
            if col == 0:
                ax.set_ylabel(r'$T$ (K)')
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        axes[0][col].set_title(
            r'$P_{\mathrm{surv}}$, $J=%.1f\times10^{11}$'
            % (peak_j * 1e-11))
        axes[1][col].set_title(
            r'$\theta_H$ (deg), $J=%.1f\times10^{11}$'
            % (peak_j * 1e-11))
    fig.suptitle('%s -- survival and Hall angle (square, '
                 'intermediate T)' % box_tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_config(width_dir, box_tag, temperatures, peak_j, widths,
                 out_path):
    """Final ens0 m_z on the T x duration grid for one current.

    Parameters
    ----------
    width_dir : str
        Dataset directory for the box.
    box_tag : str
        Box directory name (title only).
    temperatures : list[float]
        Substrate temperatures, one row each (K).
    peak_j : float
        Peak current density (A/m^2).
    widths : list[int]
        Pulse durations, one column each (ps).
    out_path : str
        Destination PNG.
    """
    x_gate = 0.5
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    order = sorted(range(len(temperatures)),
                   key=lambda k: temperatures[k], reverse=True)
    nt, nw = len(temperatures), len(widths)
    fig, axes = plt.subplots(nt, nw, figsize=(2.2 * nw, 2.0 * nt),
                             squeeze=False)
    for r, i in enumerate(order):
        t_sub = temperatures[i]
        for k, width_ps in enumerate(widths):
            ax = axes[r][k]
            ax.set_xticks([])
            ax.set_yticks([])
            paths = sorted(glob.glob(os.path.join(
                width_dir, 'square_tp%04d_T%05.1f_j%.2e_ens000.npz'
                % (width_ps, t_sub, peak_j))))
            if not paths:
                ax.text(0.5, 0.5, '--', transform=ax.transAxes,
                        ha='center', va='center', color='0.6')
            else:
                npz = np.load(paths[0], allow_pickle=True)
                config = _decode_config(npz)
                code, metrics = _classify(npz, config)
                nx, ny = int(config['nx']), int(config['ny'])
                mz = np.asarray(npz['mz_final_top'],
                                dtype=float).reshape(ny, nx)
                a_nm = float(npz['L_x']) / nx * 1e9
                ax.imshow(mz, origin='lower', cmap='RdBu_r', norm=norm,
                          extent=[0.0, nx * a_nm, 0.0, ny * a_nm],
                          aspect='equal')
                if metrics['dx_lx'] >= x_gate:
                    code = code + 'p'
                ax.text(0.03, 0.93, code, transform=ax.transAxes,
                        ha='left', va='top', fontsize=11,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=1.2))
            if k == 0:
                ax.set_ylabel('%.0f' % t_sub, fontsize=12, rotation=0,
                              labelpad=16, va='center', ha='right')
            if r == nt - 1:
                ax.set_xlabel('%d' % width_ps, fontsize=11)
    fig.supxlabel(r'$t_{\mathrm{pulse}}$ (ps)', fontsize=14)
    fig.supylabel(r'$T$ (K)', fontsize=14)
    fig.suptitle(r'square, $J=%.1f\times10^{11}$ A/m$^2$: '
                 r'final configuration' % (peak_j * 1e-11), fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.08,
                        hspace=0.3, wspace=0.06)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the sweep and write the T6 analysis figures."""
    plt.rcParams.update({
        'axes.labelsize': 14, 'axes.titlesize': 13,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'figure.titlesize': 15})
    # =========================== User Configuration =========================
    root_base = ('output/'
                 'sweeps_driving_T/pulse_shape/width_transition')
    box_tag = 'hk36_D0p72_700x500'
    out_dir = 'output/figures_driving_T/width_transition'
    temperatures = [55.0, 65.0, 75.0, 85.0, 95.0]
    currents = [1.0e11, 3.0e11]
    widths = [100, 200, 300, 400, 600, 700]
    classes = ['S', 'E', 'P', 'L', 'A', 'R']
    colors = ['#1a9850', '#a6d96a', '#fdae61', '#d73027',
              '#999999', '#762a83']
    # ======================= End User Configuration =========================
    width_dir = os.path.join(root_base, box_tag)
    if not os.path.isdir(width_dir):
        raise RuntimeError('main: dataset not found: %r.' % width_dir)
    os.makedirs(out_dir, exist_ok=True)
    print('loading %s ...' % box_tag)
    grids = {j: _grid(width_dir, temperatures, j, widths, classes)
             for j in currents}
    _plot_stability(
        box_tag, grids, temperatures, currents, widths, classes, colors,
        os.path.join(out_dir, 'width_trans_stability_%s.png' % box_tag))
    _plot_survival_hall(
        box_tag, grids, temperatures, currents, widths,
        os.path.join(out_dir,
                     'width_trans_survival_hall_%s.png' % box_tag))
    for peak_j in currents:
        _plot_config(
            width_dir, box_tag, temperatures, peak_j, widths,
            os.path.join(out_dir, 'width_trans_config_j%.1f_%s.png'
                         % (peak_j * 1e-11, box_tag)))


# =============================================================================
if __name__ == '__main__':
    main()
