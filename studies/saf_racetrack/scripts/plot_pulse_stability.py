"""Per-shape stability maps and class composition of the pulse study.

For each of the six drive shapes, classifies every realization of the
pulse-shape production over the (T, J) grid and renders:

pulse_stability_maps_<box>.png
    One panel per shape, the dominant final class in each (T, J) cell
    (compact S, elongated E, spanning P, labyrinth L, annihilated A,
    reversed R), annotated with the surviving-skyrmion fraction
    P_surv = (S + E + P) / n. Cells absent from production (the shape
    exceeded its stability cap there) are left blank.
pulse_composition_<box>.png
    A grid of per-class ensemble-fraction heatmaps, rows the shapes and
    columns the classes S, E, P, L, so the loss channel of each shape
    can be read directly.

No argparse; configure via the variables at the top of main().

Run with:
    python -m studies.saf_racetrack.scripts.plot_pulse_stability

Functions
---------
main
    Classify the production once and write the two figures per box.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
from collections import Counter
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
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


def _cell_counts(prod_dir, shape, t_sub, peak_j):
    """Class counts of one (shape, T, J) cell.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    counts : collections.Counter
        Class code -> member count (empty if the cell is absent).
    n_total : int
        Number of members found.
    """
    paths = sorted(glob.glob(os.path.join(
        prod_dir, '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))))
    counts = Counter()
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        code, _ = _classify(npz, _decode_config(npz))
        counts[code] += 1
    return counts, len(paths)
# -------------------------------------------------------------------------


def _currents_union(prod_dir, shapes, temperatures):
    """All peak currents present for any (shape, T), ascending.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    currents : list[float]
        Peak current densities (A/m^2), sorted ascending.
    """
    currents = set()
    for shape in shapes:
        for t_sub in temperatures:
            pattern = os.path.join(
                prod_dir, '%s_T%05.1f_j*_ens000.npz' % (shape, t_sub))
            for path in glob.glob(pattern):
                tag = os.path.basename(path).split('_j')[1].split(
                    '_ens')[0]
                currents.add(float(tag))
    return sorted(currents)
# -------------------------------------------------------------------------


def _build_grids(prod_dir, shapes, temperatures, currents, classes):
    """Dominant class, survival and per-class fraction grids.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K), the grid rows.
    currents : list[float]
        Peak currents (A/m^2), the grid columns.
    classes : list[str]
        Class codes in display order.

    Returns
    -------
    grids : dict
        Per shape: 'dominant' (int grid, -1 = no data), 'surv'
        (float grid, NaN = no data), 'frac' (dict class -> grid).
    """
    grids = {}
    surv_set = ('S', 'E', 'P')
    for shape in shapes:
        n_t, n_j = len(temperatures), len(currents)
        dominant = np.full((n_t, n_j), -1, dtype=int)
        surv = np.full((n_t, n_j), np.nan)
        frac = {c: np.full((n_t, n_j), np.nan) for c in classes}
        for i, t_sub in enumerate(temperatures):
            for k, peak_j in enumerate(currents):
                counts, n_total = _cell_counts(
                    prod_dir, shape, t_sub, peak_j)
                if n_total == 0:
                    continue
                best = max(classes, key=lambda c: counts.get(c, 0))
                dominant[i, k] = classes.index(best)
                surv[i, k] = (sum(counts.get(c, 0) for c in surv_set)
                              / n_total)
                for c in classes:
                    frac[c][i, k] = counts.get(c, 0) / n_total
        grids[shape] = {'dominant': dominant, 'surv': surv,
                        'frac': frac}
    return grids
# -------------------------------------------------------------------------


def _class_cmap(classes, colors):
    """Discrete colormap and norm over the class indices.

    Parameters
    ----------
    classes : list[str]
        Class codes.
    colors : list
        One colour per class, in the same order.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Colormap over 0..len(classes)-1.
    norm : matplotlib.colors.BoundaryNorm
        Matching boundaries.
    """
    cmap = mcolors.ListedColormap(colors)
    cmap.set_bad('white')
    bounds = np.arange(-0.5, len(classes) + 0.5, 1.0)
    return cmap, mcolors.BoundaryNorm(bounds, cmap.N)
# -------------------------------------------------------------------------


def _ticks(ax, temperatures, currents):
    """Label a (T, J) grid axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    temperatures : list[float]
        Row values (K).
    currents : list[float]
        Column values (A/m^2).
    """
    ax.set_xticks(range(len(currents)))
    ax.set_xticklabels(['%.1f' % (j / 1e11) for j in currents],
                       fontsize=8)
    ax.set_yticks(range(len(temperatures)))
    ax.set_yticklabels(['%g' % t for t in temperatures])
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$T$ (K)')
# -------------------------------------------------------------------------


def _plot_stability_maps(box_tag, grids, shapes, temperatures,
                         currents, classes, colors, out_path):
    """Dominant-class map per shape, with P_surv annotation.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    grids : dict
        Output of `_build_grids`.
    shapes : list[str]
        Shape names, one panel each.
    temperatures : list[float]
        Grid rows (K).
    currents : list[float]
        Grid columns (A/m^2).
    classes : list[str]
        Class codes.
    colors : list
        Per-class colours.
    out_path : str
        Destination PNG.
    """
    cmap, norm = _class_cmap(classes, colors)
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), squeeze=False)
    for idx, shape in enumerate(shapes):
        ax = axes[idx // 3][idx % 3]
        dom = np.ma.masked_less(grids[shape]['dominant'], 0)
        ax.imshow(dom, origin='lower', aspect='auto', cmap=cmap,
                  norm=norm)
        surv = grids[shape]['surv']
        for i in range(len(temperatures)):
            for k in range(len(currents)):
                if np.isfinite(surv[i, k]):
                    code = classes[grids[shape]['dominant'][i, k]]
                    col = 'w' if code in ('L', 'R') else 'k'
                    ax.text(k, i, '%.2f' % surv[i, k], ha='center',
                            va='center', fontsize=7, color=col)
        ax.set_title(shape)
        _ticks(ax, temperatures, currents)
    handles = [mpatches.Patch(color=colors[c], label=classes[c])
               for c in range(len(classes))]
    fig.legend(handles=handles, loc='lower center', ncol=len(classes),
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('%s -- dominant class and P_surv' % box_tag,
                 fontsize=16)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_composition(box_tag, grids, shapes, temperatures, currents,
                      comp_classes, out_path):
    """Per-class ensemble-fraction heatmaps, rows shapes, cols classes.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    grids : dict
        Output of `_build_grids`.
    shapes : list[str]
        Shape names, one row each.
    temperatures : list[float]
        Grid rows (K).
    currents : list[float]
        Grid columns (A/m^2).
    comp_classes : list[str]
        Classes to show, one column each.
    out_path : str
        Destination PNG.
    """
    n_r, n_c = len(shapes), len(comp_classes)
    fig, axes = plt.subplots(
        n_r, n_c, figsize=(3.2 * n_c, 2.6 * n_r), squeeze=False)
    im = None
    for r, shape in enumerate(shapes):
        for c, code in enumerate(comp_classes):
            ax = axes[r][c]
            z = np.ma.masked_invalid(grids[shape]['frac'][code])
            im = ax.imshow(z, origin='lower', aspect='auto',
                           cmap='viridis', vmin=0.0, vmax=1.0)
            if r == 0:
                ax.set_title('class %s' % code)
            if c == 0:
                ax.set_ylabel('%s\n$T$ (K)' % shape, fontsize=9)
                ax.set_yticks(range(len(temperatures)))
                ax.set_yticklabels(['%g' % t for t in temperatures],
                                   fontsize=8)
            else:
                ax.set_yticks([])
            if r == n_r - 1:
                ax.set_xticks(range(len(currents)))
                ax.set_xticklabels(
                    ['%.1f' % (j / 1e11) for j in currents],
                    fontsize=7)
                ax.set_xlabel(r'$J$ ($10^{11}$)', fontsize=9)
            else:
                ax.set_xticks([])
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01,
                 label='ensemble fraction')
    fig.suptitle('%s -- class composition' % box_tag, fontsize=16)
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Classify the production once and write the two figures per box."""
    plt.rcParams.update({
        'axes.labelsize': 12,
        'axes.titlesize': 13,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
    })
    # =========================== User Configuration =========================
    prod_root = ('output/'
                 'sweeps_driving_T/pulse_shape/production')
    box_tags = ['hk36_D0p72_350x500', 'hk36_D0p72_700x500',
                'hk36_D0p72_1400x500']
    out_dir = 'output/figures_driving_T/stability'
    shapes = ['square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
              'tri_symmetric', 'gaussian']
    temperatures = [0.0, 10.0, 50.0, 100.0]
    classes = ['S', 'E', 'P', 'L', 'A', 'R']
    colors = ['#1a9850', '#a6d96a', '#fdae61', '#d73027', '#999999',
              '#762a83']
    comp_classes = ['S', 'E', 'P', 'L']
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    for box_tag in box_tags:
        prod_dir = os.path.join(prod_root, box_tag)
        if not os.path.isdir(prod_dir):
            raise RuntimeError(
                'main: production directory not found: %r.' % prod_dir)
        print('classifying %s ...' % box_tag)
        currents = _currents_union(prod_dir, shapes, temperatures)
        grids = _build_grids(prod_dir, shapes, temperatures, currents,
                             classes)
        _plot_stability_maps(
            box_tag, grids, shapes, temperatures, currents, classes,
            colors,
            os.path.join(out_dir,
                         'pulse_stability_maps_%s.png' % box_tag))
        _plot_composition(
            box_tag, grids, shapes, temperatures, currents,
            comp_classes,
            os.path.join(out_dir,
                         'pulse_composition_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
