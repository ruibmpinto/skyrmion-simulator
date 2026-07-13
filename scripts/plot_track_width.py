"""Plots for the skyrmion length-scale vs track-width scan.

Reads the `aggregate.npz` written by
`scripts/aggregate_sllg.py scan_track_width` and the per-cell
`anim_*.npz` full-field dumps written by the production driver,
and renders:

- Elliptical axes D_1, D_2 versus current density (one curve per
  temperature) and versus temperature (one curve per current).
- Heatmaps of the length-scale ratios D_1 / L_x and D_2 / L_y
  over the (T, j) grid (L_y is the track width).
- Survival probability and skyrmion-Hall-angle maps marking the
  burst / annihilation boundary.
- An m_z animation of one designated realization, plus a
  three-panel initial / relaxed / final still.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m scripts.plot_track_width
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull
import matplotlib.animation as manim
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Local
from src.plots.plot_snapshot_mz import plot_snapshot_mz

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _load_aggregate(in_dir, agg_name):
    """Load the aggregate NPZ as a plain dict of arrays.

    Parameters
    ----------
    in_dir : str
        Directory holding the aggregate file.
    agg_name : str
        Aggregate file name (box/BC-tagged, e.g.
        'aggregate_350x500_pbcx_pbcy.npz').
    """
    path = os.path.join(in_dir, agg_name)
    if not os.path.isfile(path):
        raise RuntimeError(
            f'plot_track_width: aggregate not found at {path!r}; '
            f'run `aggregate_sllg scan_track_width` first.')
    d = np.load(path, allow_pickle=True)
    return {k: d[k] for k in d.files}


# -----------------------------------------------------------------------------
def _plot_axes_vs_j(agg, out_path):
    """D_1, D_2 (nm) versus current density, one curve per T."""
    Ts = agg['Ts']
    Js = agg['Js']
    D1 = agg['D1_mean'] * 1e9
    D2 = agg['D2_mean'] * 1e9
    D1e = agg['D1_se'] * 1e9
    D2e = agg['D2_se'] * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for i, T in enumerate(Ts):
        label = f'{T:.0f} K'
        axes[0].errorbar(
            Js * 1e-11, D1[i, :], yerr=D1e[i, :],
            marker='o', capsize=3, label=label)
        axes[1].errorbar(
            Js * 1e-11, D2[i, :], yerr=D2e[i, :],
            marker='s', capsize=3, label=label)
    axes[0].set_title(r'major axis $D_1$')
    axes[1].set_title(r'minor axis $D_2$')
    for ax in axes:
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylabel(r'diameter (nm)')
        ax.legend(title='T')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _periodic_x_components(mask, min_cells):
    """Connected components of a Boolean field with periodic x, free y.

    Labels the 4-connected components of `mask`, merges labels that touch
    across the periodic seam (column 0 <-> column nx-1) via a union-find,
    and DROPS components smaller than `min_cells` -- at finite T the
    reversed-domain mask is peppered with single-cell thermal speckles
    that would otherwise be counted as domains. Returns the retained
    component count, the largest-component area fraction, whether any
    retained component wraps the x seam (connects to its own periodic
    image -> spanning stripe), and the total retained area (cells).

    Parameters
    ----------
    mask : numpy.ndarray(2d, bool)
        Reversed-domain mask, shape (ny, nx). x is the periodic axis.
    min_cells : int
        Minimum component size (cells) to count as a real domain.

    Returns
    -------
    n_comp : int
        Number of retained components after the seam merge + size filter.
    f_max : float
        Largest retained component cell count / total cells.
    x_percolates : bool
        True if a retained component spans the periodic-x seam.
    area : int
        Total retained (domain) area in cells.
    dx_cells : int
        Periodic-aware x-extent (columns) of the largest retained
        component -- the span along the periodic axis, used for the
        loss-of-periodicity gate (this is D_x, not the ellipse major
        axis D_1 which may lie along free-y).
    solidity : float
        Area / convex-hull-area of the largest retained component
        (periodic-unwrapped in x). Near 1 for a convex blob or stripe
        (skyrmion); low for a serpentine, hull-underfilling labyrinth.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return 0, 0.0, False, 0, 0, 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Union-find over labels; merge across the periodic-x seam.
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    left = lab[:, 0]
    right = lab[:, -1]
    for a, b in zip(left, right):
        if a > 0 and b > 0:
            union(int(a), int(b))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Merged sizes; drop speckles below min_cells; seam-wrap test on the
    # retained (large) components only.
    roots = np.array([find(i) for i in range(n + 1)])
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    merged = {}
    for label in range(1, n + 1):
        r = roots[label]
        merged[r] = merged.get(r, 0) + int(sizes[label])
    big = {r: s for r, s in merged.items() if s >= min_cells}
    if not big:
        return 0, 0.0, False, 0, 0, 1.0
    n_comp = len(big)
    area = int(sum(big.values()))
    f_max = max(big.values()) / float(mask.size)
    seam = ({roots[int(a)] for a in left if a > 0}
            & {roots[int(b)] for b in right if b > 0})
    x_percolates = any(r in big for r in seam)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Largest retained component; its periodic x-extent (D_x) = nx minus
    # the largest cyclic gap of empty columns, and its solidity on the
    # x-unwrapped mask (rolled so the gap sits at the seam).
    r_max = max(big, key=big.get)
    comp = (roots[lab] == r_max)
    occ = np.any(comp, axis=0)
    nx = mask.shape[1]
    if occ.all():
        dx_cells = nx
        comp_uw = comp
    else:
        idx = np.flatnonzero(occ)
        gaps = np.diff(idx) - 1
        wrap_gap = idx[0] + nx - idx[-1] - 1
        max_gap = int(max(gaps.max() if gaps.size else 0, wrap_gap))
        dx_cells = nx - max_gap
        if gaps.size and gaps.max() >= wrap_gap:
            g = int(gaps.argmax())
            comp_uw = np.roll(comp, nx - int(idx[g + 1]), axis=1)
        else:
            comp_uw = comp
    pts = np.column_stack(np.nonzero(comp_uw))
    try:
        solidity = comp_uw.sum() / float(ConvexHull(pts).volume)
    except Exception:
        solidity = 1.0
    return n_comp, f_max, x_percolates, area, dx_cells, solidity


def _classify_field(mz, q_abs, D1, D2, L_x):
    """Classify one relaxed/driven configuration from the raw field.

    Combines the three physical signals: the topological charge |Q|
    (skyrmion present vs collapsed vs multi-domain), the periodic-x
    connected-component / percolation structure of the reversed domain
    (single vs fragmented vs spanning), and the D_1 / L_x periodic gate
    (a domain longer than half the track can bridge its own x-image, so
    the single-skyrmion ellipse is invalid).

    The loss-of-periodicity gate uses D_x, the field-measured x-extent
    of the reversed domain (periodic axis), NOT the ellipse major axis
    D_1 -- a skyrmion elongated along free-y has large D_1 but small
    D_x and does not bridge its periodic-x image.

    Parameters
    ----------
    mz : numpy.ndarray(2d)
        Top-layer m_z of the final drive frame, shape (ny, nx).
    q_abs : float
        |Q| (absolute topological charge) of that frame.
    D1, D2 : float
        LCC ellipse major / minor axes (m), ensemble mean for the cell
        (used only for the D_1/D_2 elongation ratio).
    L_x : float
        Track length along the periodic axis (m).

    Returns
    -------
    code : str
        'S' compact skyrmion, 'E' elongated / spanning skyrmion,
        'L' labyrinth / multi-domain, 'A' annihilated (ferromagnetic).
    metrics : dict
        Diagnostic values used by the decision (for the report table).
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Field-analysis thresholds (physical; local, not global state).
    core_thresh = -0.5  # m_z below this = reversed domain (not T noise)
    min_frac = 0.002    # component size below this frac of box = speckle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reversed domain, de-speckled: threshold hard (m_z < -0.5) and drop
    # components below min_frac of the box, so thermal fluctuations near
    # m_z = 0 are not counted as domains.
    core = mz < core_thresh
    min_cells = max(20, int(min_frac * mz.size))
    n_comp, f_max, x_perc, area, dx_cells, solidity = \
        _periodic_x_components(core, min_cells)
    core_frac = area / float(mz.size)
    # Loss-of-periodicity uses the field x-extent D_x (= dx_cells / nx),
    # not the ellipse major axis. D1/L_x is kept only for reference.
    dx_lx = dx_cells / float(mz.shape[1])
    d1_lx = D1 / L_x if (L_x > 0.0 and np.isfinite(D1)) else 0.0
    ratio = D1 / D2 if (D2 > 0.0 and np.isfinite(D1)
                        and np.isfinite(D2)) else 1.0
    metrics = {
        'q_abs': q_abs, 'core_frac': core_frac, 'n_comp': n_comp,
        'f_max': f_max, 'x_perc': bool(x_perc), 'dx_lx': dx_lx,
        'd1_lx': d1_lx, 'ratio': ratio, 'solidity': solidity,
    }
    return _decide_class(metrics), metrics


def _decide_class(m):
    """Decide the stability class from the per-cell metrics.

    Topology first, then shape. A single-winding reversed domain
    ($\\Q\\approx1$) is a skyrmion (compact or elongated) only if it is
    convex-like (high solidity); a serpentine, hull-underfilling domain
    is a labyrinth however its charge integrates. Neither the reversed
    fraction nor a lone satellite domain demotes a genuine skyrmion.

    Parameters
    ----------
    m : dict
        Metrics from `_classify_field`: q_abs, n_comp, core_frac,
        x_perc, dx_lx, ratio, solidity.

    Returns
    -------
    code : str
        'S', 'E', 'L', or 'A'.
    """
    q_sk = 0.5          # |Q| above -> a topological skyrmion is present
    q_multi = 1.6       # |Q| above -> more than one skyrmion
    core_min = 0.005    # domain-area frac below -> no domain (FM)
    x_gate = 0.5        # D_x/L_x above -> periodic-image break (index p)
    r_elong = 1.7       # D1/D2 above -> elongated
    n_dom = 3           # this many domains -> labyrinth (multi-domain)
    s_min = 0.7         # solidity below -> serpentine labyrinth
    q = m['q_abs']
    n_comp = m['n_comp']
    # No reversed domain (and no sub-floor winding): ferromagnetic.
    if n_comp == 0 or (m['core_frac'] < core_min and q < q_sk):
        return 'A'
    # Genuine labyrinth: more than one winding, or many domains.
    if q >= q_multi or n_comp >= n_dom:
        return 'L'
    # Chargeless texture (|Q| ~ 0): not a skyrmion -> stripe / labyrinth.
    if q < q_sk:
        return 'L'
    # One winding: a convex-like blob/stripe is a skyrmion; a serpentine,
    # hull-underfilling domain is a labyrinth.
    if m['solidity'] < s_min:
        return 'L'
    # Skyrmion: elongated/spanning (E) vs compact (S).
    if m['x_perc'] or m['dx_lx'] >= x_gate or m['ratio'] >= r_elong:
        return 'E'
    return 'S'


def _save_classes(path, Ts, Js, cls, rows):
    """Persist the class grid and per-cell decision metrics.

    Writes an NPZ with the (T, j) axes, the class-code grid, and the
    metric grids (|Q|, component count, x-percolation, D1/L_x, D1/D2,
    reversed-area fraction) that the stability-classification report
    tabulates.

    Parameters
    ----------
    path : str
        Output NPZ path.
    Ts, Js : numpy.ndarray(1d)
        Grid axes.
    cls : numpy.ndarray(2d)
        Class-code grid.
    rows : list[dict]
        Per-cell metrics from `_classify_field`.
    """
    def grid(key):
        g = np.full((Ts.size, Js.size), np.nan)
        for r in rows:
            i = int(np.argmin(np.abs(Ts - r['T'])))
            k = int(np.argmin(np.abs(Js - r['J'])))
            g[i, k] = float(r[key])
        return g
    np.savez_compressed(
        path, Ts=Ts, Js=Js, cls=cls,
        q_abs=grid('q_abs'), n_comp=grid('n_comp'),
        x_perc=grid('x_perc'), dx_lx=grid('dx_lx'),
        d1_lx=grid('d1_lx'), ratio=grid('ratio'),
        core_frac=grid('core_frac'), solidity=grid('solidity'))
    print(f'Saved: {path}')


# -----------------------------------------------------------------------------
def _plot_stability_map(agg, cls, dx_lx, out_path):
    """Skyrmion stability regime diagram over the (T, J) grid.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js).
    cls : numpy.ndarray(2d)
        Per-cell class codes from the field classifier.
    dx_lx : numpy.ndarray(2d)
        Per-cell D_x / L_x (field x-extent of the reversed domain over
        the track length); >= 0.5 appends the loss-of-periodicity 'p'.
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    # Loss-of-periodicity second index 'p': the field x-extent D_x of the
    # reversed domain reaches half the track length, so it can bridge its
    # own periodic-x image and the single-skyrmion classification is
    # compromised. dx_lx is D_x / L_x (from the field classifier).
    code = {'S': 0, 'E': 1, 'L': 2, 'A': 3, '?': 4}
    z = np.array(
        [[code[cls[i, k]] for k in range(Js.size)]
         for i in range(Ts.size)], dtype=float)
    cmap = mcolors.ListedColormap(
        ['#2c7bb6', '#fdae61', '#d7191c', '#999999', '#ffffff'])
    norm = mcolors.BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    ax.imshow(z, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(Js.size))
    ax.set_xticklabels([f'{j*1e-11:.2g}' for j in Js])
    ax.set_yticks(np.arange(Ts.size))
    ax.set_yticklabels([f'{t:.0f}' for t in Ts])
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel('T (K)')
    ax.set_title('skyrmion stability regimes')
    for i in range(Ts.size):
        for k in range(Js.size):
            c = 'k' if cls[i, k] in ('E', '?') else 'w'
            label = cls[i, k]
            # 'p' (loss of periodicity) only qualifies a skyrmion class;
            # a labyrinth/annihilated cell already voids the single-object
            # periodic picture, so it carries no 'p'.
            if (cls[i, k] in ('S', 'E') and np.isfinite(dx_lx[i, k])
                    and dx_lx[i, k] >= 0.5):
                label = label + 'p'
            ax.text(k, i, label, ha='center', va='center',
                    color=c, fontweight='bold')
    handles = [
        mpatches.Patch(color='#2c7bb6', label='S: compact skyrmion'),
        mpatches.Patch(color='#fdae61', label='E: elongated / spanning'),
        mpatches.Patch(color='#d7191c', label='L: labyrinth / multi-domain'),
        mpatches.Patch(color='#999999', label='A: annihilated (FM)'),
        mpatches.Patch(facecolor='white', edgecolor='k',
                       label=r'$\cdots$p: loss of periodicity '
                             r'($D_x\geq0.5\,L_x$)')]
    ax.legend(handles=handles, bbox_to_anchor=(1.02, 1.0),
              loc='upper left', fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_velocity_vs_j(agg, cls, out_path):
    """Skyrmion drift speed (m/s) versus current density.

    One curve per substrate temperature, coloured by T (cool->warm),
    points joined and carrying the ensemble standard error. Per-cell
    stability class (from the field classifier) is overlaid: 'E' cells
    are ringed (elongated / spanning), and 'L'/'A' cells (labyrinth /
    annihilated -- where the tracked speed is a multi-domain artifact,
    not a skyrmion velocity) are marked with a faded grey cross.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js, v_mean, v_se).
    cls : numpy.ndarray(2d)
        Per-cell class codes from `_classify_grid`.
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    v = agg['v_mean']
    v_se = agg['v_se']
    cmap = plt.get_cmap('coolwarm')
    t_min = float(np.min(Ts))
    t_span = max(float(np.max(Ts)) - t_min, 1.0)
    fig, ax = plt.subplots(figsize=(8.0, 5.5))
    for i, T in enumerate(Ts):
        color = cmap((float(T) - t_min) / t_span)
        ax.errorbar(
            Js * 1e-11, v[i, :], yerr=v_se[i, :],
            marker='o', capsize=3, color=color, label=f'{T:.0f} K')
        for k in range(Js.size):
            if not np.isfinite(v[i, k]):
                continue
            if cls[i, k] == 'E':
                ax.plot(Js[k] * 1e-11, v[i, k], marker='o', ms=11,
                        mfc='none', mec=color, mew=1.6)
            elif cls[i, k] in ('L', 'A'):
                ax.plot(Js[k] * 1e-11, v[i, k], marker='x', ms=9,
                        color='0.4', mew=2.0)
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'skyrmion speed $v$ (m/s)')
    ax.set_title('skyrmion speed vs current density')
    # Stability legend entries: ring = elongated/spanning skyrmion;
    # cross = labyrinth/annihilated (tracked speed not a skyrmion).
    ax.plot([], [], color='0.4', marker='o', ms=11, mfc='none',
            ls='none', label='ring: elongated / spanning')
    ax.plot([], [], color='0.4', marker='x', ms=9, ls='none',
            label='cross: labyrinth / annihilated')
    ax.legend(title='T', fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_axes_vs_t(agg, out_path):
    """D_1, D_2 (nm) versus temperature, one curve per J."""
    Ts = agg['Ts']
    Js = agg['Js']
    D1 = agg['D1_mean'] * 1e9
    D2 = agg['D2_mean'] * 1e9
    D1e = agg['D1_se'] * 1e9
    D2e = agg['D2_se'] * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for k, J in enumerate(Js):
        label = f'{J*1e-11:.1f}'
        axes[0].errorbar(
            Ts, D1[:, k], yerr=D1e[:, k],
            marker='o', capsize=3, label=label)
        axes[1].errorbar(
            Ts, D2[:, k], yerr=D2e[:, k],
            marker='s', capsize=3, label=label)
    axes[0].set_title(r'major axis $D_1$')
    axes[1].set_title(r'minor axis $D_2$')
    for ax in axes:
        ax.set_xlabel(r'$T_{\mathrm{sub}}$ (K)')
        ax.set_ylabel(r'diameter (nm)')
        ax.legend(title=r'$J$ ($10^{11}$ A/m$^2$)')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_relaxed_vs_t(agg, out_path):
    """Pre-drive (J=0) finite-T equilibrium size vs temperature.

    The relaxation is current-independent, so the curves for the
    different J columns should collapse; spread between them is a
    thermal-sampling consistency check. The track width L_y is drawn
    as a reference.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    D1r = agg['D1r_mean'] * 1e9
    D2r = agg['D2r_mean'] * 1e9
    D1re = agg['D1r_se'] * 1e9
    D2re = agg['D2r_se'] * 1e9
    L_y = float(agg['L_y']) * 1e9
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))
    for k, J in enumerate(Js):
        label = f'{J*1e-11:.1f}'
        axes[0].errorbar(
            Ts, D1r[:, k], yerr=D1re[:, k],
            marker='o', capsize=3, label=label)
        axes[1].errorbar(
            Ts, D2r[:, k], yerr=D2re[:, k],
            marker='s', capsize=3, label=label)
    axes[0].set_title(r'relaxed major axis $D_1$ (J=0)')
    axes[1].set_title(r'relaxed minor axis $D_2$ (J=0)')
    for ax in axes:
        ax.axhline(L_y, ls='--', color='k', lw=1.0,
                   label='track width $L_y$')
        ax.set_xlabel(r'$T_{\mathrm{sub}}$ (K)')
        ax.set_ylabel(r'diameter (nm)')
        ax.legend(title=r'$J$ ($10^{11}$ A/m$^2$)')
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _imshow_grid(ax, Ts, Js, Z, title, cmap, vmin, vmax):
    """Heatmap helper over the (T rows, J cols) grid."""
    im = ax.imshow(
        Z, origin='lower', aspect='auto', cmap=cmap,
        vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(Js.size))
    ax.set_xticklabels([f'{j*1e-11:.1f}' for j in Js])
    ax.set_yticks(np.arange(Ts.size))
    ax.set_yticklabels([f'{t:.0f}' for t in Ts])
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$T_{\mathrm{sub}}$ (K)')
    ax.set_title(title)
    return im


# -----------------------------------------------------------------------------
def _plot_ratio_heatmaps(agg, out_path):
    """Heatmaps of D_1 / L_x and D_2 / L_y over the (T, j) grid."""
    Ts = agg['Ts']
    Js = agg['Js']
    L_x = float(agg['L_x'])
    L_y = float(agg['L_y'])
    r1 = agg['D1_mean'] / L_x
    r2 = agg['D2_mean'] / L_y
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    im0 = _imshow_grid(
        axes[0], Ts, Js, r1,
        r'$D_1 / L_x$ (along motion)', 'viridis', 0.0, 1.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    im1 = _imshow_grid(
        axes[1], Ts, Js, r2,
        r'$D_2 / L_y$ (track width)', 'viridis', 0.0, 1.0)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.suptitle(
        f'length-scale / box  '
        f'(L_x={L_x*1e9:.0f} nm, L_y={L_y*1e9:.0f} nm)')
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_survival(agg, cls, out_path):
    """Survival probability and Hall-angle maps.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js, P_surv, theta_mean).
    cls : numpy.ndarray(2d)
        Per-cell class codes from the field classifier. The Hall
        angle is only meaningful for a tracked skyrmion, so cells
        that are not 'S'/'E' (labyrinth, annihilated, unknown) are
        masked in the theta panel.
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    im0 = _imshow_grid(
        axes[0], Ts, Js, agg['P_surv'],
        r'survival probability $P_{\mathrm{surv}}$',
        'magma', 0.0, 1.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    theta = np.where(
        np.isin(cls, ('S', 'E')), agg['theta_mean'], np.nan)
    tmax = float(np.nanmax(np.abs(theta))) if np.any(
        np.isfinite(theta)) else 1.0
    im1 = _imshow_grid(
        axes[1], Ts, Js, theta,
        r'skyrmion Hall angle $\theta_H$ (deg), S/E cells only',
        'coolwarm', -tmax, tmax)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _load_anim(path):
    """Load an animation dump into a format-agnostic dict.

    Supports both the Python driver layout (`field_m_*` named
    configs + `anim_mz_*` frame stack) and the C++ SnapshotBuffer
    layout (`m_top`/`m_bot` full-field stack tagged by `phase_id`,
    0 = relax, 1 = drive).

    Parameters
    ----------
    path : str
        Path to the `anim_*.npz` dump.

    Returns
    -------
    na : dict
        Keys `mi_top`/`mi_bot`, `mr_top`/`mr_bot`, `mf_top`/`mf_bot`
        (initial / relaxed / final (ny, nx, 3) fields per layer),
        `mz_top` (drive-phase m_z stack, shape (n_frames, ny, nx))
        and `t` (frame times).
    """
    d = np.load(path, allow_pickle=True)
    if 'anim_mz_top' in d.files:
        # Python driver layout.
        return {
            'mi_top': d['field_m_initial_top'],
            'mr_top': d['field_m_relaxed_top'],
            'mf_top': d['field_m_final_top'],
            'mi_bot': d['field_m_initial_bot'],
            'mr_bot': d['field_m_relaxed_bot'],
            'mf_bot': d['field_m_final_bot'],
            'mz_top': d['anim_mz_top'],
            't': d['anim_t'],
        }
    if 'm_top' in d.files and 'phase_id' in d.files:
        # C++ SnapshotBuffer layout.
        m_top = d['m_top']
        m_bot = d['m_bot']
        phase = d['phase_id']
        t = d['time_s']
        relax_idx = np.flatnonzero(phase == 0)
        drive_idx = np.flatnonzero(phase == 1)
        if relax_idx.size == 0 or drive_idx.size == 0:
            raise RuntimeError(
                f'_load_anim: {path!r} lacks both relax and '
                f'drive frames (phase_id).')
        if 'Q_top' not in d.files:
            raise RuntimeError(
                f'_load_anim: {path!r} lacks the Q_top array needed '
                f'for |Q| stability classification.')
        q_final = float(np.asarray(d['Q_top'])[drive_idx[-1]])
        return {
            'mi_top': m_top[relax_idx[0]],
            'mr_top': m_top[relax_idx[-1]],
            'mf_top': m_top[drive_idx[-1]],
            'mi_bot': m_bot[relax_idx[0]],
            'mr_bot': m_bot[relax_idx[-1]],
            'mf_bot': m_bot[drive_idx[-1]],
            'mz_top': m_top[drive_idx][..., 2],
            't': t[drive_idx],
            'q_final': q_final,
        }
    raise RuntimeError(
        f'_load_anim: unrecognised dump layout in {path!r}; '
        f'keys={list(d.files)}.')


# -----------------------------------------------------------------------------
def _plot_three_configs(na, a, out_path):
    """2x3 initial / relaxed / final m_z still.

    Top row: top layer. Bottom row: bottom layer.
    """
    rows = (
        ('top', (na['mi_top'], na['mr_top'], na['mf_top'])),
        ('bottom', (na['mi_bot'], na['mr_bot'], na['mf_bot'])),
    )
    titles = ('initial', 'relaxed', 'final')
    fig, axes = plt.subplots(2, 3, figsize=(16.0, 10.0))
    for row_axes, (layer, fields) in zip(axes, rows):
        for ax, field, title in zip(row_axes, fields, titles):
            plot_snapshot_mz(
                field, a, ax=ax, title=f'{title} ({layer})',
                show_colorbar=False)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _build_animation(na, a, out_path):
    """Render the top-layer m_z frame stack to an animated GIF."""
    frames = na['mz_top']
    times = na['t']
    n_frames = frames.shape[0]
    if n_frames == 0:
        raise RuntimeError(
            'plot_track_width: anim_mz_top has zero frames.')
    ny, nx = frames.shape[1:]
    x_nm = np.arange(nx) * a * 1e9
    y_nm = np.arange(ny) * a * 1e9
    X, Y = np.meshgrid(x_nm, y_nm, indexing='xy')
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * ny / nx))
    mesh = ax.pcolormesh(
        X, Y, frames[0], cmap='RdBu_r', norm=norm,
        shading='nearest', rasterized=True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$y$ (nm)')
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    title = ax.set_title('')

    def _update(frame_idx):
        # pcolormesh expects the flattened C-order array.
        mesh.set_array(frames[frame_idx].ravel())
        title.set_text(f't = {times[frame_idx]*1e12:.0f} ps')
        return mesh, title

    anim_obj = manim.FuncAnimation(
        fig, _update, frames=n_frames, blit=False)
    anim_obj.save(out_path, writer=manim.PillowWriter(fps=15))
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    """Render all length-scale figures and one animation."""
    # =========================== User Configuration =========================
    # Run tag: box geometry + boundary condition. Used to name the
    # aggregate, the dump folder, and the figure output folder so each
    # run's products stay separate.
    run_tag         = 'box350x500_racetrack_D0p545'
    # Directory holding the (box/BC-tagged) aggregate NPZ.
    in_dir          = 'output/stochastic_llgs/scan_track_width'
    agg_name        = 'aggregate_350x500_racetrack_D0p545.npz'
    # Directory holding the large full-field dumps (anim_*.npz). These
    # can live on a separate drive while the aggregate stays in in_dir.
    dump_dir        = ('/Volumes/T7/skyrmion_simulator/output/'
                       'stochastic_llgs/scan_track_width/' + run_tag)
    out_dir         = os.path.join(
        'output/figures_sllg/track_width', run_tag)
    # GIFs are large, so they are written directly to the T7 backup at
    # the same repo-relative path (not the local disk). The small grid
    # figures and config stills stay local in out_dir.
    gif_dir         = os.path.join(
        '/Volumes/T7/skyrmion_simulator',
        'output/figures_sllg/track_width', run_tag)
    a               = 2.0e-9         # m, lattice constant
    # Animation / still: process every dumped cell matching this glob.
    anim_glob       = 'anim_T*.npz'
    # Re-render config stills / GIFs that already exist. Default False:
    # existing outputs are skipped (the GIF encoding is the slow step).
    overwrite       = False
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(gif_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Class-independent aggregate grid figures.
    agg = _load_aggregate(in_dir, agg_name)
    _plot_axes_vs_j(
        agg, os.path.join(out_dir, 'axes_vs_J.png'))
    _plot_axes_vs_t(
        agg, os.path.join(out_dir, 'axes_vs_T.png'))
    _plot_relaxed_vs_t(
        agg, os.path.join(out_dir, 'relaxed_vs_T.png'))
    _plot_ratio_heatmaps(
        agg, os.path.join(out_dir, 'ratio_heatmaps.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Single pass over the ens0 field dumps: classify each (T, j) cell
    # (|Q| + periodic-x components + D1/L_x gate) and render its config
    # still + GIF. Each dump is loaded once (classification needs the
    # field, so the load cannot be skipped as before).
    Ts = agg['Ts']
    Js = agg['Js']
    L_x = float(agg['L_x'])
    D1m = agg['D1_mean']
    D2m = agg['D2_mean']
    anim_files = sorted(glob.glob(os.path.join(dump_dir, anim_glob)))
    if not anim_files:
        raise RuntimeError(
            f'plot_track_width: no {anim_glob!r} dump files in '
            f'{dump_dir!r}; dumping is gated to ens_idx==0.')
    # Pass 1 -- classify every cell from its final field, then write the
    # class table and the class-dependent maps first (they do not depend
    # on the slow GIF encoding that follows).
    cls = np.full((Ts.size, Js.size), '?', dtype='<U1')
    rows = []
    for anim_path in anim_files:
        stem = os.path.splitext(os.path.basename(anim_path))[0]
        T = float(stem.split('_T')[1].split('_j')[0])
        J = float(stem.split('_j')[1])
        i = int(np.argmin(np.abs(Ts - T)))
        k = int(np.argmin(np.abs(Js - J)))
        na = _load_anim(anim_path)
        code, m = _classify_field(
            na['mf_top'][..., 2], abs(na['q_final']),
            float(D1m[i, k]), float(D2m[i, k]), L_x)
        cls[i, k] = code
        rows.append({'T': T, 'J': J, 'code': code, **m})
        print(f"  {stem}: |Q|={m['q_abs']:.2f}  Nc={m['n_comp']}  "
              f"xperc={int(m['x_perc'])}  Dx/Lx={m['dx_lx']:.2f}  "
              f"D1/D2={m['ratio']:.2f}  -> {code}")
    _save_classes(
        os.path.join(in_dir, f'stability_classes_{run_tag}.npz'),
        Ts, Js, cls, rows)
    dx_lx = np.full((Ts.size, Js.size), np.nan)
    for r in rows:
        i = int(np.argmin(np.abs(Ts - r['T'])))
        k = int(np.argmin(np.abs(Js - r['J'])))
        dx_lx[i, k] = r['dx_lx']
    _plot_stability_map(
        agg, cls, dx_lx, os.path.join(out_dir, 'stability_map.png'))
    _plot_velocity_vs_j(
        agg, cls, os.path.join(out_dir, 'velocity_vs_J.png'))
    _plot_survival(
        agg, cls, os.path.join(out_dir, 'survival_hall.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pass 2 -- config stills + GIFs (skip existing; the GIF encode is the
    # slow step and its outputs go to the T7 gif_dir).
    for anim_path in anim_files:
        stem = os.path.splitext(os.path.basename(anim_path))[0]
        cfg_path = os.path.join(out_dir, f'{stem}_configs.png')
        gif_path = os.path.join(gif_dir, f'{stem}.gif')
        need_cfg = overwrite or not os.path.isfile(cfg_path)
        need_gif = overwrite or not os.path.isfile(gif_path)
        if not need_cfg and not need_gif:
            print(f'Skip (exists): {stem}')
            continue
        na = _load_anim(anim_path)
        if need_cfg:
            _plot_three_configs(na, a, cfg_path)
        if need_gif:
            _build_animation(na, a, gif_path)


# =============================================================================
if __name__ == '__main__':
    main()
