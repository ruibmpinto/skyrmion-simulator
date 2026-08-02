"""Box-size comparison of the racetrack stability classification.

Compares the finite-temperature stability maps of one anisotropy
setting across track lengths L_x at fixed track width L_y, reading the
per-case `aggregate.npz` (ensemble class counts, survival, axes) and
the per-case `stability_classes_<tag>.npz` (ens0 decision metrics)
written by `scripts/plot_track_width.py`.

Renders:

- Side-by-side ensemble stability maps, one panel per box.
- A class-flip map marking the cells whose dominant class changes
  with L_x, plus the survival difference.
- Heatmaps of the periodic-image gate metric D_x / L_x, with the
  gate contour, showing that the flips track the gate and not the
  morphology.
- Box-overlaid survival and aspect-ratio curves, the evidence that
  the object itself is box-independent.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m scripts.plot_stability_box_comparison
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _set_style():
    """Enlarge the default text sizes for figures read on a page.

    The comparison figures are reproduced at full text width in the
    report, where matplotlib's defaults render too small to read.
    """
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 18,
    })


# -----------------------------------------------------------------------------
def _load_case(campaign_root, fig_root, tag):
    """Load one case's ensemble aggregate and ens0 decision metrics.

    Parameters
    ----------
    campaign_root : str
        Directory holding the per-case campaign subdirectories.
    fig_root : str
        Directory holding the per-case figure subdirectories, where
        `plot_track_width` wrote the ens0 class NPZ.
    tag : str
        Case tag, `<hk>_<D>_<box>`.

    Returns
    -------
    case : dict
        Keys Ts, Js, L_x, L_y, dominant, P_surv, ratio, D1, D2,
        n_cmp, n_elo, n_lab, n_ann, n_ens, dx_lx, solidity, q_abs,
        v_mean, v_se, theta_mean, theta_se.
    """
    agg_path = os.path.join(campaign_root, tag, 'aggregate.npz')
    if not os.path.isfile(agg_path):
        raise RuntimeError(f'_load_case: missing aggregate {agg_path!r}.')
    d = np.load(agg_path)
    n_cmp = np.asarray(d['n_S'], float)
    n_elo = np.asarray(d['n_E'], float)
    n_lab = np.asarray(d['n_L'], float)
    n_ann = np.asarray(d['n_A'], float)
    stacks = np.stack([n_cmp, n_elo, n_lab, n_ann], axis=-1)
    total = stacks.sum(axis=-1)
    dominant = np.where(total > 0, np.argmax(stacks, axis=-1), 4)
    cls_path = os.path.join(fig_root, tag, f'stability_classes_{tag}.npz')
    if not os.path.isfile(cls_path):
        raise RuntimeError(f'_load_case: missing classes {cls_path!r}.')
    c = np.load(cls_path, allow_pickle=True)
    return {
        'Ts': np.asarray(d['Ts'], float),
        'Js': np.asarray(d['Js'], float),
        'L_x': float(d['L_x']),
        'L_y': float(d['L_y']),
        'dominant': dominant.astype(int),
        'P_surv': np.asarray(d['P_surv'], float),
        'D1': np.asarray(d['D1_mean'], float),
        'D2': np.asarray(d['D2_mean'], float),
        'ratio': np.asarray(d['D1_mean'], float)
        / np.asarray(d['D2_mean'], float),
        'n_cmp': n_cmp, 'n_elo': n_elo, 'n_lab': n_lab, 'n_ann': n_ann,
        'n_ens': np.asarray(d['n_ens'], float),
        'dx_lx': np.asarray(c['dx_lx'], float),
        'solidity': np.asarray(c['solidity'], float),
        'q_abs': np.asarray(c['q_abs'], float),
        'cls': np.asarray(c['cls']),
        'v_mean': np.asarray(d['v_mean'], float),
        'v_se': np.asarray(d['v_se'], float),
        'theta_mean': np.asarray(d['theta_mean'], float),
        'theta_se': np.asarray(d['theta_se'], float),
    }


# -----------------------------------------------------------------------------
def _class_cmap():
    """Colormap and norm for the four distinct stability classes.

    Returns
    -------
    cmap : matplotlib.colors.ListedColormap
        Compact / elongated / labyrinth / annihilated / no-data.
    norm : matplotlib.colors.BoundaryNorm
        Matching discrete norm.
    """
    cmap = mcolors.ListedColormap(
        ['#2c7bb6', '#fdae61', '#d7191c', '#999999', '#ffffff'])
    norm = mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    return cmap, norm


# -----------------------------------------------------------------------------
def _grid_ticks(ax, Ts, Js):
    """Label a (T, J) image grid with physical tick values.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    Ts, Js : numpy.ndarray(1d)
        Temperature (K) and current-density (A/m^2) axes.
    """
    ax.set_xticks(np.arange(Js.size))
    ax.set_xticklabels([f'{j*1e-11:.2g}' for j in Js])
    ax.set_yticks(np.arange(Ts.size))
    ax.set_yticklabels([f'{t:.0f}' for t in Ts])
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')


# -----------------------------------------------------------------------------
def _plot_maps(cases, labels, out_path):
    """Ensemble stability maps side by side, one panel per box.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Panel labels, one per case.
    out_path : str
        Output PNG path.
    """
    cmap, norm = _class_cmap()
    fig, axes = plt.subplots(
        1, len(cases), figsize=(5.4*len(cases), 5.0), sharey=True)
    for ax, case, lab in zip(np.atleast_1d(axes), cases, labels):
        Ts, Js = case['Ts'], case['Js']
        ax.imshow(case['dominant'], origin='lower', aspect='auto',
                  cmap=cmap, norm=norm)
        _grid_ticks(ax, Ts, Js)
        ax.set_title(lab, fontsize=16)
        for i in range(Ts.size):
            for k in range(Js.size):
                col = 'k' if case['dominant'][i, k] == 4 else 'w'
                p = case['P_surv'][i, k]
                ax.text(k, i, f'{p:.2f}' if np.isfinite(p) else '-',
                        ha='center', va='center', color=col, fontsize=11)
    np.atleast_1d(axes)[0].set_ylabel('T (K)')
    handles = [
        mpatches.Patch(color='#2c7bb6', label='compact skyrmion (S)'),
        mpatches.Patch(color='#fdae61', label='elongated (E)'),
        mpatches.Patch(color='#d7191c', label='labyrinth (L)'),
        mpatches.Patch(color='#999999', label='annihilated (A)')]
    fig.legend(handles=handles, loc='lower center', ncol=4,
               fontsize=13, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle('ensemble stability regimes; cell value = '
                 r'$P_{\mathrm{surv}}$', fontsize=17)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_composition_box(case, label, out_path):
    """Per-class ensemble fractions of one box, on a common scale.

    The annihilated channel is omitted: it is identically zero for this
    anisotropy (verified from the aggregate), so a fourth all-empty
    panel would carry no information. Its absence is the statement that
    the skyrmion is never lost by collapse to the ferromagnet.

    Parameters
    ----------
    case : dict
        Loaded case payload (Ts, Js, n_cmp, n_elo, n_lab, n_ens).
    label : str
        Box label for the title.
    out_path : str
        Output PNG path.
    """
    Ts, Js = case['Ts'], case['Js']
    n_ann = np.nansum(case['n_ann'])
    if n_ann > 0:
        raise RuntimeError(
            f'_plot_composition_box: {label} has {int(n_ann)} '
            f'annihilated realizations; the A panel cannot be dropped.')
    tot = np.where(case['n_ens'] > 0, case['n_ens'], np.nan)
    panels = ((case['n_cmp']/tot, r'$P(\mathrm{S})$ -- compact skyrmion'),
              (case['n_elo']/tot, r'$P(\mathrm{E})$ -- elongated'),
              (case['n_lab']/tot, r'$P(\mathrm{L})$ -- labyrinth'))
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    for ax, (z, name) in zip(axes, panels):
        im = ax.imshow(z, origin='lower', aspect='auto', cmap='viridis',
                       vmin=0.0, vmax=1.0)
        _grid_ticks(ax, Ts, Js)
        ax.set_title(name, fontsize=16)
        for i in range(Ts.size):
            for k in range(Js.size):
                v = z[i, k]
                if not np.isfinite(v) or v <= 0.0:
                    continue
                ax.text(k, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=11, color='w' if v < 0.6 else 'k')
    axes[0].set_ylabel('T (K)')
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02,
                 label='ensemble fraction')
    fig.suptitle(f'class composition -- {label}', fontsize=17)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_flips(cases, labels, out_path):
    """Class-flip map and survival difference between consecutive boxes.

    The top row marks, for each consecutive pair of boxes, the cells
    whose dominant class changes; the bottom row is the survival
    difference over the same pair.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Short box labels, one per case.
    out_path : str
        Output PNG path.
    """
    names = np.array(['S', 'E', 'L', 'A', '-'])
    n_pair = len(cases) - 1
    fig, axes = plt.subplots(2, n_pair, figsize=(6.0*n_pair, 9.5),
                             squeeze=False)
    for p in range(n_pair):
        lo, hi = cases[p], cases[p+1]
        Ts, Js = lo['Ts'], lo['Js']
        flip = (lo['dominant'] != hi['dominant']).astype(float)
        ax = axes[0][p]
        ax.imshow(flip, origin='lower', aspect='auto', cmap='Purples',
                  vmin=0, vmax=1.6)
        _grid_ticks(ax, Ts, Js)
        ax.set_title(f'class change: {labels[p]} $\\rightarrow$ '
                     f'{labels[p+1]}  ({int(flip.sum())} cells)',
                     fontsize=16)
        for i in range(Ts.size):
            for k in range(Js.size):
                a, b = lo['dominant'][i, k], hi['dominant'][i, k]
                txt = (f'{names[a]}$\\rightarrow${names[b]}'
                       if a != b else f'{names[a]}')
                ax.text(k, i, txt, ha='center', va='center',
                        fontsize=11,
                        color='k' if a != b else '#999999')
        ax = axes[1][p]
        diff = hi['P_surv'] - lo['P_surv']
        im = ax.imshow(diff, origin='lower', aspect='auto',
                       cmap='RdBu_r', vmin=-0.25, vmax=0.25)
        _grid_ticks(ax, Ts, Js)
        ax.set_title(r'$\Delta P_{\mathrm{surv}}$: '
                     f'{labels[p+1]} $-$ {labels[p]}', fontsize=16)
        for i in range(Ts.size):
            for k in range(Js.size):
                v = diff[i, k]
                ax.text(k, i, f'{v:+.2f}' if np.isfinite(v) else '-',
                        ha='center', va='center', fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046)
    axes[0][0].set_ylabel('T (K)')
    axes[1][0].set_ylabel('T (K)')
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_gate(cases, labels, x_gate, out_path):
    """Periodic-image gate metric D_x / L_x for each box.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Panel labels, one per case.
    x_gate : float
        Gate threshold on D_x / L_x.
    out_path : str
        Output PNG path.
    """
    fig, axes = plt.subplots(
        1, len(cases), figsize=(4.6*len(cases), 4.2), sharey=True)
    for ax, case, lab in zip(np.atleast_1d(axes), cases, labels):
        Ts, Js = case['Ts'], case['Js']
        g = case['dx_lx']
        im = ax.imshow(g, origin='lower', aspect='auto', cmap='viridis',
                       vmin=0.0, vmax=1.0)
        _grid_ticks(ax, Ts, Js)
        ax.set_title(lab, fontsize=16)
        for i in range(Ts.size):
            for k in range(Js.size):
                v = g[i, k]
                if not np.isfinite(v):
                    ax.text(k, i, '-', ha='center', va='center',
                            fontsize=11, color='k')
                    continue
                # Mark the cells the gate sends to the elongated class.
                mark = '*' if v >= x_gate else ''
                ax.text(k, i, f'{v:.2f}{mark}', ha='center',
                        va='center', fontsize=11,
                        color='w' if v < 0.75 else 'k')
    np.atleast_1d(axes)[0].set_ylabel('T (K)')
    fig.colorbar(im, ax=np.atleast_1d(axes).tolist(), fraction=0.02,
                 label=r'$D_x / L_x$ (ens0)')
    fig.suptitle(r'periodic-image gate metric; * marks '
                 f'$D_x/L_x \\geq {x_gate}$', fontsize=15, y=1.04)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_invariance(cases, labels, styles, out_path):
    """Survival, aspect ratio and D_x overlaid across boxes.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Legend labels, one per case.
    styles : list[str]
        Line styles, one per case.
    out_path : str
        Output PNG path.
    """
    Ts = cases[0]['Ts']
    Js = cases[0]['Js']
    colors = plt.cm.viridis(np.linspace(0, 0.9, Js.size))
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.2))
    ax = axes[0]
    for case, lab, st in zip(cases, labels, styles):
        for k in range(Js.size):
            ax.plot(Ts, case['P_surv'][:, k], st, color=colors[k],
                    marker='o', ms=3,
                    label=(f'J={Js[k]*1e-11:.2g}' if st == '-' else None))
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$P_{\mathrm{surv}}$')
    ax.set_title('survival (all boxes overlaid)', fontsize=18)
    # Registered as an artist so the track-length legend added below
    # does not displace it.
    ax.add_artist(ax.legend(fontsize=14, ncol=2, frameon=False,
                            loc='upper right'))
    ax = axes[1]
    for case, lab, st in zip(cases, labels, styles):
        for i in range(Ts.size):
            if not np.isfinite(case['ratio'][i, :]).any():
                continue
            ax.plot(Js*1e-11, case['ratio'][i, :], st,
                    color=plt.cm.plasma(i/max(Ts.size-1, 1)),
                    marker='s', ms=3,
                    label=(f'T={Ts[i]:.0f} K' if st == '-' else None))
    ax.axhline(1.7, color='k', ls=':', lw=1.0,
               label=r'$D_1/D_2=1.7$ (elongation)')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$D_1 / D_2$')
    ax.set_title('aspect ratio (all boxes overlaid)', fontsize=18)
    ax.legend(fontsize=14, ncol=2, frameon=False)
    ax = axes[2]
    # Only skyrmion-bearing cells: for a labyrinth D_x merely spans the
    # box and carries no length scale.
    p_min = 0.5
    for case, lab, st in zip(cases, labels, styles):
        d_x = np.where(case['P_surv'] >= p_min,
                       case['dx_lx']*case['L_x']*1e9, np.nan)
        for i in range(Ts.size):
            if not np.isfinite(d_x[i, :]).any():
                continue
            ax.plot(Js*1e-11, d_x[i, :], st,
                    color=plt.cm.plasma(i/max(Ts.size-1, 1)),
                    marker='^', ms=3,
                    label=(f'T={Ts[i]:.0f} K' if st == '-' else None))
    for case, st, lab in zip(cases, styles, labels):
        ax.axhline(0.5*case['L_x']*1e9, color='0.6', ls=st, lw=0.9,
                   label=rf'$0.5\,L_x$ gate, {lab}')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$D_x$ (nm)')
    ax.set_title(r'absolute $x$-extent (ens0)', fontsize=18)
    ax.legend(fontsize=14, ncol=2, frameon=False)
    box_handles = [mlines.Line2D([], [], color='0.3', ls=st, label=lab)
                   for lab, st in zip(labels, styles)]
    axes[0].legend(handles=box_handles, frameon=False, fontsize=12,
                   loc='lower left', title='track length')
    fig.suptitle('box-size invariance', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_transport(cases, labels, styles, out_path):
    """Drift speed and Hall angle versus current, overlaid across boxes.

    Only cells whose ensemble survival exceeds a floor are drawn: a
    dead or labyrinth cell has no single-object velocity or Hall angle.
    Error bars are the ensemble standard errors.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Legend labels, one per case.
    styles : list[str]
        Line styles, one per case.
    out_path : str
        Output PNG path.
    """
    Ts, Js = cases[0]['Ts'], cases[0]['Js']
    p_min = 0.5
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.6))
    for case, st in zip(cases, styles):
        keep = case['P_surv'] >= p_min
        v = np.where(keep, case['v_mean'], np.nan)
        v_se = np.where(keep, case['v_se'], np.nan)
        th = np.where(keep, case['theta_mean'], np.nan)
        th_se = np.where(keep, case['theta_se'], np.nan)
        for i in range(Ts.size):
            if not np.isfinite(v[i, :]).any():
                continue
            col = plt.cm.plasma(i/max(Ts.size-1, 1))
            lab = f'T={Ts[i]:.0f} K' if st == '-' else None
            axes[0].errorbar(Js*1e-11, v[i, :], yerr=v_se[i, :],
                             fmt=st, color=col, marker='o', ms=3.5,
                             capsize=2, lw=1.2, label=lab)
            axes[1].errorbar(Js*1e-11, th[i, :], yerr=th_se[i, :],
                             fmt=st, color=col, marker='s', ms=3.5,
                             capsize=2, lw=1.2, label=lab)
    axes[0].set_ylabel(r'$v$ (m/s)')
    axes[0].set_title('drift speed (ensemble mean $\\pm$ SE)',
                      fontsize=16)
    axes[1].axhline(0.0, color='k', ls=':', lw=1.0)
    axes[1].set_ylabel(r'$\bar{\theta}_H$ (deg)')
    axes[1].set_title('Hall angle (ensemble mean $\\pm$ SE)',
                      fontsize=16)
    for ax in axes:
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_box_aspect(1)
        ax.add_artist(ax.legend(fontsize=11, ncol=2, frameon=False,
                                loc='upper left'))
    box_handles = [mlines.Line2D([], [], color='0.3', ls=st, label=lab)
                   for lab, st in zip(labels, styles)]
    axes[0].legend(handles=box_handles, frameon=False, fontsize=12,
                   loc='lower right', title='track length')
    fig.suptitle('transport across box sizes', fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _final_mz(path):
    """Final drive-frame top-layer m_z of one field dump.

    Reads only the top-layer stack and returns its last drive frame,
    so a multi-GB dump costs one layer instead of the whole file.

    Parameters
    ----------
    path : str
        Path to the `anim_*.npz` dump.

    Returns
    -------
    mz : numpy.ndarray(2d)
        Final top-layer m_z, shape (ny, nx).
    """
    with np.load(path, allow_pickle=True) as d:
        if 'm_top' not in d.files or 'phase_id' not in d.files:
            raise RuntimeError(
                f'_final_mz: unexpected dump layout in {path!r}; '
                f'keys={list(d.files)}.')
        drive_idx = np.flatnonzero(np.asarray(d['phase_id']) == 1)
        if drive_idx.size == 0:
            raise RuntimeError(f'_final_mz: {path!r} has no drive frames.')
        m_top = d['m_top']
        mz = np.array(m_top[drive_idx[-1], ..., 2], dtype=np.float32)
        del m_top
    return mz


# -----------------------------------------------------------------------------
def _plot_config_table(case, dump_dir, a, x_gate, title, out_path):
    """Final top-layer m_z of every (T, J) cell on the stability grid.

    One panel per cell, ens0 realization, drawn at physical scale and
    labelled with the ens0 stability class; the suffix p marks the
    cells whose x-extent trips the periodic-image gate.

    Parameters
    ----------
    case : dict
        Loaded case payload (Ts, Js, dx_lx, and the ens0 class grid).
    dump_dir : str
        Directory holding the `anim_*.npz` dumps of this case.
    a : float
        Lattice constant (m).
    x_gate : float
        Gate threshold on D_x / L_x.
    title : str
        Figure title.
    out_path : str
        Output PNG path.
    """
    a_nm = a*1e9
    Ts, Js = case['Ts'], case['Js']
    cls = case['cls']
    nt, nj = Ts.size, Js.size
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    # Height per row follows the track aspect, so a long flat track does
    # not end up with its rows crammed together; the extra term is the
    # inter-row gap that gridspec then distributes.
    panel_w = 2.4
    row_h = panel_w*(case['L_y']/case['L_x']) + 0.55
    fig, axes = plt.subplots(nt, nj, figsize=(panel_w*nj, row_h*nt),
                             gridspec_kw={'hspace': 0.45,
                                          'wspace': 0.08})
    # Rows top-to-bottom = decreasing T, columns = increasing J, so the
    # sheet reads like the stability map.
    order = np.argsort(Ts)[::-1]
    for r, i in enumerate(order):
        for k in range(nj):
            ax = axes[r, k]
            fn = os.path.join(
                dump_dir, f'anim_T{Ts[i]:05.1f}_j{Js[k]:.2e}.npz')
            code = str(cls[i, k])
            if (code in ('S', 'E') and np.isfinite(case['dx_lx'][i, k])
                    and case['dx_lx'][i, k] >= x_gate):
                code = code + 'p'
            if os.path.isfile(fn):
                mz = _final_mz(fn)
                ny, nx = mz.shape
                ax.imshow(mz, origin='lower', cmap='RdBu_r', norm=norm,
                          extent=[0.0, nx*a_nm, 0.0, ny*a_nm],
                          aspect='equal')
                del mz
            else:
                ax.text(0.5, 0.5, 'no dump', transform=ax.transAxes,
                        ha='center', va='center', fontsize=13,
                        color='0.5')
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(0.03, 0.94, code, transform=ax.transAxes,
                    ha='left', va='top', fontsize=16, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8,
                              edgecolor='none', pad=1.5))
            if k == 0:
                ax.set_ylabel(f'{Ts[i]:.0f}', fontsize=18, rotation=0,
                              labelpad=18, va='center', ha='right')
            if r == nt - 1:
                ax.set_xlabel(f'{Js[k]*1e-11:.2g}', fontsize=18)
    fig.supxlabel(r'$J$ ($10^{11}$ A/m$^2$)', fontsize=21)
    fig.supylabel(r'$T$ (K)', fontsize=21)
    fig.suptitle(title, fontsize=19)
    # Explicit margins: tight_layout would discard the gridspec hspace.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.07)
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _equil_stats(case_dir):
    """Ensemble mean and spread of the equilibration axes per T.

    Reads every `equil_series_T*_ens*.npz` in `case_dir`, groups by
    substrate temperature and truncates each group to the window all
    its members share before averaging.

    Parameters
    ----------
    case_dir : str
        Campaign case directory holding the equil_series files.

    Returns
    -------
    stats : dict
        Maps T (K) to (t_ns, d1_mean, d1_std, d2_mean, d2_std, n_ens).
    """
    paths = sorted(glob.glob(
        os.path.join(case_dir, 'equil_series_T*_ens*.npz')))
    if not paths:
        raise RuntimeError(
            f'_equil_stats: no equil_series files in {case_dir!r}.')
    by_t = {}
    for path in paths:
        base = os.path.basename(path)
        t_tok = base.split('_T')[1].split('_ens')[0]
        by_t.setdefault(float(t_tok), []).append(path)
    stats = {}
    for t_sub in sorted(by_t):
        series = [np.load(p) for p in by_t[t_sub]]
        n = min(len(s['t_s']) for s in series)
        t_ns = series[0]['t_s'][:n]*1e9
        d1 = np.stack([s['D1_m'][:n] for s in series])*1e9
        d2 = np.stack([s['D2_m'][:n] for s in series])*1e9
        stats[t_sub] = (t_ns, d1.mean(axis=0), d1.std(axis=0),
                        d2.mean(axis=0), d2.std(axis=0), len(series))
    return stats


# -----------------------------------------------------------------------------
def _plot_equil_boxes(campaign_root, tags, labels, out_path):
    """Equilibration axes for every box on shared axes limits.

    Rows are D_1 and D_2, columns are the boxes; every panel in a row
    shares its y limits and all panels share x, so the boxes can be
    read against each other directly. Square panels, no grid.

    Parameters
    ----------
    campaign_root : str
        Directory holding the per-case campaign subdirectories.
    tags : list[str]
        Case tags, ordered by increasing L_x.
    labels : list[str]
        Column labels, one per case.
    out_path : str
        Output PNG path.
    """
    stats = [_equil_stats(os.path.join(campaign_root, t)) for t in tags]
    t_values = sorted(stats[0])
    cmap = plt.get_cmap('viridis')
    # Common limits over every box, so the panels are comparable.
    d1_hi = max(np.nanmax(s[t][1] + s[t][2]) for s in stats
                for t in s)
    d2_hi = max(np.nanmax(s[t][3] + s[t][4]) for s in stats
                for t in s)
    lo = min(min(np.nanmin(s[t][1] - s[t][2]),
                 np.nanmin(s[t][3] - s[t][4])) for s in stats for t in s)
    t_hi = max(np.nanmax(s[t][0]) for s in stats for t in s)
    fig, axes = plt.subplots(2, len(tags), figsize=(5.2*len(tags), 10.2))
    for col, (st, lab) in enumerate(zip(stats, labels)):
        for k, t_sub in enumerate(t_values):
            if t_sub not in st:
                continue
            t_ns, d1_m, d1_s, d2_m, d2_s, n_ens = st[t_sub]
            c = cmap(k/max(len(t_values) - 1, 1))
            lbl = f'$T = {t_sub:.0f}$ K ({n_ens} ens)'
            for row, (mean, std) in enumerate(
                    ((d1_m, d1_s), (d2_m, d2_s))):
                ax = axes[row][col]
                ax.plot(t_ns, mean, color=c,
                        label=(lbl if row == 0 else None))
                ax.fill_between(t_ns, mean - std, mean + std,
                                color=c, alpha=0.2)
        axes[0][col].set_title(lab, fontsize=17)
    for row, (hi, name) in enumerate(((d1_hi, r'$D_1$'),
                                      (d2_hi, r'$D_2$'))):
        for col in range(len(tags)):
            ax = axes[row][col]
            ax.set_xlim(0.0, t_hi)
            ax.set_ylim(0.95*lo, 1.05*hi)
            ax.set_xlabel(r'$t$ (ns)')
            ax.set_box_aspect(1)
            if col == 0:
                ax.set_ylabel(f'{name} (nm)')
    axes[0][0].legend(loc='upper left', frameon=False, fontsize=12)
    fig.suptitle('thermal equilibration', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_velocity_boxes(cases, labels, out_path):
    """Drift speed versus current for every box on shared limits.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Panel labels, one per case.
    out_path : str
        Output PNG path.
    """
    Ts, Js = cases[0]['Ts'], cases[0]['Js']
    p_min = 0.5
    v_hi = np.nanmax([np.nanmax(np.where(c['P_surv'] >= p_min,
                                        c['v_mean'] + c['v_se'], np.nan))
                      for c in cases])
    fig, axes = plt.subplots(1, len(cases), figsize=(5.2*len(cases), 5.2))
    for ax, case, lab in zip(np.atleast_1d(axes), cases, labels):
        keep = case['P_surv'] >= p_min
        v = np.where(keep, case['v_mean'], np.nan)
        v_se = np.where(keep, case['v_se'], np.nan)
        for i in range(Ts.size):
            if not np.isfinite(v[i, :]).any():
                continue
            ax.errorbar(Js*1e-11, v[i, :], yerr=v_se[i, :], fmt='-',
                        color=plt.cm.plasma(i/max(Ts.size-1, 1)),
                        marker='o', ms=4, capsize=2,
                        label=f'T={Ts[i]:.0f} K')
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_title(lab, fontsize=17)
        ax.set_ylim(0.0, 1.05*v_hi)
        ax.set_box_aspect(1)
        ax.legend(fontsize=12, frameon=False)
    np.atleast_1d(axes)[0].set_ylabel(r'$v$ (m/s)')
    fig.suptitle(r'drift speed (ensemble mean $\pm$ SE)',
                 fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_survival_hall_boxes(cases, labels, out_path):
    """Survival and Hall-angle maps for every box on shared scales.

    Rows are P_surv, the ensemble-mean Hall angle and its standard
    error; columns are the boxes. Each row uses one colour scale over
    all boxes, so the maps are directly comparable.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Column labels, one per case.
    out_path : str
        Output PNG path.
    """
    Ts, Js = cases[0]['Ts'], cases[0]['Js']
    th_hi = np.nanmax([np.nanmax(np.abs(c['theta_mean'])) for c in cases])
    se_hi = np.nanmax([np.nanmax(c['theta_se']) for c in cases])
    rows = (('P_surv', r'$P_{\mathrm{surv}}$', 'magma', 0.0, 1.0),
            ('theta_mean', r'$\bar{\theta}_H$ (deg)', 'RdBu_r',
             -th_hi, th_hi),
            ('theta_se', r'$\sigma_{\theta_H}/\sqrt{n}$ (deg)',
             'viridis', 0.0, se_hi))
    fig, axes = plt.subplots(3, len(cases),
                             figsize=(5.2*len(cases), 13.8),
                             gridspec_kw={'hspace': 0.42})
    for row, (key, name, cmap, vmin, vmax) in enumerate(rows):
        for col, (case, lab) in enumerate(zip(cases, labels)):
            ax = axes[row][col]
            im = ax.imshow(case[key], origin='lower', aspect='auto',
                           cmap=cmap, vmin=vmin, vmax=vmax)
            _grid_ticks(ax, Ts, Js)
            if col == 0:
                ax.set_ylabel('T (K)')
            ax.set_title(f'{name} -- {lab}', fontsize=16)
        fig.colorbar(im, ax=axes[row].tolist(), fraction=0.025,
                     label=name)
    fig.suptitle('survival and Hall angle', fontsize=18)
    fig.savefig(out_path, dpi=160, bbox_inches='tight')
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _write_summary(cases, labels, x_gate, out_path):
    """Write the per-cell comparison table consumed by the report.

    Parameters
    ----------
    cases : list[dict]
        Loaded cases, ordered by increasing L_x.
    labels : list[str]
        Short box labels, one per case.
    x_gate : float
        Gate threshold on D_x / L_x.
    out_path : str
        Output text path.
    """
    names = np.array(['S', 'E', 'L', 'A', '-'])
    Ts, Js = cases[0]['Ts'], cases[0]['Js']
    lines = []
    head = ('  T    J   ' + '  '.join(f'{lab:>12s}' for lab in labels)
            + '   ratio    flips')
    lines.append(head)
    for i in range(Ts.size):
        for k in range(Js.size):
            cols = []
            for case in cases:
                c = names[case['dominant'][i, k]]
                g = case['dx_lx'][i, k]
                cols.append(f'{c} {g:5.2f}'
                            f'{"*" if np.isfinite(g) and g >= x_gate else " "}'
                            f'{case["P_surv"][i, k]:5.2f}')
            codes = [names[case['dominant'][i, k]] for case in cases]
            flip = 'yes' if len(set(codes)) > 1 else ''
            r = cases[-1]['ratio'][i, k]
            lines.append(
                f'{Ts[i]:5.0f} {Js[k]*1e-11:4.2g}  '
                + '  '.join(f'{c:>12s}' for c in cols)
                + f'   {r:6.2f}   {flip}')
    n_flip = sum(1 for i in range(Ts.size) for k in range(Js.size)
                 if len({names[c['dominant'][i, k]] for c in cases}) > 1)
    lines.append(f'\ncells with any class change across boxes: {n_flip}'
                 f' / {Ts.size*Js.size}')
    for p in range(len(cases)-1):
        d = int((cases[p]['dominant'] != cases[p+1]['dominant']).sum())
        lines.append(f'{labels[p]} -> {labels[p+1]}: {d} cells change')
        dp = np.abs(cases[p+1]['P_surv'] - cases[p]['P_surv'])
        lines.append(f'  max |dP_surv| = {np.nanmax(dp):.3f}, '
                     f'mean = {np.nanmean(dp):.3f}')
    with open(out_path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    print(f'Saved: {out_path}')
    print('\n'.join(lines))


# -----------------------------------------------------------------------------
def main():
    """Render the box-size comparison of the stability classification.

    Config is variables at the top of main() (no argparse).
    """
    # =========================== User Configuration =========================
    campaign_root = ('/Volumes/T7/skyrmion_simulator/output/'
                     'stochastic_llgs/scan_track_width/campaign')
    fig_root = 'output/figures_sllg/track_width'
    out_dir = 'output/figures_sllg/track_width/hk36_box_comparison'
    tags = ['hk36_D0p72_350x500',
            'hk36_D0p72_700x500',
            'hk36_D0p72_1400x500']
    labels = [r'$L_x=700$ nm', r'$L_x=1400$ nm', r'$L_x=2800$ nm']
    short = ['700 nm', '1400 nm', '2800 nm']
    styles = ['-', '--', ':']
    x_gate = 0.5               # D_x/L_x gate used by the classifier
    a = 2.0e-9                 # m, lattice constant
    do_config = True           # 36-panel field sheets (reads the dumps)
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    _set_style()
    cases = [_load_case(campaign_root, fig_root, t) for t in tags]
    for tag, case in zip(tags, cases):
        print(f'{tag}: L_x={case["L_x"]*1e9:.0f} nm '
              f'L_y={case["L_y"]*1e9:.0f} nm '
              f'n_ens={int(np.nansum(case["n_ens"]))}')
    _plot_maps(cases, labels, os.path.join(out_dir, 'stability_maps.png'))
    for tag, case, lab in zip(tags, cases, labels):
        _plot_composition_box(
            case, lab, os.path.join(out_dir, f'composition_{tag}.png'))
    _plot_flips(cases, short, os.path.join(out_dir, 'class_flips.png'))
    _plot_gate(cases, labels, x_gate,
               os.path.join(out_dir, 'gate_dx_lx.png'))
    _plot_invariance(cases, short, styles,
                     os.path.join(out_dir, 'box_invariance.png'))
    _plot_transport(cases, short, styles,
                    os.path.join(out_dir, 'transport.png'))
    _plot_equil_boxes(campaign_root, tags, labels,
                      os.path.join(out_dir, 'equil_D1D2_boxes.png'))
    _plot_velocity_boxes(cases, labels,
                         os.path.join(out_dir, 'velocity_boxes.png'))
    _plot_survival_hall_boxes(
        cases, labels, os.path.join(out_dir, 'survival_hall_boxes.png'))
    _write_summary(cases, short, x_gate,
                   os.path.join(out_dir, 'comparison_table.txt'))
    if do_config:
        for tag, case, lab in zip(tags, cases, labels):
            _plot_config_table(
                case, os.path.join(campaign_root, tag), a, x_gate,
                f'final driven top-layer $m_z$ (ens0) -- {lab}',
                os.path.join(out_dir, f'config_table_{tag}.png'))


# =============================================================================
if __name__ == '__main__':
    main()
