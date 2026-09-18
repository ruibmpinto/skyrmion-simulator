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
import zipfile
# Third-party
import numpy as np
import matplotlib.animation as manim
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Local
from src.plots.plot_snapshot_mz import plot_snapshot_mz
from src.stochastic_llgs.diagnostics import hall_angle, \
    unwrap_trajectory
from src.stochastic_llgs.stability import classify_field

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
        Per-cell metrics from `classify_field`.
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
def _plot_stability_map(agg, out_path):
    """Ensemble skyrmion stability regime diagram over the (T, J) grid.

    Each cell is coloured by its dominant class over all realizations
    -- the argmax of the four ensemble class counts, kept distinct
    (compact skyrmion, elongated, labyrinth, annihilated) -- and
    annotated with the survival probability
    P_surv = (n_compact + n_elongated)/n_ens. Compact and elongated
    are separate phases and are never merged here; the per-run split is
    shown in the class-composition figure. Cells with no realizations
    are left white.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js, n_S, n_E, n_L, n_A, P_surv).
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    n_cmp = np.asarray(agg['n_S'], float)
    n_elo = np.asarray(agg['n_E'], float)
    n_lab = np.asarray(agg['n_L'], float)
    n_ann = np.asarray(agg['n_A'], float)
    p_surv = np.asarray(agg['P_surv'], float)
    # Dominant class per cell among the four distinct phases:
    # 0 = compact, 1 = elongated, 2 = labyrinth, 3 = annihilated,
    # 4 = no data.
    stacks = np.stack([n_cmp, n_elo, n_lab, n_ann], axis=-1)
    total = stacks.sum(axis=-1)
    z = np.where(total > 0, np.argmax(stacks, axis=-1), 4).astype(float)
    cmap = mcolors.ListedColormap(
        ['#2c7bb6', '#fdae61', '#d7191c', '#999999', '#ffffff'])
    norm = mcolors.BoundaryNorm(
        [-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], cmap.N)
    fig, ax = plt.subplots(figsize=(7.8, 5.0))
    ax.imshow(z, origin='lower', aspect='auto', cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(Js.size))
    ax.set_xticklabels([f'{j*1e-11:.2g}' for j in Js])
    ax.set_yticks(np.arange(Ts.size))
    ax.set_yticklabels([f'{t:.0f}' for t in Ts])
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel('T (K)')
    ax.set_title(r'stability regimes (ensemble; cell = $P_{\mathrm{surv}}$)')
    # Annotate each cell with its ensemble survival probability.
    for i in range(Ts.size):
        for k in range(Js.size):
            c = 'k' if z[i, k] == 4 else 'w'
            txt = (f'{p_surv[i, k]:.2f}'
                   if np.isfinite(p_surv[i, k]) else '-')
            ax.text(k, i, txt, ha='center', va='center',
                    color=c, fontsize=8)
    handles = [
        mpatches.Patch(color='#2c7bb6', label='compact skyrmion'),
        mpatches.Patch(color='#fdae61', label='elongated'),
        mpatches.Patch(color='#d7191c', label='labyrinth'),
        mpatches.Patch(color='#999999', label='annihilated')]
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
        Per-cell class codes from the field classifier.
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
def _plot_survival(agg, out_path):
    """Ensemble survival-probability and Hall-angle maps over (T, J).

    Uses the aggregate's per-cell ensemble statistics computed over
    every realization (not just the ens0 dump):

    - `P_surv` = (n_S + n_E) / n_ens, the fraction of members whose
      final field classifies as a skyrmion ('S' or 'E').
    - `theta_mean` +/- `theta_se`: the skyrmion-Hall angle mean and
      standard error over the surviving members (the aggregator gates
      v/theta on survival). Cells with no survivors are NaN and blank.

    Three panels: survival probability, Hall-angle mean, Hall-angle
    standard error.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js, P_surv, theta_mean, theta_se).
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    p_surv = np.asarray(agg['P_surv'], dtype=float)
    th = np.asarray(agg['theta_mean'], dtype=float)
    th_se = np.asarray(agg['theta_se'], dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.0))
    im0 = _imshow_grid(
        axes[0], Ts, Js, p_surv,
        'survival probability $P_{\\mathrm{surv}}$ (all ens)',
        'magma', 0.0, 1.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    tmax = float(np.nanmax(np.abs(th))) if np.any(
        np.isfinite(th)) else 1.0
    im1 = _imshow_grid(
        axes[1], Ts, Js, th,
        r'Hall angle $\overline{\theta}_H$ (deg, ens mean)',
        'coolwarm', -tmax, tmax)
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)
    se_max = float(np.nanmax(th_se)) if np.any(
        np.isfinite(th_se)) else 1.0
    im2 = _imshow_grid(
        axes[2], Ts, Js, th_se,
        r'Hall angle SE $\sigma_{\theta_H}/\sqrt{n}$ (deg)',
        'viridis', 0.0, se_max)
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_class_composition(agg, out_path):
    """Per-cell ensemble class fractions over the (T, J) grid.

    Four heatmaps -- the fraction of the n_ens realizations ending in
    each class (S compact, E elongated, L labyrinth, A annihilated),
    each cell annotated with that fraction. Reveals cells whose
    outcome is not deterministic across runs (e.g. a 50/50 split
    between elongated and labyrinth at the same T and J), which the
    dominant-class stability map collapses to a single label.

    Parameters
    ----------
    agg : dict
        Aggregate payload (keys Ts, Js, n_ens, n_S, n_E, n_L, n_A).
    out_path : str
        Output PNG path.
    """
    Ts = agg['Ts']
    Js = agg['Js']
    n = np.asarray(agg['n_ens'], float)
    n = np.where(n > 0, n, np.nan)
    fracs = {
        'S': np.asarray(agg['n_S'], float) / n,
        'E': np.asarray(agg['n_E'], float) / n,
        'L': np.asarray(agg['n_L'], float) / n,
        'A': np.asarray(agg['n_A'], float) / n,
    }
    titles = {'S': 'compact skyrmion', 'E': 'elongated',
              'L': 'labyrinth', 'A': 'annihilated'}
    fig, axes = plt.subplots(1, 4, figsize=(22.0, 5.0))
    for ax, code in zip(axes, ('S', 'E', 'L', 'A')):
        z = fracs[code]
        im = _imshow_grid(
            ax, Ts, Js, z,
            f'P({code}) -- {titles[code]}', 'viridis', 0.0, 1.0)
        for i in range(Ts.size):
            for k in range(Js.size):
                v = z[i, k]
                if np.isfinite(v) and v > 0.0:
                    ax.text(k, i, f'{v:.2f}', ha='center', va='center',
                            color=('w' if v < 0.6 else 'k'), fontsize=6)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _try_load_anim(path):
    """`_load_anim` guarded against empty / corrupt dumps.

    Returns the payload dict, or None (with a warning) if the file is
    zero-byte or fails to load. The ens0 field dump can be empty when
    the driver's snapshot write ran out of memory on a large box; the
    trajectories and aggregate are unaffected, so the ensemble figures
    still render and only this cell's ens0 stills / GIF are skipped.
    """
    if os.path.getsize(path) == 0:
        print(f'  WARN skip empty anim: {os.path.basename(path)}')
        return None
    try:
        return _load_anim(path)
    except (EOFError, OSError, ValueError, zipfile.BadZipFile,
            KeyError) as exc:
        # zipfile.BadZipFile: nonzero but truncated / corrupt npz
        # (e.g. an interrupted transfer); KeyError: a missing array.
        print(f'  WARN skip unreadable anim '
              f'{os.path.basename(path)}: {type(exc).__name__}')
        return None


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
        and `t` (frame times). C++ dumps additionally carry
        `mm_top` (mid-drive frame, start of the Hall fit window),
        `q_final` and the tracked centre series `cx`, `cy`.
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
        if 'cx_top' not in d.files or 'cy_top' not in d.files:
            raise RuntimeError(
                f'_load_anim: {path!r} lacks the cx_top/cy_top '
                f'series needed for the dump-based Hall fit.')
        q_final = float(np.asarray(d['Q_top'])[drive_idx[-1]])
        # Mid-drive frame: the start of the last-half Hall fit window
        # (hall_angle uses i0 = int(0.5 * n)).
        mid = drive_idx[int(0.5 * drive_idx.size)]
        return {
            'mi_top': m_top[relax_idx[0]],
            'mr_top': m_top[relax_idx[-1]],
            'mm_top': m_top[mid],
            'mf_top': m_top[drive_idx[-1]],
            'mi_bot': m_bot[relax_idx[0]],
            'mr_bot': m_bot[relax_idx[-1]],
            'mf_bot': m_bot[drive_idx[-1]],
            'mz_top': m_top[drive_idx][..., 2],
            't': t[drive_idx],
            'q_final': q_final,
            'cx': np.asarray(d['cx_top'], dtype=float)[drive_idx],
            'cy': np.asarray(d['cy_top'], dtype=float)[drive_idx],
        }
    raise RuntimeError(
        f'_load_anim: unrecognised dump layout in {path!r}; '
        f'keys={list(d.files)}.')


# -----------------------------------------------------------------------------
def _hall_from_dump(na, L_x, L_y, periodic_y):
    """Hall angle of one ens0 dump trajectory.

    Same dump-based route as the stability classification: the
    (T, j) cell is characterised directly from its `anim_*.npz`
    realization instead of the aggregate, using the identical
    unwrap + last-half linear fit as the production diagnostics.

    Parameters
    ----------
    na : dict
        `_load_anim` payload; requires the drive-phase tracked
        centre series `cx`, `cy` and the frame times `t`.
    L_x : float
        Box length in x, in meters.
    L_y : float
        Box length in y, in meters.
    periodic_y : bool
        Whether y is periodic (False for the free-y racetrack).

    Returns
    -------
    theta_deg : float
        Hall angle atan2(v_y, |v_x|) in degrees, range (-90, 90];
        NaN when the centre series is unusable (core lost).
    """
    if 'cx' not in na or 'cy' not in na:
        raise RuntimeError(
            '_hall_from_dump: dump payload lacks the tracked '
            'centre series (Python-driver layout?); the '
            'dump-based Hall fit needs the C++ SnapshotBuffer '
            'cx_top/cy_top arrays.')
    cx = na['cx']
    cy = na['cy']
    if cx.size < 4 or not (np.all(np.isfinite(cx))
                           and np.all(np.isfinite(cy))):
        return float('nan')
    cx_u, cy_u = unwrap_trajectory(cx, cy, L_x, L_y, periodic_y)
    _vx, _vy, theta_deg = hall_angle(na['t'], cx_u, cy_u, half=0.5)
    return theta_deg


# -----------------------------------------------------------------------------
def _plot_hall_deviation(na, a, L_x, theta_deg, theta_se, out_path):
    """Three-panel visualization of the Hall-angle deviation.

    Left: drive-start (relaxed) configuration with the drive
    direction (J along +x) and the tracked centre. Middle:
    mid-drive frame -- the first point of the last-half Hall fit
    window -- with that point ringed. Right: final frame with the
    final tracked point, the full centre path (broken at
    periodic-x wraps), the zero-deflection reference line and the
    fit-window chord, annotated with theta_H (+- SE when an
    ensemble value is available).

    Parameters
    ----------
    na : dict
        `_load_anim` payload (C++ layout: needs `mr_top`,
        `mm_top`, `mf_top`, `cx`, `cy`).
    a : float
        Lattice constant (m).
    L_x : float
        Track length along the periodic axis (m), for the
        wrap-break of the path overlay.
    theta_deg : float
        Ensemble-mean Hall angle over all surviving runs (deg); the
        single realization drawn here is only the visual context.
    theta_se : float
        Ensemble standard error on theta (deg); NaN when no
        ensemble aggregate is available yet.
    out_path : str
        Output PNG path.
    """
    cx = na['cx'] * 1e9
    cy = na['cy'] * 1e9
    i0 = int(0.5 * cx.size)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 7.5))
    panels = (
        (na['mr_top'], 'drive start (relaxed)', 0),
        (na['mm_top'], 'mid-drive (fit-window start)', i0),
        (na['mf_top'], 'final', cx.size - 1),
    )
    for ax, (field, title, idx) in zip(axes, panels):
        plot_snapshot_mz(
            field, a, ax=ax, title=title, show_colorbar=False)
        ax.plot(cx[idx], cy[idx], marker='o', ms=12, mfc='none',
                mec='lime', mew=2.0)
        ax.plot(cx[idx], cy[idx], marker='+', ms=8, color='lime',
                mew=1.5)
    # Drive direction: current along +x.
    axes[0].annotate(
        '', xy=(0.38, 0.88), xytext=(0.08, 0.88),
        xycoords='axes fraction',
        arrowprops={'arrowstyle': '->', 'color': 'k', 'lw': 2.0})
    axes[0].text(0.23, 0.905, r'$J \parallel x$', ha='center',
                 fontsize=16, transform=axes[0].transAxes)
    # Path overlay on the final panel, broken at periodic-x wraps.
    cx_path = cx.copy()
    wrap = np.flatnonzero(np.abs(np.diff(cx_path)) > 0.5 * L_x * 1e9)
    cx_path[wrap + 1] = np.nan
    axes[2].plot(cx_path, cy, color='lime', lw=1.0, alpha=0.8)
    # Zero-deflection reference and the fit-window chord.
    axes[2].axhline(cy[i0], ls='--', color='k', lw=1.0)
    axes[2].plot([cx[i0], cx[-1]], [cy[i0], cy[-1]],
                 color='k', lw=1.5)
    se_txt = f'{theta_se:.1f}' if np.isfinite(theta_se) else 'n/a'
    axes[2].text(
        0.03, 0.985,
        rf'$\overline{{\theta}}_H = {theta_deg:+.1f}^\circ \pm$ '
        rf'{se_txt}$^\circ$ (ens)',
        transform=axes[2].transAxes, va='top', fontsize=16,
        bbox={'facecolor': 'w', 'alpha': 0.8, 'edgecolor': 'none'})
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved: {out_path}')
    plt.close(fig)


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
def _plot_equil_series(case_dir, out_path):
    """Thermal-equilibration D1(t) / D2(t), one color per T_sub.

    Reads every `equil_series_T*_ens*.npz` (keys `step`, `t_s`,
    `D1_m`, `D2_m`) written by stage 2 (equilibrate_track_width) in
    `case_dir`, groups them by substrate temperature, and plots the
    ensemble-mean axes versus time with a +/-1 sigma band. The
    per-ensemble series are ragged (each stops at its own plateau),
    so within each T they are truncated to the common (shortest)
    length before averaging -- the window every member shares.

    Parameters
    ----------
    case_dir : str
        Campaign case directory holding the equil_series files.
    out_path : str
        Output PNG path.
    """
    paths = sorted(glob.glob(
        os.path.join(case_dir, 'equil_series_T*_ens*.npz')))
    if not paths:
        raise RuntimeError(
            f'_plot_equil_series: no equil_series_T*_ens*.npz in '
            f'{case_dir!r}.')
    # Group file paths by the T token in the filename.
    by_t = {}
    for path in paths:
        base = os.path.basename(path)
        t_tok = base.split('_T')[1].split('_ens')[0]
        by_t.setdefault(float(t_tok), []).append(path)
    t_values = sorted(by_t)
    cmap = plt.get_cmap('viridis')
    fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True, figsize=(11, 5.4))
    for k, t_sub in enumerate(t_values):
        series = [np.load(p) for p in by_t[t_sub]]
        # Common (shortest) length so the ens mean is well-defined.
        n = min(len(s['t_s']) for s in series)
        t_ns = series[0]['t_s'][:n] * 1e9
        d1 = np.stack([s['D1_m'][:n] for s in series]) * 1e9
        d2 = np.stack([s['D2_m'][:n] for s in series]) * 1e9
        c = cmap(k / max(len(t_values) - 1, 1))
        label = f'$T = {t_sub:.0f}$ K ({len(series)} ens)'
        for ax, d in ((ax1, d1), (ax2, d2)):
            mean = d.mean(axis=0)
            std = d.std(axis=0)
            ax.plot(t_ns, mean, color=c, label=label)
            ax.fill_between(
                t_ns, mean - std, mean + std, color=c, alpha=0.2)
    ax1.set_ylabel(r'$D_1$ (nm)')
    ax2.set_ylabel(r'$D_2$ (nm)')
    for ax in (ax1, ax2):
        ax.set_xlabel(r'$t$ (ns)')
        ax.set_box_aspect(1)
    fig.suptitle(os.path.basename(case_dir.rstrip('/'))
                 + r' -- thermal equilibration')
    ax1.legend(loc='best', frameon=False, ncol=2, fontsize=9)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f'Saved {out_path}')


# -----------------------------------------------------------------------------
def _postprocess_case(case_dir, out_dir, gif_dir, a, anim_glob,
                      do_equil, do_drive, do_gif, overwrite):
    """Post-process one campaign case dir (<hk>_<D>_<box>).

    Reads `aggregate.npz`, `anim_T*.npz`, and `equil_series_*` from
    `case_dir` (on T7 with the large dumps), writing small figures to
    `out_dir` and GIFs to `gif_dir`. Reuses the module's plot helpers.
    """
    os.makedirs(out_dir, exist_ok=True)
    # --- Thermal equilibration D1/D2 vs t (stage 2) ---
    if do_equil and glob.glob(os.path.join(
            case_dir, 'equil_series_T*_ens*.npz')):
        _plot_equil_series(
            case_dir, os.path.join(out_dir, 'equil_D1D2_vs_t.png'))
    if not (do_drive or do_gif):
        return
    agg_path = os.path.join(case_dir, 'aggregate.npz')
    if not os.path.isfile(agg_path):
        print(f'  no aggregate.npz in {case_dir}; skip drive/gif.')
        return
    agg = _load_aggregate(case_dir, 'aggregate.npz')
    # --- Drive-phase aggregate grid figures ---
    if do_drive:
        _plot_axes_vs_j(agg, os.path.join(out_dir, 'axes_vs_J.png'))
        _plot_axes_vs_t(agg, os.path.join(out_dir, 'axes_vs_T.png'))
        _plot_relaxed_vs_t(
            agg, os.path.join(out_dir, 'relaxed_vs_T.png'))
        _plot_ratio_heatmaps(
            agg, os.path.join(out_dir, 'ratio_heatmaps.png'))
    anim_files = sorted(glob.glob(os.path.join(case_dir, anim_glob)))
    if not anim_files:
        print(f'  no {anim_glob} in {case_dir}; skip classify/gif.')
        return
    Ts = agg['Ts']
    Js = agg['Js']
    L_x = float(agg['L_x'])
    L_y = float(agg['L_y'])
    D1m = agg['D1_mean']
    D2m = agg['D2_mean']
    # Pass 1: classify each cell + Hall angle from the ens0 dump.
    cls = np.full((Ts.size, Js.size), '?', dtype='<U1')
    th_ens0 = np.full((Ts.size, Js.size), np.nan)
    rows = []
    for anim_path in anim_files:
        stem = os.path.splitext(os.path.basename(anim_path))[0]
        T = float(stem.split('_T')[1].split('_j')[0])
        J = float(stem.split('_j')[1])
        i = int(np.argmin(np.abs(Ts - T)))
        k = int(np.argmin(np.abs(Js - J)))
        na = _try_load_anim(anim_path)
        if na is None:
            continue
        code, m = classify_field(
            na['mf_top'][..., 2], abs(na['q_final']),
            float(D1m[i, k]), float(D2m[i, k]), L_x)
        cls[i, k] = code
        # Racetrack: y is free (not periodic), x is periodic.
        th_ens0[i, k] = _hall_from_dump(na, L_x, L_y, periodic_y=False)
        rows.append({'T': T, 'J': J, 'code': code, **m})
        print(f"  {stem}: |Q|={m['q_abs']:.2f} D1/D2={m['ratio']:.2f} "
              f"theta={th_ens0[i, k]:+.1f} -> {code}")
    tag = os.path.basename(out_dir.rstrip('/'))
    if do_drive:
        # Keep the ens0 class table as a per-realization record.
        _save_classes(
            os.path.join(out_dir, f'stability_classes_{tag}.npz'),
            Ts, Js, cls, rows)
        # Ensemble-based maps (all realizations, from the aggregate).
        _plot_stability_map(
            agg, os.path.join(out_dir, 'stability_map.png'))
        _plot_class_composition(
            agg, os.path.join(out_dir, 'class_composition.png'))
        _plot_velocity_vs_j(
            agg, cls, os.path.join(out_dir, 'velocity_vs_J.png'))
        _plot_survival(
            agg, os.path.join(out_dir, 'survival_hall.png'))
    # Pass 2: config stills, Hall stills (S/E cells) + GIFs.
    if do_gif:
        os.makedirs(gif_dir, exist_ok=True)
        for anim_path in anim_files:
            stem = os.path.splitext(os.path.basename(anim_path))[0]
            T = float(stem.split('_T')[1].split('_j')[0])
            J = float(stem.split('_j')[1])
            i = int(np.argmin(np.abs(Ts - T)))
            k = int(np.argmin(np.abs(Js - J)))
            cfg_path = os.path.join(out_dir, f'{stem}_configs.png')
            hall_path = os.path.join(out_dir, f'{stem}_hall.png')
            gif_path = os.path.join(gif_dir, f'{stem}.gif')
            need_cfg = overwrite or not os.path.isfile(cfg_path)
            need_hall = cls[i, k] in ('S', 'E') and (
                overwrite or not os.path.isfile(hall_path))
            need_gif = overwrite or not os.path.isfile(gif_path)
            if not need_cfg and not need_hall and not need_gif:
                print(f'  skip (exists): {stem}')
                continue
            na = _try_load_anim(anim_path)
            if na is None:
                continue
            if need_cfg:
                _plot_three_configs(na, a, cfg_path)
            if need_hall:
                # Annotate with the ensemble Hall angle mean +/- SE
                # (from the aggregate over all surviving runs), not the
                # single ens0 realization drawn as visual context.
                _plot_hall_deviation(
                    na, a, L_x,
                    float(agg['theta_mean'][i, k]),
                    float(agg['theta_se'][i, k]),
                    hall_path)
            if need_gif:
                _build_animation(na, a, gif_path)


# -----------------------------------------------------------------------------
def main():
    """Post-process every pulled track-width campaign case: thermal
    D1/D2-vs-t, drive-phase aggregate panels, stability / velocity /
    survival maps, and per-cell config stills + GIFs. Config is
    variables at the top of main() (no argparse).
    """
    # =========================== User Configuration =========================
    # Campaign case dirs (aggregate.npz + anim_*.npz + equil_series)
    # live on T7 with the large dumps; each subdir is one
    # <tag> = <hk>_<D>_<box>. Small figures go to the local fig root;
    # GIFs go beside the dumps on T7.
    campaign_root = ('/Volumes/T7/skyrmion_simulator/output/'
                     'stochastic_llgs/scan_track_width/campaign')
    fig_root_local = 'output/figures_sllg/track_width'
    fig_root_t7 = ('/Volumes/T7/skyrmion_simulator/output/'
                   'figures_sllg/track_width')
    a = 3.0e-9                 # m, lattice constant
    anim_glob = 'anim_T*.npz'
    # Stage toggles.
    do_equil = False           # thermal D1/D2-vs-t (needs equil_series)
    do_drive = True            # aggregate panels + stability maps
    do_gif = False             # per-cell config stills + GIFs (slow)
    overwrite = False          # re-render existing stills / GIFs
    # Restrict to these tags; empty => every case dir present with an
    # aggregate.npz (skips cases still running on the cluster). The
    # 350x500 / 700x500 cases are already processed, so scope to the
    # remaining 1400x500 boxes.
    only_tags = ['hk36_D0p72_467x333_a3nm']
    # ======================= End User Configuration =========================
    case_dirs = sorted(
        d for d in glob.glob(os.path.join(campaign_root, '*'))
        if os.path.isdir(d))
    if only_tags:
        case_dirs = [d for d in case_dirs
                     if os.path.basename(d) in only_tags]
    if not case_dirs:
        raise RuntimeError(
            f'main: no campaign case dirs under {campaign_root!r}.')
    for case_dir in case_dirs:
        tag = os.path.basename(case_dir)
        print(f'===== {tag} =====')
        _postprocess_case(
            case_dir,
            os.path.join(fig_root_local, tag),
            os.path.join(fig_root_t7, tag),
            a, anim_glob, do_equil, do_drive, do_gif, overwrite)


# =============================================================================
if __name__ == '__main__':
    main()
