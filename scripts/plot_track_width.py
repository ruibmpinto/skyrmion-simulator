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
import matplotlib.animation as manim
import matplotlib.colors as mcolors
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


def _load_aggregate(in_dir):
    """Load the aggregate NPZ as a plain dict of arrays."""
    path = os.path.join(in_dir, 'aggregate.npz')
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
    """Survival probability and Hall-angle maps."""
    Ts = agg['Ts']
    Js = agg['Js']
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))
    im0 = _imshow_grid(
        axes[0], Ts, Js, agg['P_surv'],
        r'survival probability $P_{\mathrm{surv}}$',
        'magma', 0.0, 1.0)
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)
    theta = agg['theta_mean']
    tmax = float(np.nanmax(np.abs(theta))) if np.any(
        np.isfinite(theta)) else 1.0
    im1 = _imshow_grid(
        axes[1], Ts, Js, theta,
        r'skyrmion Hall angle $\theta_H$ (deg)',
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
        Keys `mi_top`, `mr_top`, `mf_top` (initial / relaxed /
        final (ny, nx, 3) top-layer fields), `mz_top` (drive-phase
        m_z stack, shape (n_frames, ny, nx)) and `t` (frame times).
    """
    d = np.load(path, allow_pickle=True)
    if 'anim_mz_top' in d.files:
        # Python driver layout.
        return {
            'mi_top': d['field_m_initial_top'],
            'mr_top': d['field_m_relaxed_top'],
            'mf_top': d['field_m_final_top'],
            'mz_top': d['anim_mz_top'],
            't': d['anim_t'],
        }
    if 'm_top' in d.files and 'phase_id' in d.files:
        # C++ SnapshotBuffer layout.
        m_top = d['m_top']
        phase = d['phase_id']
        t = d['time_s']
        relax_idx = np.flatnonzero(phase == 0)
        drive_idx = np.flatnonzero(phase == 1)
        if relax_idx.size == 0 or drive_idx.size == 0:
            raise RuntimeError(
                f'_load_anim: {path!r} lacks both relax and '
                f'drive frames (phase_id).')
        return {
            'mi_top': m_top[relax_idx[0]],
            'mr_top': m_top[relax_idx[-1]],
            'mf_top': m_top[drive_idx[-1]],
            'mz_top': m_top[drive_idx][..., 2],
            't': t[drive_idx],
        }
    raise RuntimeError(
        f'_load_anim: unrecognised dump layout in {path!r}; '
        f'keys={list(d.files)}.')


# -----------------------------------------------------------------------------
def _plot_three_configs(na, a, out_path):
    """Three-panel initial / relaxed / final m_z still (top layer)."""
    fields = (na['mi_top'], na['mr_top'], na['mf_top'])
    titles = ('initial', 'relaxed', 'final')
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 5.0))
    for ax, field, title in zip(axes, fields, titles):
        plot_snapshot_mz(
            field, a, ax=ax, title=title, show_colorbar=False)
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
    in_dir          = 'output/stochastic_llgs/scan_track_width'
    out_dir         = 'output/figures_sllg/track_width'
    a               = 2.0e-9         # m, lattice constant
    # Animation / still: pick one dumped cell by its file glob.
    anim_glob       = 'anim_T*.npz'
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Aggregate-derived grid figures.
    agg = _load_aggregate(in_dir)
    _plot_axes_vs_j(
        agg, os.path.join(out_dir, 'axes_vs_J.png'))
    _plot_axes_vs_t(
        agg, os.path.join(out_dir, 'axes_vs_T.png'))
    _plot_relaxed_vs_t(
        agg, os.path.join(out_dir, 'relaxed_vs_T.png'))
    _plot_ratio_heatmaps(
        agg, os.path.join(out_dir, 'ratio_heatmaps.png'))
    _plot_survival(
        agg, os.path.join(out_dir, 'survival_hall.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Full-field still and animation for one dumped realization.
    anim_files = sorted(glob.glob(os.path.join(in_dir, anim_glob)))
    if not anim_files:
        raise RuntimeError(
            f'plot_track_width: no {anim_glob!r} dump files in '
            f'{in_dir!r}; dumping is gated to ens_idx==0.')
    anim_path = anim_files[0]
    na = _load_anim(anim_path)
    stem = os.path.splitext(os.path.basename(anim_path))[0]
    _plot_three_configs(
        na, a,
        os.path.join(out_dir, f'{stem}_configs.png'))
    _build_animation(
        na, a,
        os.path.join(out_dir, f'{stem}.gif'))


# =============================================================================
if __name__ == '__main__':
    main()
