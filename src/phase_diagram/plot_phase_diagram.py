"""Visualization of the (D, H_z) phase-diagram NPZ.

Produces three figures from a sweep output:
    1. Color-coded phase map.
    2. Order-parameter overlays: <m_z> and Q.
    3. Representative ground-state spin textures.

Usage
-----
Edit the variables in the "User Configuration" block at the
top of `main()` (grid_name, in_path, out_dir, units), then:

    python -m src.phase_diagram.plot_phase_diagram

Functions
---------
load
    Load a sweep NPZ into a dict of arrays.
plot_phase_map
    Discrete-color (D, H_z) phase map.
plot_order_parameters
    Side-by-side <m_z> and Q ground-state contour panels.
plot_textures
    Grid of representative ground-state spin textures.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
# Third-party
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
_PHASE_COLORS = {
    'FM_anti': '#9e9e9e',
    'FM_par+': '#d73027',
    'FM_par-': '#4575b4',
    'iSk':     '#fdae61',
    'SkX':     '#fee090',
    'BX':      '#762a83',
    'SS':      '#74add1',
    'Lab':     '#a6d96a',
    'undetermined': '#bdbdbd',
}


def load(path):
    """Load a sweep NPZ produced by `sweep.sweep`."""
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    data['labels'] = list(data['labels'])
    data['ic_names'] = list(data['ic_names'])
    return data


# ---------------------------------------------------------------------
def _axes_units(data, units):
    """Return (x_arr, y_arr, x_label, y_label, title_tag).

    For `units='reduced'` the NPZ must contain `D_c` and
    `H_K` scalars (sweep.py writes them); otherwise a
    `RuntimeError` is raised so absent normalizations are
    not silently swapped for fallback absolute axes.
    """
    D = data['D']            # J/m^2
    H = data['H_z']          # T
    if units == 'absolute':
        return (
            D * 1.0e3, H,
            r'DMI strength $D$ (mJ/m$^2$)',
            r'External field $H_z$ (T)',
            '',
        )
    if units == 'reduced':
        if 'D_c' not in data or 'H_K' not in data:
            raise RuntimeError(
                'Reduced units requested but the NPZ does '
                'not store `D_c` and `H_K`. Re-run the '
                'sweep with the current `sweep.py`, or use '
                '`--units absolute`.'
            )
        D_c = float(data['D_c'])
        H_K = float(data['H_K'])
        if D_c <= 0.0 or H_K <= 0.0:
            raise RuntimeError(
                f'Reduced units require positive D_c and '
                f'H_K; got D_c={D_c}, H_K={H_K}. '
                f'(K_eff_avg <= 0?)'
            )
        return (
            D / D_c, H / H_K,
            r'$D / D_c$',
            r'$H_z / H_K$',
            (
                f'$D_c = {D_c * 1e3:.3f}$ mJ/m$^2$,  '
                f'$H_K = {H_K:.3f}$ T'
            ),
        )
    raise RuntimeError(
        f"Unknown units {units!r}; use 'reduced' or "
        f"'absolute'."
    )


# ---------------------------------------------------------------------
def _phase_label_grid(data):
    """Return an (n_H, n_D) array of label strings."""
    labels = data['labels']
    gs_idx = data['gs_label_idx']  # (n_D, n_H)
    out = np.empty(gs_idx.shape, dtype=object)
    for i in range(gs_idx.shape[0]):
        for j in range(gs_idx.shape[1]):
            k = int(gs_idx[i, j])
            out[i, j] = (
                labels[k] if 0 <= k < len(labels)
                else 'undetermined'
            )
    return out.T  # transpose so H is row, D is column


# ---------------------------------------------------------------------
def plot_phase_map(data, ax=None, units='reduced'):
    """Draw the discrete color phase map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.5))
    x, y, xlabel, ylabel, tag = _axes_units(data, units)
    label_grid = _phase_label_grid(data)
    # Build categorical colormap
    cats = list(_PHASE_COLORS.keys())
    cat_index = np.array(
        [[cats.index(label_grid[j, i])
          for i in range(label_grid.shape[1])]
         for j in range(label_grid.shape[0])]
    )
    cmap = mcolors.ListedColormap(
        [_PHASE_COLORS[c] for c in cats]
    )
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, len(cats) + 0.5, 1.0), cmap.N,
    )
    ax.pcolormesh(
        x, y, cat_index, cmap=cmap, norm=norm,
        shading='auto',
    )
    if units == 'reduced':
        ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    # Custom legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=_PHASE_COLORS[c])
        for c in cats
    ]
    ax.legend(
        handles, cats, loc='upper left',
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
        fontsize=9, frameon=False,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = 'SAF phase diagram'
    if tag:
        title = f'{title}\n{tag}'
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------
def _ground_state_field(data, name):
    """Return the (n_D, n_H) ground-state value of `name`."""
    full = data[name]  # (n_D, n_H, n_IC)
    gs = data['gs_idx']
    out = np.zeros(gs.shape)
    for i in range(gs.shape[0]):
        for j in range(gs.shape[1]):
            k = int(gs[i, j])
            if k < 0:
                out[i, j] = np.nan
            else:
                out[i, j] = full[i, j, k]
    return out


def plot_order_parameters(data, axes=None, units='reduced'):
    """Side-by-side <m_z> and Q ground-state heatmaps."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    x, y, xlabel, ylabel, _ = _axes_units(data, units)
    mz_gs = _ground_state_field(data, 'mz_top').T
    Q_gs = _ground_state_field(data, 'Q').T
    im0 = axes[0].pcolormesh(
        x, y, mz_gs, cmap='RdBu_r', vmin=-1.0, vmax=1.0,
        shading='auto',
    )
    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].set_title(r'Ground-state $\langle m_z \rangle$')
    if units == 'reduced':
        axes[0].axvline(1.0, color='k', lw=0.8, ls='--',
                        alpha=0.6)
    plt.colorbar(im0, ax=axes[0])
    Q_max = float(np.nanmax(np.abs(Q_gs))) or 1.0
    im1 = axes[1].pcolormesh(
        x, y, Q_gs, cmap='PuOr',
        vmin=-Q_max, vmax=Q_max, shading='auto',
    )
    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel)
    axes[1].set_title('Ground-state topological charge $Q$')
    if units == 'reduced':
        axes[1].axvline(1.0, color='k', lw=0.8, ls='--',
                        alpha=0.6)
    plt.colorbar(im1, ax=axes[1])
    return axes


# ---------------------------------------------------------------------
def _select_texture_points(data, n_per_axis=3):
    """Return a small (i, j) grid spanning the parameter space."""
    n_D = len(data['D'])
    n_H = len(data['H_z'])
    i_idx = np.linspace(0, n_D - 1, n_per_axis, dtype=int)
    j_idx = np.linspace(0, n_H - 1, n_per_axis, dtype=int)
    return [(int(i), int(j))
            for j in j_idx for i in i_idx]


def plot_textures(data, fig=None, n_per_axis=3,
                  units='reduced'):
    """Plot a grid of ground-state m_z textures."""
    points = _select_texture_points(data, n_per_axis)
    if fig is None:
        fig, axes = plt.subplots(
            n_per_axis, n_per_axis,
            figsize=(2.6 * n_per_axis, 2.6 * n_per_axis),
        )
    else:
        axes = fig.subplots(n_per_axis, n_per_axis)
    axes = np.atleast_2d(axes)
    labels = data['labels']
    if units == 'reduced':
        D_c = float(data['D_c'])
        H_K = float(data['H_K'])
    for ax, (i, j) in zip(axes.ravel(), points):
        m_top = data['gs_m_top'][i, j]
        if m_top.shape[0] == 0:
            ax.axis('off')
            continue
        ax.imshow(
            m_top[..., 2], cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, origin='lower',
        )
        if units == 'reduced':
            x_val = float(data['D'][i]) / D_c
            y_val = float(data['H_z'][j]) / H_K
            title_xy = (
                f'$D/D_c$={x_val:.2f}, '
                f'$H_z/H_K$={y_val:+.2f}'
            )
        else:
            x_val = float(data['D'][i]) * 1e3
            y_val = float(data['H_z'][j])
            title_xy = (
                f'D={x_val:.2f}, $H_z$={y_val:+.2f}'
            )
        gs_lab = labels[int(data['gs_label_idx'][i, j])]
        ax.set_title(
            f'{title_xy}\n{gs_lab}',
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
def render_all(npz_path, out_dir=None, units='reduced'):
    """Generate the three standard figures from an NPZ.

    Parameters
    ----------
    npz_path : str
        Path to a sweep NPZ.
    out_dir : str or None, default=None
        Output directory. Defaults to the NPZ's directory.
    units : {'reduced', 'absolute'}, default='reduced'
        Axis units. 'reduced' uses (D/D_c, H_z/H_K) and
        requires the NPZ to contain D_c and H_K (sweeps
        written before this feature do not; pass
        'absolute' for those).
    """
    data = load(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    if out_dir is None:
        out_dir = os.path.dirname(npz_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    suffix = f'_{units}'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_map, ax_map = plt.subplots(figsize=(7.0, 5.5))
    plot_phase_map(data, ax_map, units=units)
    fig_map.tight_layout()
    map_path = os.path.join(
        out_dir, f'{base}_phase_map{suffix}.png',
    )
    fig_map.savefig(map_path, dpi=150, bbox_inches='tight')
    plt.close(fig_map)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_op, axes_op = plt.subplots(1, 2, figsize=(11.0, 4.5))
    plot_order_parameters(data, axes_op, units=units)
    fig_op.tight_layout()
    op_path = os.path.join(
        out_dir, f'{base}_order_params{suffix}.png',
    )
    fig_op.savefig(op_path, dpi=150, bbox_inches='tight')
    plt.close(fig_op)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_tex = plt.figure(figsize=(8.0, 8.0))
    plot_textures(data, fig_tex, n_per_axis=3, units=units)
    tex_path = os.path.join(
        out_dir, f'{base}_textures{suffix}.png',
    )
    fig_tex.savefig(tex_path, dpi=150, bbox_inches='tight')
    plt.close(fig_tex)
    return [map_path, op_path, tex_path]


# ---------------------------------------------------------------------
def main():
    """Read the User Configuration block and render PNGs."""
    # ================ User Configuration ================
    # Source NPZ. Use either `in_path` (direct path) or
    # `grid_name` (loads output/phase_diagram/<grid>.npz);
    # exactly one must be non-None.
    in_path = None
    grid_name = 'medium'
    # Output directory. None = same directory as the NPZ.
    out_dir = None
    # Axis units: 'reduced' uses (D/D_c, H_z/H_K) and
    # requires D_c, H_K in the NPZ. 'absolute' uses
    # (D in mJ/m^2, H_z in T).
    units = 'reduced'
    # ============ End User Configuration =================
    if in_path is not None and grid_name is not None:
        raise RuntimeError(
            'Set exactly one of in_path or grid_name in '
            'main(); both are set.'
        )
    if in_path is not None:
        path = in_path
    elif grid_name is not None:
        path = os.path.join(
            'output', 'phase_diagram', f'{grid_name}.npz',
        )
    else:
        raise RuntimeError(
            'Set either grid_name or in_path in main().'
        )
    resolved_out_dir = out_dir or os.path.dirname(path) or '.'
    paths = render_all(
        path, out_dir=resolved_out_dir, units=units,
    )
    for p in paths:
        print(f'Wrote {p}')


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
