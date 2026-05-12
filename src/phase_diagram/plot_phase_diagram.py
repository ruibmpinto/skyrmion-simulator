"""Visualization of the (D, H_z) phase-diagram NPZ.

Produces three figures from a sweep output:
    1. Color-coded phase map.
    2. Order-parameter overlays: <m_z> and Q.
    3. Representative ground-state spin textures.

Usage
-----
    python -m src.phase_diagram.plot_phase_diagram coarse
    python -m src.phase_diagram.plot_phase_diagram \
        --in output/phase_diagram/test.npz

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
import argparse
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
    'FM+': '#d73027',
    'FM-': '#4575b4',
    'iSk': '#fdae61',
    'SkX': '#fee090',
    'BX':  '#762a83',
    'SS':  '#74add1',
    'Lab': '#a6d96a',
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
def plot_phase_map(data, ax=None):
    """Draw the discrete color phase map."""
    if ax is None:
        _, ax = plt.subplots(figsize=(7.0, 5.5))
    D = data['D'] * 1e3   # mJ/m^2
    H = data['H_z']       # Tesla
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
        D, H, cat_index, cmap=cmap, norm=norm,
        shading='auto',
    )
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
    ax.set_xlabel(r'DMI strength $D$ (mJ/m$^2$)')
    ax.set_ylabel(r'External field $H_z$ (T)')
    ax.set_title('SAF phase diagram')
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


def plot_order_parameters(data, axes=None):
    """Side-by-side <m_z> and Q ground-state heatmaps."""
    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    D = data['D'] * 1e3
    H = data['H_z']
    mz_gs = _ground_state_field(data, 'mz_top').T
    Q_gs = _ground_state_field(data, 'Q').T
    im0 = axes[0].pcolormesh(
        D, H, mz_gs, cmap='RdBu_r', vmin=-1.0, vmax=1.0,
        shading='auto',
    )
    axes[0].set_xlabel(r'$D$ (mJ/m$^2$)')
    axes[0].set_ylabel(r'$H_z$ (T)')
    axes[0].set_title(r'Ground-state $\langle m_z \rangle$')
    plt.colorbar(im0, ax=axes[0])
    Q_max = float(np.nanmax(np.abs(Q_gs))) or 1.0
    im1 = axes[1].pcolormesh(
        D, H, Q_gs, cmap='PuOr',
        vmin=-Q_max, vmax=Q_max, shading='auto',
    )
    axes[1].set_xlabel(r'$D$ (mJ/m$^2$)')
    axes[1].set_ylabel(r'$H_z$ (T)')
    axes[1].set_title('Ground-state topological charge $Q$')
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


def plot_textures(data, fig=None, n_per_axis=3):
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
    for ax, (i, j) in zip(axes.ravel(), points):
        m_top = data['gs_m_top'][i, j]
        if m_top.shape[0] == 0:
            ax.axis('off')
            continue
        ax.imshow(
            m_top[..., 2], cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, origin='lower',
        )
        D_val = float(data['D'][i]) * 1e3
        H_val = float(data['H_z'][j])
        gs_lab = labels[int(data['gs_label_idx'][i, j])]
        ax.set_title(
            f'D={D_val:.2f}, '
            f'$H_z$={H_val:+.2f}\n{gs_lab}',
            fontsize=9,
        )
        ax.set_xticks([])
        ax.set_yticks([])
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
def render_all(npz_path, out_dir=None):
    """Generate the three standard figures from an NPZ."""
    data = load(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    if out_dir is None:
        out_dir = os.path.dirname(npz_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_map, ax_map = plt.subplots(figsize=(7.0, 5.5))
    plot_phase_map(data, ax_map)
    fig_map.tight_layout()
    map_path = os.path.join(out_dir, f'{base}_phase_map.png')
    fig_map.savefig(map_path, dpi=150, bbox_inches='tight')
    plt.close(fig_map)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_op, axes_op = plt.subplots(1, 2, figsize=(11.0, 4.5))
    plot_order_parameters(data, axes_op)
    fig_op.tight_layout()
    op_path = os.path.join(out_dir, f'{base}_order_params.png')
    fig_op.savefig(op_path, dpi=150, bbox_inches='tight')
    plt.close(fig_op)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig_tex = plt.figure(figsize=(8.0, 8.0))
    plot_textures(data, fig_tex, n_per_axis=3)
    tex_path = os.path.join(out_dir, f'{base}_textures.png')
    fig_tex.savefig(tex_path, dpi=150, bbox_inches='tight')
    plt.close(fig_tex)
    return [map_path, op_path, tex_path]


# ---------------------------------------------------------------------
def _parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            'Render figures from a phase-diagram sweep NPZ.'
        ),
    )
    parser.add_argument(
        'grid', nargs='?', default=None,
        help=(
            'Grid name; default loads '
            'output/phase_diagram/<grid>.npz.'
        ),
    )
    parser.add_argument(
        '--in', dest='in_path', default=None,
        help='Direct path to the NPZ.',
    )
    parser.add_argument(
        '--out-dir', default=None,
        help='Output directory for PNGs.',
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = _parse_args(argv)
    if args.in_path is not None:
        path = args.in_path
    elif args.grid is not None:
        path = os.path.join(
            'output', 'phase_diagram', f'{args.grid}.npz',
        )
    else:
        raise RuntimeError(
            'Provide either a grid name or --in PATH.'
        )
    out_dir = args.out_dir or os.path.dirname(path) or '.'
    paths = render_all(path, out_dir=out_dir)
    for p in paths:
        print(f'Wrote {p}')


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
