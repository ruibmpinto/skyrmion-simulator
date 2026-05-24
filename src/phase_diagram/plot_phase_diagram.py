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
# Local
from src.phase_diagram.axis_specs import (
    display_value,
    label_for,
)
from src.phase_diagram.params_helper import make_params
from src.simulator.energy import (
    critical_dmi,
    pma_anisotropy_field,
)

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
    """Load a generic 2D sweep NPZ.

    Required keys: `axis_x_name`, `axis_x_values`,
    `axis_y_name`, `axis_y_values`, `labels`, `ic_names`,
    `gs_label_idx`. Missing required keys raise.
    """
    raw = np.load(path, allow_pickle=True)
    data = {k: raw[k] for k in raw.files}
    required = {
        'axis_x_name', 'axis_x_values',
        'axis_y_name', 'axis_y_values',
        'labels', 'ic_names', 'gs_label_idx',
    }
    missing = required - set(data)
    if missing:
        raise RuntimeError(
            f'NPZ {path!r} is missing required keys: '
            f'{sorted(missing)}.'
        )
    data['labels'] = list(data['labels'])
    data['ic_names'] = list(data['ic_names'])
    data['axis_x_name'] = str(data['axis_x_name'])
    data['axis_y_name'] = str(data['axis_y_name'])
    return data


# ---------------------------------------------------------------------
def _axes_units(data, units):
    """Return (x_arr, y_arr, x_label, y_label, title_tag).

    Two unit modes:

    - `units='absolute'`: axis values multiplied by the
      `display_scale` from `axis_specs.axes`; labels taken
      from the same registry. Works for any axis pair in
      the registry.
    - `units='reduced'`: supported only when the axes are
      exactly `('D', 'H_z')` and `fixed_overrides` carried
      a fixed K so that D_c, H_K are well-defined scalars.
      Other axis pairs raise.

    Any other value of `units` raises.
    """
    x_name = data['axis_x_name']
    y_name = data['axis_y_name']
    x_si = np.asarray(data['axis_x_values'])
    y_si = np.asarray(data['axis_y_values'])
    if units == 'absolute':
        return (
            np.array([display_value(x_name, v) for v in x_si]),
            np.array([display_value(y_name, v) for v in y_si]),
            label_for(x_name),
            label_for(y_name),
            '',
        )
    if units == 'reduced':
        if (x_name, y_name) != ('D', 'H_z'):
            raise RuntimeError(
                f"units='reduced' is only defined for "
                f"axes ('D', 'H_z'); got "
                f"({x_name!r}, {y_name!r}).  Use "
                f"units='absolute'."
            )
        # Rebuild D_c, H_K from the K stored in the NPZ.
        # NPZ must carry K_top_probe and K_bot_probe so we
        # can reconstruct the material consistently.
        for key in ('K_top_probe', 'K_bot_probe'):
            if key not in data:
                raise RuntimeError(
                    f"Reduced units require {key!r} in the "
                    f"NPZ. Re-run the sweep with the "
                    f"current sweep.py."
                )
        p = make_params(
            K_top=float(data['K_top_probe']),
            K_bot=float(data['K_bot_probe']),
        )
        D_c = critical_dmi(p)
        H_K = pma_anisotropy_field(p)
        if D_c <= 0.0 or H_K <= 0.0:
            raise RuntimeError(
                f'Reduced units require positive D_c, H_K; '
                f'got D_c={D_c}, H_K={H_K}.'
            )
        return (
            x_si / D_c, y_si / H_K,
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
    """Return an (n_y, n_x) array of label strings.

    Raises if any `gs_label_idx` value is out of range:
    the sweep must always set gs_label_idx >= 0 for cells
    where at least one IC converged, and the per-IC label
    indices must come from `data['labels']`.
    """
    labels = data['labels']
    gs_idx = data['gs_label_idx']  # (n_x, n_y)
    out = np.empty(gs_idx.shape, dtype=object)
    for i in range(gs_idx.shape[0]):
        for j in range(gs_idx.shape[1]):
            k = int(gs_idx[i, j])
            if k < -1 or k >= len(labels):
                raise RuntimeError(
                    f'gs_label_idx[{i},{j}] = {k} is out '
                    f'of range for labels of length '
                    f'{len(labels)}.'
                )
            out[i, j] = (
                labels[k] if k >= 0 else 'undetermined'
            )
    return out.T  # transpose so y is row, x is column


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
    Q_abs = np.abs(Q_gs)
    if np.isnan(Q_abs).all():
        Q_max = 1.0
    else:
        Q_max = float(np.nanmax(Q_abs)) or 1.0
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
    n_x = len(data['axis_x_values'])
    n_y = len(data['axis_y_values'])
    i_idx = np.linspace(0, n_x - 1, n_per_axis, dtype=int)
    j_idx = np.linspace(0, n_y - 1, n_per_axis, dtype=int)
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
    x_si, y_si, x_label, y_label, _ = _axes_units(data, units)
    for ax, (i, j) in zip(axes.ravel(), points):
        m_top = data['gs_m_top'][i, j]
        if m_top.shape[0] == 0:
            ax.axis('off')
            continue
        ax.imshow(
            m_top[..., 2], cmap='RdBu_r',
            vmin=-1.0, vmax=1.0, origin='lower',
        )
        title_xy = (
            f'{x_label}={x_si[i]:.2f}, '
            f'{y_label}={y_si[j]:+.2f}'
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
    in_path = 'output/phase_diagram/D_H_z.npz'
    grid_name = None
    out_dir = None
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
