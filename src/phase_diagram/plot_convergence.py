"""Diagnostic map of LLGS relaxation convergence per sweep cell.

Renders two panels from a sweep NPZ:
    1. Convergence fraction per (axis_x, axis_y) cell:
       fraction of the IC ensemble whose LLGS relaxation hit the
       (tol_torque, tol_dE) thresholds before max_steps.
    2. Ground-state initial condition per cell, categorical:
       which IC won the lowest-energy competition. Cells whose
       winning IC did not itself converge are marked with an
       overlaid black `x` so a slow/diverging winner cannot be
       confused with a genuinely settled ground state.

Usage
-----
Edit the User Configuration block at the top of `main()`
(`in_path`, `out_dir`, `units`), then:

    python -m src.phase_diagram.plot_convergence

Functions
---------
plot_convergence_fraction
    Per-cell fraction of converged ICs.
plot_ground_state_ic
    Per-cell winning IC, with non-converged overlays.
render
    Build the two-panel figure and write the PNG.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
# Third-party
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.phase_diagram.plot_phase_diagram import _axes_units, load

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
# Categorical palette for the IC map. Ordered to match the canonical
# IC list emitted by `sweep._ic_specs()`; extra names fall back to a
# qualitative tab20 cycle.
_IC_BASE_COLORS = {
    'random':         '#1f77b4',
    'fm_anti':        '#9e9e9e',
    'fm_par':         '#d62728',
    'skyrmion':       '#ff7f0e',
    'stripe':         '#2ca02c',
    'stripe_y':       '#8c564b',
    'sk_lattice':     '#e377c2',
    'bubble_lattice': '#bcbd22',
}


def _ic_color_map(ic_names):
    """Return an ordered list of distinct hex colors per IC name.

    Parameters
    ----------
    ic_names : list[str]
        IC names in the order they appear in the NPZ (length n_IC).

    Returns
    -------
    colors : list[str]
        Length n_IC. Recurring names are dimmed by index so two
        `random` ICs (different seeds) get different shades.
    """
    seen = {}
    out = []
    tab = plt.get_cmap('tab20').colors
    for k, name in enumerate(ic_names):
        n_prev = seen.get(name, 0)
        if name in _IC_BASE_COLORS and n_prev == 0:
            out.append(_IC_BASE_COLORS[name])
        elif name in _IC_BASE_COLORS:
            # Same family, later occurrence: shift hue via tab20.
            out.append(mcolors.to_hex(tab[(k + 10) % 20]))
        else:
            out.append(mcolors.to_hex(tab[k % 20]))
        seen[name] = n_prev + 1
    return out


# ---------------------------------------------------------------------
def plot_convergence_fraction(data, ax=None, units='reduced'):
    """Per-cell fraction of converged ICs.

    Parameters
    ----------
    data : dict
        Output of `plot_phase_diagram.load`.
    ax : matplotlib.axes.Axes or None, default=None
        Axes to draw on; created if None.
    units : {'reduced', 'absolute'}, default='reduced'
        Axis units, forwarded to `_axes_units`.

    Returns
    -------
    ax : matplotlib.axes.Axes
        The drawn axes.
    """
    if 'converged' not in data:
        raise RuntimeError(
            "NPZ does not contain 'converged' per-IC array; "
            "re-run the sweep with the current sweep.py."
        )
    conv = np.asarray(data['converged'])  # (n_x, n_y, n_IC)
    if conv.ndim != 3:
        raise RuntimeError(
            f"'converged' must be 3D (n_x, n_y, n_IC); got "
            f"shape {conv.shape}."
        )
    # Fraction of the IC ensemble that converged, per cell.
    frac = conv.mean(axis=2).T  # (n_y, n_x)
    if ax is None:
        _, ax = plt.subplots(figsize=(6.5, 5.0))
    x, y, xlabel, ylabel, tag = _axes_units(data, units)
    im = ax.pcolormesh(
        x, y, frac, cmap='viridis', vmin=0.0, vmax=1.0,
        shading='auto',
    )
    if units == 'reduced':
        ax.axvline(1.0, color='w', lw=0.8, ls='--', alpha=0.7)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Fraction of ICs converged')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = 'Convergence fraction'
    if tag:
        title = f'{title}\n{tag}'
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------
def plot_ground_state_ic(data, ax=None, units='reduced'):
    """Per-cell winning IC, categorical map.

    Cells whose winning IC failed the convergence criterion are
    marked with a black `x` so an unsettled minimum cannot be read
    as a clean ground state.
    """
    for key in ('gs_idx', 'ic_names', 'converged'):
        if key not in data:
            raise RuntimeError(
                f"NPZ is missing required key {key!r} for the "
                f"IC map."
            )
    gs_idx = np.asarray(data['gs_idx'])  # (n_x, n_y)
    ic_names = list(data['ic_names'])
    conv = np.asarray(data['converged'])  # (n_x, n_y, n_IC)
    n_ic = len(ic_names)
    if gs_idx.ndim != 2:
        raise RuntimeError(
            f"'gs_idx' must be 2D; got shape {gs_idx.shape}."
        )
    if conv.shape[:2] != gs_idx.shape or conv.shape[2] != n_ic:
        raise RuntimeError(
            f"'converged' shape {conv.shape} inconsistent with "
            f"gs_idx {gs_idx.shape} and n_IC={n_ic}."
        )
    if ax is None:
        _, ax = plt.subplots(figsize=(7.5, 5.0))
    x, y, xlabel, ylabel, tag = _axes_units(data, units)
    colors = _ic_color_map(ic_names)
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(
        np.arange(-0.5, n_ic + 0.5, 1.0), cmap.N,
    )
    # gs_idx is (n_x, n_y); pcolormesh wants (n_y, n_x).
    ax.pcolormesh(
        x, y, gs_idx.T, cmap=cmap, norm=norm, shading='auto',
    )
    # Mark cells whose ground-state IC did not converge.
    bad_x, bad_y = [], []
    for i in range(gs_idx.shape[0]):
        for j in range(gs_idx.shape[1]):
            k = int(gs_idx[i, j])
            if k < 0 or not bool(conv[i, j, k]):
                bad_x.append(x[i])
                bad_y.append(y[j])
    if bad_x:
        ax.plot(
            bad_x, bad_y, marker='x', ls='', color='k',
            ms=4.0, mew=0.8,
            label='ground-state IC unconverged',
        )
    if units == 'reduced':
        ax.axvline(1.0, color='k', lw=0.8, ls='--', alpha=0.6)
    # Legend: one swatch per distinct (name, index) pair.
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=colors[k])
        for k in range(n_ic)
    ]
    legend_labels = [f'{k}: {ic_names[k]}' for k in range(n_ic)]
    if bad_x:
        handles.append(
            plt.Line2D(
                [0], [0], marker='x', ls='', color='k',
                ms=6.0, mew=1.0,
            )
        )
        legend_labels.append('unconverged GS')
    ax.legend(
        handles, legend_labels, loc='upper left',
        bbox_to_anchor=(1.02, 1.0), borderaxespad=0.0,
        fontsize=8, frameon=False,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    title = 'Ground-state initial condition per cell'
    if tag:
        title = f'{title}\n{tag}'
    ax.set_title(title)
    return ax


# ---------------------------------------------------------------------
def render(npz_path, out_dir=None, units='reduced'):
    """Render the two-panel convergence figure and write the PNG.

    Parameters
    ----------
    npz_path : str
        Path to a sweep NPZ.
    out_dir : str or None, default=None
        Output directory; defaults to the NPZ's directory.
    units : {'reduced', 'absolute'}, default='reduced'
        Axis units, forwarded to `_axes_units`.

    Returns
    -------
    out_path : str
        Path to the written PNG.
    """
    data = load(npz_path)
    base = os.path.splitext(os.path.basename(npz_path))[0]
    if out_dir is None:
        out_dir = os.path.dirname(npz_path) or '.'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(15.0, 5.5))
    plot_convergence_fraction(data, axes[0], units=units)
    plot_ground_state_ic(data, axes[1], units=units)
    fig.tight_layout()
    out_path = os.path.join(
        out_dir, f'{base}_convergence_{units}.png',
    )
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------
def main():
    """Read the User Configuration block and render the PNG."""
    # ================ User Configuration ================
    in_path = 'output/phase_diagram/D_H_z.npz'
    out_dir = None
    units = 'reduced'
    # ============ End User Configuration =================
    if in_path is None:
        raise RuntimeError(
            'Set in_path in main() to a sweep NPZ path.'
        )
    out_path = render(in_path, out_dir=out_dir, units=units)
    print(f'Wrote {out_path}')


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
