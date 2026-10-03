"""Snapshot heatmap of the out-of-plane magnetization m_z.

Lightweight matplotlib helper that takes a single layer spin
array (ny, nx, 3) and renders the m_z component as a heatmap
with a symmetric color scale in [-1, 1]. Used for snapshot
panels and any other quick visual check of the simulation
state.

Functions
---------
plot_snapshot_mz
    Render m_z(x, y) for a single layer to a matplotlib axes.
plot_snapshot_mz_pair
    Side-by-side heatmaps for the top and bottom SAF layers.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults.
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


def plot_snapshot_mz(m, a, ax=None, title=None,
                     show_colorbar=True):
    """Render the out-of-plane magnetization m_z(x, y).

    The plotting area is forced to a square aspect (equal x and
    y physical scales). The figure aspect is left to matplotlib,
    so the colorbar can extend on the right without compressing
    the data area.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant (m); used to convert site indices to
        physical (nm) coordinates on the axes.
    ax : matplotlib.axes.Axes or None, default=None
        Existing axes to draw into. If None, a new square
        figure and axes are created at the project default
        size (6 x 6 inches).
    title : str or None, default=None
        Optional axis title.
    show_colorbar : bool, default=True
        Whether to attach a colorbar on the right of the axes.

    Returns
    -------
    fig : matplotlib.figure.Figure
        Parent figure (newly created if ax was None).
    ax : matplotlib.axes.Axes
        Axes object used for the heatmap.
    """
    # Lattice shape: ny rows (y), nx columns (x).
    ny, nx = m.shape[:2]
    # Physical coordinate arrays in nm (a is in metres).
    x_nm = np.arange(nx) * a * 1e9
    y_nm = np.arange(ny) * a * 1e9
    # Build the (X, Y) grid for pcolormesh.
    X, Y = np.meshgrid(x_nm, y_nm, indexing='xy')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Create a fresh figure if the caller did not supply axes.
    if ax is None:
        # Project default figure size (6 x 6 inches, square).
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    # Symmetric colour scale in [-1, 1]; m_z is a unit-vector component.
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    # Pcolormesh with the divergent RdBu_r colormap (red=+1, blue=-1).
    im = ax.pcolormesh(
        X, Y, m[..., 2], cmap='RdBu_r', norm=norm,
        shading='nearest', rasterized=True,)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Optional colorbar in a divided slot so it does not steal width
    # from the (square) data area.
    if show_colorbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='4%', pad=0.08)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r'$m_z$')
    # Force a square data area: equal x and y physical scales,
    # then make the box itself square so the heatmap renders as a
    # true square regardless of the data aspect ratio.
    ax.set_aspect('equal', adjustable='box')
    # Label the axes in nm.
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$y$ (nm)')
    # Optional title.
    if title is not None:
        ax.set_title(title)
    return fig, ax


# -----------------------------------------------------------------------------
def plot_snapshot_mz_pair(m_top, m_bot, a, suptitle=None,
                          save=None):
    """Side-by-side m_z heatmaps for top and bottom SAF layers.

    Each panel is rendered with a square data area; the figure
    is sized so the two panels share the same scale without
    distortion.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spin configuration, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in metres.
    suptitle : str or None, default=None
        Optional super-title across both axes.
    save : str or None, default=None
        If given, save the figure to this path and print a
        confirmation. Otherwise the figure is returned without
        being saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple of matplotlib.axes.Axes
        (axes_top, axes_bot).
    """
    # Two square panels side-by-side; total width = 2 x project default.
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 6.0))
    # Top layer in the left column.
    plot_snapshot_mz(m_top, a, ax=axes[0], title='Top layer')
    # Bottom layer in the right column.
    plot_snapshot_mz(m_bot, a, ax=axes[1], title='Bottom layer')
    # Optional super-title.
    if suptitle is not None:
        fig.suptitle(suptitle)
    # Tighten the layout to remove dead space.
    fig.tight_layout()
    # Optionally write to disk.
    if save is not None:
        # Ensure the destination directory exists; create on demand.
        os.makedirs(os.path.dirname(save) or '.', exist_ok=True)
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, axes
