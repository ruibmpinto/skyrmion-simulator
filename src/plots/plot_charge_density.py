"""Topological charge density maps for SAF skyrmions.

Computes and plots the topological charge density
rho_Q = (1/4pi) m . (dm/dx x dm/dy) for individual
layers and frames from a LAMMPS dump file.

Functions
---------
compute_charge_density
    Compute the 2D topological charge density.
plot_charge_density
    Plot the charge density map for one layer/frame.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
# Local
from src.plot_figures import read_dump_frame, _extract_layer

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def compute_charge_density(mx, my, mz, a):
    """Compute the 2D topological charge density.

    Parameters
    ----------
    mx : numpy.ndarray(2d)
        Spin x-component, shape (ny, nx).
    my : numpy.ndarray(2d)
        Spin y-component, shape (ny, nx).
    mz : numpy.ndarray(2d)
        Spin z-component, shape (ny, nx).
    a : float
        Lattice constant in nm.

    Returns
    -------
    rho : numpy.ndarray(2d)
        Topological charge density, shape (ny, nx).
    Q : float
        Integrated topological charge.

    Notes
    -----
    rho_Q = (1/4pi) m . (dm/dx x dm/dy)

    Central finite differences with PBC (np.roll).
    """
    m = np.stack([mx, my, mz], axis=-1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Central differences
    dmdx = (
        np.roll(m, -1, axis=1)
        - np.roll(m, +1, axis=1)
    ) / (2.0 * a)
    dmdy = (
        np.roll(m, -1, axis=0)
        - np.roll(m, +1, axis=0)
    ) / (2.0 * a)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cross product dm/dx x dm/dy
    cross = np.cross(dmdx, dmdy)
    # Dot with m
    rho = np.sum(m * cross, axis=-1) / (4.0 * np.pi)
    # Integrated charge
    Q = np.sum(rho) * a * a
    return rho, Q


# ---------------------------------------------------------------------
def plot_charge_density(filepath, frame=0, layer=1,
                        figsize=(5.5, 5.0), save=None):
    """Plot topological charge density for one layer/frame.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    frame : int, default=0
        Frame index.
    layer : int, default=1
        Layer type (1=top, 2=bottom).
    figsize : tuple, default=(5.5, 5.0)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    step, data = read_dump_frame(filepath, frame)
    x, y, mx, my, mz = _extract_layer(data, layer)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reshape to 2D grid
    nx = len(np.unique(x))
    ny = len(np.unique(y))
    X = x.reshape(ny, nx)
    Y = y.reshape(ny, nx)
    MX = mx.reshape(ny, nx)
    MY = my.reshape(ny, nx)
    MZ = mz.reshape(ny, nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Lattice constant from positions
    x_unique = np.sort(np.unique(x))
    a = x_unique[1] - x_unique[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Compute charge density
    rho, Q = compute_charge_density(MX, MY, MZ, a)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    # Symmetric color scale
    vmax = np.max(np.abs(rho))
    if vmax < 1e-10:
        vmax = 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    im = ax.pcolormesh(
        X, Y, rho, cmap='RdBu_r', norm=norm,
        shading='nearest', rasterized=True,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(
        'right', size='4%', pad=0.08,
    )
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(
        r'$\rho_Q$ (nm$^{-2}$)'
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Integrated Q as text
    layer_name = 'Top' if layer == 1 else 'Bottom'
    ax.text(
        0.03, 0.97,
        f'$Q = {Q:+.3f}$',
        transform=ax.transAxes,
        fontsize=12, va='top', ha='left',
        bbox=dict(
            boxstyle='round,pad=0.3',
            facecolor='white', alpha=0.85,
            edgecolor='0.7',
        ),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlabel('$x$ (nm)')
    ax.set_ylabel('$y$ (nm)')
    ax.set_title(
        f'{layer_name} layer, step {step}'
    )
    ax.set_aspect('equal')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, ax


# =====================================================================
if __name__ == '__main__':
    dump = 'output/skyrmion.dump'
    out = 'output/figures'
    os.makedirs(out, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Count frames
    with open(dump, 'r') as f:
        n_frames = sum(
            1 for line in f
            if line.startswith('ITEM: TIMESTEP')
        )
    last = n_frames - 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (a) Top layer, static (frame 0)
    plot_charge_density(
        dump, frame=0, layer=1,
        save=os.path.join(
            out, 'charge_density_top_static.png',
        ),
    )
    # (b) Bottom layer, static (frame 0)
    plot_charge_density(
        dump, frame=0, layer=2,
        save=os.path.join(
            out, 'charge_density_bot_static.png',
        ),
    )
    # (c) Top layer, moving (last frame)
    plot_charge_density(
        dump, frame=last, layer=1,
        save=os.path.join(
            out, 'charge_density_top_moving.png',
        ),
    )
    # (d) Bottom layer, moving (last frame)
    plot_charge_density(
        dump, frame=last, layer=2,
        save=os.path.join(
            out, 'charge_density_bot_moving.png',
        ),
    )
    plt.show()
