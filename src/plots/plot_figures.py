"""Publication-ready figures for SAF skyrmion simulations.

Reads LAMMPS dump files and generates matplotlib figures
of the skyrmion spin texture, radial profile, and
topological charge evolution.

Functions
---------
read_dump_frame
    Parse a single frame from a LAMMPS dump file.
read_all_frames
    Parse all frames from a LAMMPS dump file.
plot_spin_texture
    Top-view color map of m_z with in-plane arrows.
plot_radial_profile
    Radial m_z profile vs analytical prediction.
plot_topological_charge
    Time evolution of Q for both layers.
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

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

# Publication style
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
    'text.usetex': False,
})


def read_dump_frame(filepath, frame=0):
    """Parse a single frame from a LAMMPS dump file.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.
    frame : int, default=0
        Frame index to read.

    Returns
    -------
    step : int
        Timestep number.
    data : dict
        Keys: 'id', 'type', 'x', 'y', 'z',
        'fx', 'fy', 'fz' (all numpy arrays).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find frame boundaries
    frame_starts = []
    for i, line in enumerate(lines):
        if line.startswith('ITEM: TIMESTEP'):
            frame_starts.append(i)
    if frame >= len(frame_starts):
        raise ValueError(
            f'Frame {frame} not found. '
            f'File has {len(frame_starts)} frames.'
        )
    start = frame_starts[frame]
    step = int(lines[start + 1].strip())
    n_atoms = int(lines[start + 3].strip())
    data_start = start + 9
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parse atom data
    block = lines[data_start:data_start + n_atoms]
    arr = np.loadtxt(block)
    data = {
        'id': arr[:, 0].astype(int),
        'type': arr[:, 1].astype(int),
        'x': arr[:, 2],
        'y': arr[:, 3],
        'z': arr[:, 4],
        'fx': arr[:, 5],
        'fy': arr[:, 6],
        'fz': arr[:, 7],
    }
    return step, data


# ---------------------------------------------------------------------
def read_all_frames(filepath):
    """Parse all frames from a LAMMPS dump file.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.

    Returns
    -------
    frames : list[tuple]
        List of (step, data) tuples.
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()
    frame_starts = []
    for i, line in enumerate(lines):
        if line.startswith('ITEM: TIMESTEP'):
            frame_starts.append(i)
    frames = []
    for idx in range(len(frame_starts)):
        start = frame_starts[idx]
        step = int(lines[start + 1].strip())
        n_atoms = int(lines[start + 3].strip())
        data_start = start + 9
        block = lines[data_start:data_start + n_atoms]
        arr = np.loadtxt(block)
        data = {
            'id': arr[:, 0].astype(int),
            'type': arr[:, 1].astype(int),
            'x': arr[:, 2],
            'y': arr[:, 3],
            'z': arr[:, 4],
            'fx': arr[:, 5],
            'fy': arr[:, 6],
            'fz': arr[:, 7],
        }
        frames.append((step, data))
    return frames


# ---------------------------------------------------------------------
def _extract_layer(data, layer_type=1):
    """Extract one layer from frame data.

    Parameters
    ----------
    data : dict
        Frame data from read_dump_frame.
    layer_type : int, default=1
        Particle type (1=top, 2=bottom).

    Returns
    -------
    x : numpy.ndarray(1d)
        x-coordinates in nm.
    y : numpy.ndarray(1d)
        y-coordinates in nm.
    mx : numpy.ndarray(1d)
        Spin x-component.
    my : numpy.ndarray(1d)
        Spin y-component.
    mz : numpy.ndarray(1d)
        Spin z-component.
    """
    mask = data['type'] == layer_type
    return (
        data['x'][mask],
        data['y'][mask],
        data['fx'][mask],
        data['fy'][mask],
        data['fz'][mask],
    )


# ---------------------------------------------------------------------
def plot_spin_texture(filepath, frame=0, layer=1,
                      arrow_skip=8, arrow_scale=15,
                      figsize=(5.5, 5.0), save=None):
    """Top-view color map of m_z with in-plane arrows.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    frame : int, default=0
        Frame index.
    layer : int, default=1
        Layer type (1=top, 2=bottom).
    arrow_skip : int, default=8
        Plot every Nth arrow.
    arrow_scale : float, default=15
        Arrow scale factor.
    figsize : tuple, default=(5.5, 5.0)
        Figure size in inches.
    save : str, default=None
        If given, save figure to this path.

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
    MZ = mz.reshape(ny, nx)
    MX = mx.reshape(ny, nx)
    MY = my.reshape(ny, nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    # Color map for m_z
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    im = ax.pcolormesh(
        X, Y, MZ, cmap=cmap, norm=norm,
        shading='nearest', rasterized=True,
    )
    # Arrows for in-plane components
    s = arrow_skip
    ax.quiver(
        X[::s, ::s], Y[::s, ::s],
        MX[::s, ::s], MY[::s, ::s],
        scale=arrow_scale, width=0.003,
        color='k', alpha=0.7,
        headwidth=3, headlength=4,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Colorbar
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4%', pad=0.08)
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(r'$m_z$')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Labels
    layer_name = 'Top' if layer == 1 else 'Bottom'
    dt_ps = step * 5e-14 * 1e12  # assumes dt=50 fs
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


# ---------------------------------------------------------------------
def plot_radial_profile(filepath, frame=0, layer=1,
                        R_sk=80.0, dw=27.0,
                        figsize=(5.5, 3.5), save=None):
    """Radial m_z profile vs analytical prediction.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    frame : int, default=0
        Frame index.
    layer : int, default=1
        Layer type.
    R_sk : float, default=80.0
        Skyrmion radius in nm.
    dw : float, default=27.0
        Domain wall width in nm.
    figsize : tuple, default=(5.5, 3.5)
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
    # Compute radial distance from center
    cx = (x.max() + x.min()) / 2.0
    cy = (y.max() + y.min()) / 2.0
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Bin the radial profile
    r_max = 200.0
    bins = np.arange(0, r_max, 2.0)
    r_mid = 0.5 * (bins[:-1] + bins[1:])
    mz_avg = np.zeros(len(r_mid))
    mz_std = np.zeros(len(r_mid))
    for i in range(len(r_mid)):
        mask = (r >= bins[i]) & (r < bins[i + 1])
        if mask.sum() > 0:
            mz_avg[i] = mz[mask].mean()
            mz_std[i] = mz[mask].std()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytical profile
    r_an = np.linspace(0, r_max, 500)
    theta_an = 2.0 * np.arctan(
        np.exp(-(r_an - R_sk) / dw)
    )
    mz_an = np.cos(theta_an)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(
        r_mid, mz_avg - mz_std, mz_avg + mz_std,
        alpha=0.2, color='C0', label=None,
    )
    ax.plot(
        r_mid, mz_avg, 'o', ms=3, color='C0',
        label='Simulation',
    )
    ax.plot(
        r_an, mz_an, '-', color='C3', lw=1.5,
        label=(
            f'Analytical '
            r'($R_{{\mathrm{{sk}}}}$'
            f'={R_sk:.0f} nm, '
            r'$\Delta$'
            f'={dw:.0f} nm)'
        ),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.axhline(0, color='gray', ls=':', lw=0.5)
    ax.axvline(R_sk, color='gray', ls='--', lw=0.5,
               label=f'$R_{{\\mathrm{{sk}}}}$ = {R_sk:.0f} nm')
    ax.set_xlabel('$r$ (nm)')
    ax.set_ylabel('$m_z$')
    ax.set_title(f'Radial profile, step {step}')
    ax.set_xlim(0, r_max)
    ax.set_ylim(-1.1, 1.1)
    ax.legend(loc='lower right', framealpha=0.9)
    fig.tight_layout()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, ax


# ---------------------------------------------------------------------
def plot_topological_charge(filepath, dt=5e-14,
                            figsize=(5.5, 3.0),
                            save=None):
    """Time evolution of Q for both layers.

    Parameters
    ----------
    filepath : str
        Path to LAMMPS dump file.
    dt : float, default=5e-14
        Time step in seconds.
    figsize : tuple, default=(5.5, 3.0)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    frames = read_all_frames(filepath)
    steps = []
    Q_top_list = []
    Q_bot_list = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for step, data in frames:
        steps.append(step)
        for layer, Q_list in [(1, Q_top_list),
                              (2, Q_bot_list)]:
            x, y, mx, my, mz = _extract_layer(
                data, layer
            )
            nx = len(np.unique(x))
            ny = len(np.unique(y))
            a = np.unique(np.diff(np.sort(
                np.unique(x)
            )))[0]
            MX = mx.reshape(ny, nx)
            MY = my.reshape(ny, nx)
            MZ = mz.reshape(ny, nx)
            m = np.stack([MX, MY, MZ], axis=-1)
            dmdx = (
                np.roll(m, -1, axis=1)
                - np.roll(m, +1, axis=1)
            ) / (2.0 * a)
            dmdy = (
                np.roll(m, -1, axis=0)
                - np.roll(m, +1, axis=0)
            ) / (2.0 * a)
            cross = np.cross(dmdx, dmdy)
            density = np.sum(m * cross, axis=-1)
            Q = np.sum(density) * a * a / (4 * np.pi)
            Q_list.append(Q)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    t_ps = np.array(steps) * dt * 1e12
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t_ps, Q_top_list, '-', color='C0',
            lw=1.5, label='Top layer')
    ax.plot(t_ps, Q_bot_list, '--', color='C3',
            lw=1.5, label='Bottom layer')
    ax.axhline(-1, color='gray', ls=':', lw=0.5)
    ax.axhline(+1, color='gray', ls=':', lw=0.5)
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Topological charge $Q$')
    ax.set_title('Topological charge evolution')
    ax.legend(loc='center right', framealpha=0.9)
    fig.tight_layout()
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
    # Fig 1: Spin texture (first frame after relaxation)
    plot_spin_texture(
        dump, frame=0, layer=1,
        save=os.path.join(out, 'fig1_spin_texture.png'),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fig 2: Radial profile
    plot_radial_profile(
        dump, frame=0, layer=1,
        R_sk=80.0, dw=27.0,
        save=os.path.join(out, 'fig2_radial_profile.png'),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fig 3: Topological charge
    plot_topological_charge(
        dump,
        save=os.path.join(out, 'fig3_topological_charge.png'),
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plt.show()
