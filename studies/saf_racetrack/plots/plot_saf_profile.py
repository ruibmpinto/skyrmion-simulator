"""SAF skyrmion cross-section profile and spin arrows.

Generates three separate figures:
1. m_z profile for both SAF layers (top and bottom).
2. Side-view spin arrows for the top Co layer.
3. Side-view spin arrows for the bottom Co layer.

Functions
---------
plot_mz_profile
    m_z cross-section for both layers.
plot_spin_cross_section
    Side-view spin arrows for one layer.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt

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
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})


def _skyrmion_profile(x, R_sk, dw, polarity=1):
    """Compute skyrmion m_z and m_x along a cross-section.

    Parameters
    ----------
    x : numpy.ndarray
        Position along cross-section in nm.
    R_sk : float
        Skyrmion radius in nm.
    dw : float
        Domain wall width in nm.
    polarity : {1, -1}
        +1: core down (top layer).
        -1: core up (bottom layer).

    Returns
    -------
    mz : numpy.ndarray
        Out-of-plane magnetization component.
    mx : numpy.ndarray
        In-plane (radial) magnetization component.
    """
    r = np.abs(x)
    theta = 2.0 * np.arctan(
        np.exp(-(r - R_sk) / dw)
    )
    if polarity == -1:
        theta = np.pi - theta
    mz = np.cos(theta)
    mx = np.sin(theta) * np.sign(x)
    mx[x == 0] = 0.0
    if polarity == -1:
        mx = -mx
    return mz, mx


# ---------------------------------------------------------------------
def _draw_spin_row(ax, x_arr, y0, mz_arr, mx_arr,
                   length=4.0, lw=1.8):
    """Draw a row of spin arrows in side view.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_arr : numpy.ndarray
        Horizontal positions.
    y0 : float
        Vertical baseline.
    mz_arr : numpy.ndarray
        Vertical spin component.
    mx_arr : numpy.ndarray
        Horizontal spin component.
    length : float
        Arrow length scale.
    lw : float
        Line width.
    """
    for x, mz, mx in zip(x_arr, mz_arr, mx_arr):
        dx = length * mx
        dy = length * mz
        if mz > 0.3:
            color = 'C3'
        elif mz < -0.3:
            color = 'C0'
        else:
            color = 'C2'
        ax.annotate(
            '',
            xy=(x + dx / 2, y0 + dy / 2),
            xytext=(x - dx / 2, y0 - dy / 2),
            arrowprops=dict(
                arrowstyle=(
                    '->,head_width=0.25,'
                    'head_length=0.15'
                ),
                color=color, lw=lw,
                shrinkA=0, shrinkB=0,
            ),
        )


# ---------------------------------------------------------------------
def plot_mz_profile(R_sk=80.0, dw=27.0,
                    figsize=(6.0, 4.0), save=None):
    """m_z cross-section for both SAF layers.

    Parameters
    ----------
    R_sk : float, default=80.0
        Skyrmion radius in nm.
    dw : float, default=27.0
        Domain wall width in nm.
    figsize : tuple, default=(6.0, 4.0)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    x = np.linspace(-200, 200, 1000)
    mz_top, _ = _skyrmion_profile(
        x, R_sk, dw, polarity=1,
    )
    mz_bot, _ = _skyrmion_profile(
        x, R_sk, dw, polarity=-1,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(
        x, -mz_top, '--',
        color='magenta', lw=1.8,
        label=r'$-$Co (top)',
    )
    ax.plot(
        x, mz_bot, '-.',
        color='k', lw=1.8,
        label='Co (bottom)',
    )
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$m_z$')
    ax.set_xlim(-200, 200)
    ax.set_ylim(-1.15, 1.15)
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.legend(loc='upper left', framealpha=0.9)
    ax.axhline(0, color='gray', lw=0.5)
    fig.tight_layout()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, ax


# ---------------------------------------------------------------------
def plot_spin_cross_section(R_sk=80.0, dw=27.0,
                            polarity=1, n_arrows=25,
                            figsize=(8.0, 1.8),
                            save=None):
    """Side-view spin arrows for one layer.

    Parameters
    ----------
    R_sk : float, default=80.0
        Skyrmion radius in nm.
    dw : float, default=27.0
        Domain wall width in nm.
    polarity : {1, -1}, default=1
        +1: top layer (core down).
        -1: bottom layer (core up).
    n_arrows : int, default=35
        Number of spin arrows.
    figsize : tuple, default=(8.0, 1.8)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    x_arr = np.linspace(-180, 180, n_arrows)
    mz, mx = _skyrmion_profile(
        x_arr, R_sk, dw, polarity=polarity,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=figsize)
    _draw_spin_row(
        ax, x_arr, 0.0, mz, mx,
        length=40.0, lw=2.0,
    )
    # Box lines
    ax.plot(
        [-210, 210], [-25, -25],
        color='0.5', lw=0.8,
    )
    ax.plot(
        [-210, 210], [25, 25],
        color='0.5', lw=0.8,
    )
    ax.set_xlim(-220, 220)
    ax.set_ylim(-40.0, 40.0)
    ax.set_aspect('equal')
    ax.axis('off')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, ax


# =====================================================================
if __name__ == '__main__':
    out = 'output/figures'
    os.makedirs(out, exist_ok=True)
    plot_mz_profile(
        save=os.path.join(out, 'saf_mz_profile.png'),
    )
    plot_spin_cross_section(
        polarity=1,
        save=os.path.join(
            out, 'saf_spins_top.png',
        ),
    )
    plot_spin_cross_section(
        polarity=-1,
        save=os.path.join(
            out, 'saf_spins_bottom.png',
        ),
    )
    plt.show()
