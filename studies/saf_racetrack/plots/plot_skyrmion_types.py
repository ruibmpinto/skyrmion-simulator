"""Plot Neel, Bloch, and mixed skyrmion configurations.

Generates publication-ready figures showing the three
fundamental skyrmion types distinguished by their helicity
angle gamma:
    - Neel (gamma = 0):   in-plane spins point radially
    - Bloch (gamma = pi/2): in-plane spins point tangentially
    - Mixed (0 < gamma < pi/2): intermediate

Functions
---------
make_skyrmion
    Generate a skyrmion with arbitrary helicity.
plot_skyrmion_type
    Plot a single skyrmion configuration.
plot_all_types
    Plot Neel, Bloch, and mixed side by side.
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
})


def make_skyrmion(n=128, R_sk=0.3, dw=0.08,
                  helicity=0.0):
    """Generate a skyrmion with arbitrary helicity.

    Parameters
    ----------
    n : int, default=128
        Grid size (n x n).
    R_sk : float, default=0.3
        Skyrmion radius in normalized units [0, 1].
    dw : float, default=0.08
        Domain wall width in normalized units.
    helicity : float, default=0.0
        Helicity angle gamma in radians.
        0 = Neel, pi/2 = Bloch.

    Returns
    -------
    X : numpy.ndarray(2d)
        x-coordinates, shape (n, n).
    Y : numpy.ndarray(2d)
        y-coordinates, shape (n, n).
    mx : numpy.ndarray(2d)
        Spin x-component.
    my : numpy.ndarray(2d)
        Spin y-component.
    mz : numpy.ndarray(2d)
        Spin z-component.

    Notes
    -----
    The magnetization vector is:
        m_x = sin(theta) * cos(phi + gamma)
        m_y = sin(theta) * sin(phi + gamma)
        m_z = cos(theta)

    where theta(r) is the domain-wall profile,
    phi = atan2(y, x), and gamma is the helicity.
    """
    x = np.linspace(-0.5, 0.5, n)
    X, Y = np.meshgrid(x, x)
    r = np.sqrt(X ** 2 + Y ** 2)
    phi = np.arctan2(Y, X)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Domain-wall profile
    theta = 2.0 * np.arctan(np.exp(-(r - R_sk) / dw))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # In-plane angle = position angle + helicity
    psi = phi + helicity
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin components
    sin_t = np.sin(theta)
    cos_t = np.cos(theta)
    mx = sin_t * np.cos(psi)
    my = sin_t * np.sin(psi)
    mz = cos_t
    return X, Y, mx, my, mz


# ---------------------------------------------------------------------
def _add_helicity_diagram(ax, helicity, pos=(0.32, -0.38),
                          size=0.1):
    """Add a small helicity angle diagram to an axis.

    Shows the radial direction r_hat and the in-plane
    magnetization direction m_ip, with the angle gamma
    between them.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    helicity : float
        Helicity angle in radians.
    pos : tuple, default=(0.32, -0.38)
        Center position in data coordinates.
    size : float, default=0.1
        Arrow length in data coordinates.
    """
    cx, cy = pos
    # Radial direction (reference, pointing right)
    r_angle = 0.0
    # In-plane m direction
    m_angle = helicity
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw r_hat arrow (gray, dashed-style)
    rx = cx + size * np.cos(r_angle)
    ry = cy + size * np.sin(r_angle)
    ax.annotate(
        '', xy=(rx, ry), xytext=(cx, cy),
        arrowprops=dict(
            arrowstyle='->', color='0.4',
            lw=1.5, ls='--',
        ),
    )
    ax.text(
        rx + 0.015, ry + 0.01,
        r'$\hat{r}$', fontsize=11, color='0.4',
        ha='left', va='bottom',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw m_ip arrow (black, solid)
    mx = cx + size * np.cos(m_angle)
    my = cy + size * np.sin(m_angle)
    ax.annotate(
        '', xy=(mx, my), xytext=(cx, cy),
        arrowprops=dict(
            arrowstyle='->', color='k', lw=1.8,
        ),
    )
    ax.text(
        mx + 0.015, my + 0.01,
        r'$\mathbf{m}_{\perp}$', fontsize=11,
        color='k', ha='left', va='bottom',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Draw arc for gamma angle
    if abs(helicity) > 0.01:
        n_pts = 30
        arc_r = size * 0.55
        angles = np.linspace(r_angle, m_angle, n_pts)
        arc_x = cx + arc_r * np.cos(angles)
        arc_y = cy + arc_r * np.sin(angles)
        ax.plot(
            arc_x, arc_y, '-', color='C3',
            lw=1.2,
        )
        # Label at mid-angle
        mid = (r_angle + m_angle) / 2.0
        lx = cx + (arc_r + 0.025) * np.cos(mid)
        ly = cy + (arc_r + 0.025) * np.sin(mid)
        ax.text(
            lx, ly, r'$\gamma$', fontsize=12,
            color='C3', ha='center', va='center',
            fontweight='bold',
        )


# ---------------------------------------------------------------------
def plot_all_types(figsize=(15, 5.0), save=None):
    """Plot Neel, Bloch, and mixed skyrmions.

    Each panel shows:
    - m_z color map with in-plane arrows
    - Helicity angle diagram
    - Magnetization vector equations

    Parameters
    ----------
    figsize : tuple, default=(15, 5.0)
        Figure size in inches.
    save : str, default=None
        If given, save figure to this path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    configs = [
        {
            'helicity': 0.0,
            'title': (
                r'(a) N$\mathrm{\acute{e}}$el'
            ),
            'gamma_str': r'$\gamma = 0$',
            'desc': 'Radial',
        },
        {
            'helicity': np.pi / 2.0,
            'title': r'(b) Bloch',
            'gamma_str': r'$\gamma = \pi/2$',
            'desc': 'Tangential',
        },
        {
            'helicity': np.pi / 4.0,
            'title': r'(c) Mixed',
            'gamma_str': r'$\gamma = \pi/4$',
            'desc': 'Intermediate',
        },
    ]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    skip = 6
    scale = 25
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ax, cfg in zip(axes, configs):
        X, Y, mx, my, mz = make_skyrmion(
            n=128, R_sk=0.3, dw=0.08,
            helicity=cfg['helicity'],
        )
        im = ax.pcolormesh(
            X, Y, mz, cmap=cmap, norm=norm,
            shading='nearest', rasterized=True,
        )
        s = skip
        ax.quiver(
            X[::s, ::s], Y[::s, ::s],
            mx[::s, ::s], my[::s, ::s],
            scale=scale, width=0.004,
            color='k', alpha=0.8,
            headwidth=3, headlength=4,
        )
        # Title
        ax.set_title(cfg['title'], fontsize=14)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Gamma label
        ax.text(
            0.05, 0.95, cfg['gamma_str'],
            transform=ax.transAxes,
            fontsize=13, va='top', ha='left',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.85,
                edgecolor='0.7',
            ),
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Helicity diagram
        _add_helicity_diagram(
            ax, cfg['helicity'],
            pos=(0.32, -0.38), size=0.1,
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared colorbar
    fig.subplots_adjust(right=0.89, wspace=0.08)
    cax = fig.add_axes([0.91, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
    )
    cbar.set_label(r'$m_z$')
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Magnetization vector equation (below panels)
    fig.text(
        0.45, 0.01,
        r'$\mathbf{m} = '
        r'\sin\theta\,\cos(\varphi + \gamma)\,'
        r'\hat{x}'
        r' + \sin\theta\,\sin(\varphi + \gamma)\,'
        r'\hat{y}'
        r' + \cos\theta\,\hat{z}$'
        r'$\qquad$'
        r'$\theta(r) = 2\arctan'
        r'\!\left(e^{-(r - R_{sk})/\Delta}\right)$',
        fontsize=12, ha='center', va='bottom',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, axes


# ---------------------------------------------------------------------
def plot_individual_types(save_dir=None):
    """Plot each skyrmion type as a separate figure.

    Parameters
    ----------
    save_dir : str, default=None
        Directory to save figures.

    Returns
    -------
    figs : list[matplotlib.figure.Figure]
    """
    configs = [
        {
            'helicity': 0.0,
            'title': (
                r'N$\mathrm{\acute{e}}$el skyrmion'
            ),
            'gamma_str': r'$\gamma = 0$',
            'filename': 'skyrmion_neel.png',
        },
        {
            'helicity': np.pi / 2.0,
            'title': r'Bloch skyrmion',
            'gamma_str': r'$\gamma = \pi/2$',
            'filename': 'skyrmion_bloch.png',
        },
        {
            'helicity': np.pi / 4.0,
            'title': r'Mixed skyrmion',
            'gamma_str': r'$\gamma = \pi/4$',
            'filename': 'skyrmion_mixed.png',
        },
    ]
    cmap = plt.cm.RdBu_r
    norm = mcolors.Normalize(vmin=-1, vmax=1)
    figs = []
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for cfg in configs:
        X, Y, mx, my, mz = make_skyrmion(
            n=128, R_sk=0.3, dw=0.08,
            helicity=cfg['helicity'],
        )
        fig, ax = plt.subplots(figsize=(5.5, 5.0))
        im = ax.pcolormesh(
            X, Y, mz, cmap=cmap, norm=norm,
            shading='nearest', rasterized=True,
        )
        s = 6
        ax.quiver(
            X[::s, ::s], Y[::s, ::s],
            mx[::s, ::s], my[::s, ::s],
            scale=25, width=0.004,
            color='k', alpha=0.8,
            headwidth=3, headlength=4,
        )
        ax.set_title(cfg['title'], fontsize=14)
        ax.set_xlim(-0.5, 0.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Gamma label
        ax.text(
            0.05, 0.95, cfg['gamma_str'],
            transform=ax.transAxes,
            fontsize=13, va='top', ha='left',
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white', alpha=0.85,
                edgecolor='0.7',
            ),
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Helicity diagram
        _add_helicity_diagram(
            ax, cfg['helicity'],
            pos=(0.32, -0.38), size=0.1,
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Magnetization vector
        ax.text(
            0.5, -0.02,
            r'$\mathbf{m} = '
            r'(\sin\theta\cos(\varphi+\gamma),\;'
            r'\sin\theta\sin(\varphi+\gamma),\;'
            r'\cos\theta)$',
            transform=ax.transAxes,
            fontsize=10, ha='center', va='top',
        )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # Colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(
            'right', size='4%', pad=0.08,
        )
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label(r'$m_z$')
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1])
        fig.tight_layout()
        figs.append(fig)
        if save_dir:
            path = os.path.join(
                save_dir, cfg['filename'],
            )
            fig.savefig(path)
            print(f'Saved: {path}')
    return figs


# =====================================================================
if __name__ == '__main__':
    out = 'output/figures'
    os.makedirs(out, exist_ok=True)
    plot_all_types(
        save=os.path.join(out, 'skyrmion_types.png'),
    )
    plot_individual_types(save_dir=out)
    plt.show()
