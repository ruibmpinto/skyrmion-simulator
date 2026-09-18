"""Figures explaining the spin reorientation transition.

Generates publication-ready figures illustrating why the
Co layers are tuned near the SRT for skyrmion stabilization.

Functions
---------
plot_keff_vs_thickness
    K_eff as a function of Co thickness.
plot_energy_landscape
    Domain wall energy and skyrmion stability near SRT.
plot_hysteresis_comparison
    Square loop vs linear loop.
plot_spin_regimes
    Three magnetization regimes with spin schematics.
plot_srt_overview
    Combined 4-panel overview figure.
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

# Physical constants
MU0 = 4.0 * np.pi * 1e-7
MS = 1.43e6       # A/m
A_EX = 16e-12     # J/m
D = 0.62e-3        # J/m^2
DEMAG = MU0 * MS ** 2 / 2  # ~1.286e6 J/m^3


def plot_keff_vs_thickness(figsize=(5.5, 3.5),
                           save=None):
    """K_eff as a function of effective K (proxy for Co thickness).

    Shows how K_eff = K - mu0*Ms^2/2 crosses zero at the
    spin reorientation transition. The operating point of
    the paper is marked.

    Parameters
    ----------
    figsize : tuple, default=(5.5, 3.5)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    """
    # K varies with Co thickness (interface anisotropy
    # contribution scales as K_s / t_Co)
    t = np.linspace(0.8, 2.5, 200)  # nm
    # Model: K = K_s / t + K_v, fit to match paper
    K_s = 1.7e-3   # J/m^2, interface anisotropy
    K_v = -0.1e6   # J/m^3, volume contribution
    K = K_s / (t * 1e-9) + K_v
    K_eff = K - DEMAG
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(t, K_eff / 1e3, 'k-', lw=2)
    ax.axhline(0, color='gray', ls='-', lw=0.5)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # SRT point
    t_srt = K_s / ((DEMAG - K_v) * 1e-9)
    ax.axvline(
        t_srt, color='C3', ls=':', lw=1.2,
        label=f'SRT ($t_{{\\mathrm{{Co}}}}$'
        f' = {t_srt:.2f} nm)',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Operating point
    K_eff_op = (1.294e6 + 1.31e6) / 2 - DEMAG
    t_op = 1.58
    ax.plot(
        t_op, K_eff_op / 1e3, 'o', ms=8,
        color='C0', zorder=5,
        label=f'Paper ($t_{{\\mathrm{{Co}}}}$'
              f' = {t_op} nm)',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shade regions
    ax.fill_between(
        t, K_eff / 1e3, 0,
        where=(K_eff > 0),
        alpha=0.1, color='C0',
    )
    ax.fill_between(
        t, K_eff / 1e3, 0,
        where=(K_eff < 0),
        alpha=0.1, color='C3',
    )
    ax.text(
        1.0, -80, 'In-plane\n(easy plane)',
        fontsize=10, ha='center', color='C3',
    )
    ax.text(
        2.2, 80, 'Out-of-plane\n(easy axis)',
        fontsize=10, ha='center', color='C0',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ax.set_xlabel(
        r'Co thickness $t_{\mathrm{Co}}$ (nm)'
    )
    ax.set_ylabel(
        r'$K_{\mathrm{eff}}$ (kJ/m$^3$)'
    )
    ax.set_title(
        'Spin reorientation transition'
    )
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_xlim(0.8, 2.5)
    fig.tight_layout()
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, ax


# ---------------------------------------------------------------------
def plot_energy_landscape(figsize=(5.5, 3.5),
                          save=None):
    """Domain wall energy and critical DMI vs K_eff.

    Shows that near the SRT, the domain wall energy
    vanishes and the DMI can stabilize skyrmions.

    Parameters
    ----------
    figsize : tuple, default=(5.5, 3.5)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : tuple
        (ax1, ax2) for left and right y-axes.
    """
    K_eff = np.linspace(1e3, 2e5, 200)
    # Domain wall energy: sigma = 4*sqrt(A*K_eff) - pi*D
    sigma = (
        4.0 * np.sqrt(A_EX * K_eff) - np.pi * D
    )
    # Domain wall width
    delta = np.sqrt(A_EX / K_eff) * 1e9  # nm
    # Critical DMI
    D_c = 4.0 * np.sqrt(A_EX * K_eff) / np.pi
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax1 = plt.subplots(figsize=figsize)
    color1 = 'C0'
    color2 = 'C3'
    # DW energy
    ln1 = ax1.plot(
        K_eff / 1e3, sigma * 1e3, '-',
        color=color1, lw=2,
        label=r'$\sigma_{\mathrm{DW}}'
              r' = 4\sqrt{AK_{\mathrm{eff}}}'
              r' - \pi D$',
    )
    ax1.axhline(0, color='gray', ls='-', lw=0.5)
    ax1.set_xlabel(
        r'$K_{\mathrm{eff}}$ (kJ/m$^3$)'
    )
    ax1.set_ylabel(
        r'DW energy $\sigma$ (mJ/m$^2$)',
        color=color1,
    )
    ax1.tick_params(axis='y', labelcolor=color1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # DW width on right axis
    ax2 = ax1.twinx()
    ln2 = ax2.plot(
        K_eff / 1e3, delta, '--',
        color=color2, lw=2,
        label=r'$\Delta = \sqrt{A/K_{\mathrm{eff}}}$',
    )
    ax2.set_ylabel(
        r'DW width $\Delta$ (nm)',
        color=color2,
    )
    ax2.tick_params(axis='y', labelcolor=color2)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Operating point
    K_op = 17000  # average K_eff
    sigma_op = (
        4.0 * np.sqrt(A_EX * K_op) - np.pi * D
    )
    delta_op = np.sqrt(A_EX / K_op) * 1e9
    ax1.plot(
        K_op / 1e3, sigma_op * 1e3, 'o',
        ms=8, color=color1, zorder=5,
    )
    ax2.plot(
        K_op / 1e3, delta_op, 's',
        ms=8, color=color2, zorder=5,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Skyrmion stability region (sigma < 0)
    sigma_zero = K_eff[np.argmin(np.abs(sigma))]
    ax1.axvspan(
        0, sigma_zero / 1e3,
        alpha=0.08, color='green',
    )
    ax1.text(
        sigma_zero / 1e3 / 2, -0.3,
        'Skyrmions\nstable\n'
        r'($\sigma < 0$)',
        fontsize=9, ha='center', va='top',
        color='green',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Combined legend
    lns = ln1 + ln2
    labs = [ln.get_label() for ln in lns]
    ax1.legend(
        lns, labs, loc='center right',
        framealpha=0.9,
    )
    ax1.set_title(
        'Domain wall energy and width'
    )
    ax1.set_xlim(0, 200)
    fig.tight_layout()
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, (ax1, ax2)


# ---------------------------------------------------------------------
def plot_hysteresis_comparison(figsize=(10, 3.5),
                               save=None):
    """Square loop (strong PMA) vs linear loop (near SRT).

    Parameters
    ----------
    figsize : tuple, default=(10, 3.5)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (a) Strong PMA: square hysteresis
    ax = axes[0]
    H = np.linspace(-1, 1, 500)
    # Simulate square loop with tanh
    M_up = np.tanh(20 * (H - 0.3))
    M_down = np.tanh(20 * (H + 0.3))
    ax.plot(H, M_up, 'C3-', lw=2)
    ax.plot(H, M_down, 'C0-', lw=2)
    ax.set_xlabel(r'$B_z$ (a.u.)')
    ax.set_ylabel(r'$M / M_s$')
    ax.set_title(
        r'(a) Strong PMA ($K_{\mathrm{eff}} \gg 0$)'
    )
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    # Annotation
    ax.annotate(
        'Single domain\nswitching',
        xy=(0.3, 0.5), xytext=(0.6, -0.3),
        fontsize=9, ha='center',
        arrowprops=dict(
            arrowstyle='->', color='0.3',
        ),
    )
    # Spin schematics
    for yp, label in [(0.85, r'$\uparrow\uparrow'
                       r'\uparrow\uparrow$'),
                      (-0.85, r'$\downarrow\downarrow'
                       r'\downarrow\downarrow$')]:
        ax.text(
            -0.8, yp, label,
            fontsize=14, ha='center', va='center',
            color='0.3',
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (b) Near SRT: linear, reversible
    ax = axes[1]
    M_linear = np.tanh(1.5 * H)
    ax.plot(H, M_linear, 'C0-', lw=2)
    ax.set_xlabel(r'$B_z$ (a.u.)')
    ax.set_ylabel(r'$M / M_s$')
    ax.set_title(
        r'(b) Near SRT ($K_{\mathrm{eff}} \approx 0$)'
    )
    ax.set_ylim(-1.3, 1.3)
    ax.axhline(0, color='gray', lw=0.5)
    ax.axvline(0, color='gray', lw=0.5)
    # Annotation
    ax.annotate(
        'Multidomain\n(gradual rotation)',
        xy=(0.2, 0.3), xytext=(0.6, -0.4),
        fontsize=9, ha='center',
        arrowprops=dict(
            arrowstyle='->', color='0.3',
        ),
    )
    # Spin schematic: mixed domains
    ax.text(
        -0.8, 0.0,
        r'$\uparrow\downarrow'
        r'\uparrow\downarrow$',
        fontsize=14, ha='center', va='center',
        color='0.3',
    )
    ax.text(
        -0.8, -0.25, 'domains',
        fontsize=8, ha='center', color='0.5',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.tight_layout()
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, axes


# ---------------------------------------------------------------------
def _draw_spins(ax, x_arr, y_arr, angles, colors,
                length=0.3):
    """Draw spin arrows at given positions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x_arr : array-like
        x positions.
    y_arr : array-like
        y positions.
    angles : array-like
        Spin angles from +z (0 = up, pi = down,
        pi/2 = in-plane).
    colors : array-like
        Colors for each arrow.
    length : float
        Arrow length.
    """
    for x, y, ang, c in zip(
        x_arr, y_arr, angles, colors
    ):
        dx = length * np.sin(ang)
        dy = length * np.cos(ang)
        ax.annotate(
            '', xy=(x + dx, y + dy),
            xytext=(x - dx, y - dy),
            arrowprops=dict(
                arrowstyle='->', color=c,
                lw=1.8, shrinkA=0, shrinkB=0,
            ),
        )


# ---------------------------------------------------------------------
def plot_spin_regimes(figsize=(13, 4.0), save=None):
    """Three magnetization regimes with spin schematics.

    Shows (a) strong PMA, (b) near SRT with skyrmion,
    (c) in-plane.

    Parameters
    ----------
    figsize : tuple, default=(13, 4.0)
        Figure size.
    save : str, default=None
        Save path.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : list[matplotlib.axes.Axes]
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    nx, ny = 7, 5
    xs = np.arange(nx, dtype=float)
    ys = np.arange(ny, dtype=float)
    XX, YY = np.meshgrid(xs, ys)
    xx = XX.ravel()
    yy = YY.ravel()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (a) Strong PMA: all spins up
    ax = axes[0]
    angles = np.zeros(len(xx))
    colors = ['C3'] * len(xx)
    _draw_spins(ax, xx, yy, angles, colors)
    ax.set_title(
        r'(a) Strong PMA'
        '\n'
        r'$K_{\mathrm{eff}} \gg 0$',
    )
    ax.text(
        3, -1.0, 'Uniform state',
        ha='center', fontsize=10, style='italic',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (b) Near SRT: skyrmion (center down, edge up)
    ax = axes[1]
    cx, cy = 3.0, 2.0
    r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    R_sk = 1.8
    dw = 0.6
    theta = 2.0 * np.arctan(
        np.exp(-(r - R_sk) / dw)
    )
    colors_b = []
    for t in theta:
        if t > 2.5:
            colors_b.append('C0')
        elif t < 0.6:
            colors_b.append('C3')
        else:
            colors_b.append('C2')
    _draw_spins(ax, xx, yy, theta, colors_b)
    ax.set_title(
        r'(b) Near SRT'
        '\n'
        r'$K_{\mathrm{eff}} \approx 0$',
    )
    ax.text(
        3, -1.0,
        'Skyrmions / multidomain',
        ha='center', fontsize=10, style='italic',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # (c) In-plane: all spins along x
    ax = axes[2]
    angles = np.full(len(xx), np.pi / 2)
    colors = ['C0'] * len(xx)
    _draw_spins(ax, xx, yy, angles, colors)
    ax.set_title(
        r'(c) Easy plane'
        '\n'
        r'$K_{\mathrm{eff}} < 0$',
    )
    ax.text(
        3, -1.0, 'In-plane state',
        ha='center', fontsize=10, style='italic',
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for ax in axes:
        ax.set_xlim(-1, nx)
        ax.set_ylim(-1.8, ny)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.tight_layout()
    if save:
        fig.savefig(save)
        print(f'Saved: {save}')
    return fig, axes


# ---------------------------------------------------------------------
def plot_srt_overview(save_dir=None):
    """Generate all SRT explanation figures.

    Parameters
    ----------
    save_dir : str, default=None
        Directory to save all figures.
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    plot_keff_vs_thickness(
        save=(os.path.join(save_dir, 'srt_keff.png')
              if save_dir else None),
    )
    plot_energy_landscape(
        save=(os.path.join(save_dir, 'srt_energy.png')
              if save_dir else None),
    )
    plot_hysteresis_comparison(
        save=(os.path.join(
            save_dir, 'srt_hysteresis.png')
              if save_dir else None),
    )
    plot_spin_regimes(
        save=(os.path.join(
            save_dir, 'srt_spin_regimes.png')
              if save_dir else None),
    )


# =====================================================================
if __name__ == '__main__':
    out = 'output/figures'
    plot_srt_overview(save_dir=out)
    plt.show()
