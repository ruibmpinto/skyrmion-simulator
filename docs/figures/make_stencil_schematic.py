"""Render the 2D finite-difference stencil used by the simulator.

Produces a single figure with two panels:

(a) Exchange Laplacian: 5-point stencil with weights
    {-4 (center), +1 (4 nearest neighbours)}.
(b) Interfacial DMI: central-difference stencil with weights
    {+1, -1} on opposite neighbours, 0 at the centre. Arrows
    indicate the sign convention. The lattice constant `a`
    sets the grid spacing.

Output
------
docs/figures/stencil_schematic.pdf (and .png)
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10


def _draw_lattice(ax, n=5):
    """Faint background lattice of n x n points centred on origin."""
    # n must be odd so the centre point sits at (0, 0).
    k = (n - 1) // 2
    xs, ys = np.meshgrid(np.arange(-k, k + 1),
                         np.arange(-k, k + 1))
    ax.plot(xs.ravel(), ys.ravel(), 'o',
            color='0.82', markersize=8, zorder=1)
    # Grid lines for visual context.
    for v in range(-k, k + 1):
        ax.axhline(v, color='0.92', lw=0.6, zorder=0)
        ax.axvline(v, color='0.92', lw=0.6, zorder=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-k - 0.6, k + 0.6)
    ax.set_ylim(-k - 0.6, k + 0.6)
    ax.set_aspect('equal')
    for sp in ax.spines.values():
        sp.set_visible(False)


def _annotate_node(ax, x, y, label, color, fontcolor='white',
                   ms=22):
    """Draw a coloured circle at (x, y) with a centered label."""
    ax.plot(x, y, 'o', color=color, markersize=ms,
            markeredgecolor='black', zorder=3)
    ax.text(x, y, label, ha='center', va='center',
            color=fontcolor, fontsize=11, fontweight='bold',
            zorder=4)


def _panel_exchange(ax):
    """5-point Laplacian: centre -4, NN +1, derived from
    sum_{nn} m_nn - 4 m_0 over the 4 NN."""
    _draw_lattice(ax, n=5)
    # Centre point with weight -4.
    _annotate_node(ax, 0, 0, r'$-4$', color='#C0392B')
    # Four nearest neighbours with weight +1.
    for dx, dy, tag in [(+1, 0, r'$+1$'),
                        (-1, 0, r'$+1$'),
                        (0, +1, r'$+1$'),
                        (0, -1, r'$+1$')]:
        _annotate_node(ax, dx, dy, tag, color='#2E86C1')
    # Connecting bonds.
    for dx, dy in [(+1, 0), (-1, 0), (0, +1), (0, -1)]:
        ax.plot([0, dx], [0, dy], '-', color='0.5',
                lw=1.2, zorder=2)
    # Axis labels indicating the lattice spacing.
    ax.annotate('', xy=(1.0, -1.6), xytext=(0.0, -1.6),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=1.0))
    ax.text(0.5, -1.85, r'$a$', ha='center', va='top',
            fontsize=11)
    ax.set_title(
        r'(a) Exchange: 5-point Laplacian',
        fontsize=12, pad=10)
    # Formula in the lower-right.
    ax.text(
        2.05, 2.05,
        r'$\mathbf{H}_\mathrm{ex} = '
        r'\dfrac{2 A_\mathrm{ex}}{M_s a^{2}}\,'
        r'(\sum_\mathrm{nn} \mathbf{m}_\mathrm{nn}'
        r' - 4\,\mathbf{m}_0)$',
        ha='right', va='top', fontsize=10,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))


def _panel_dmi(ax):
    """Central-difference DMI: +1 on the +x/+y neighbour, -1
    on the -x/-y neighbour, 0 at the centre."""
    _draw_lattice(ax, n=5)
    # Centre node with weight 0 (light fill).
    ax.plot(0, 0, 'o', color='white', markersize=22,
            markeredgecolor='black', zorder=3)
    ax.text(0, 0, r'$0$', ha='center', va='center',
            color='black', fontsize=11, fontweight='bold',
            zorder=4)
    # +x and +y neighbours: weight +1.
    _annotate_node(ax, +1, 0, r'$+1$', color='#27AE60')
    _annotate_node(ax, 0, +1, r'$+1$', color='#27AE60')
    # -x and -y neighbours: weight -1.
    _annotate_node(ax, -1, 0, r'$-1$', color='#7D3C98')
    _annotate_node(ax, 0, -1, r'$-1$', color='#7D3C98')
    # Directed arrows: positive contribution from +x, +y.
    ax.annotate('', xy=(1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27AE60',
                                lw=1.6))
    ax.annotate('', xy=(0, 1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#27AE60',
                                lw=1.6))
    # Directed arrows: negative contribution from -x, -y.
    ax.annotate('', xy=(-1, 0), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#7D3C98',
                                lw=1.6))
    ax.annotate('', xy=(0, -1), xytext=(0, 0),
                arrowprops=dict(arrowstyle='->', color='#7D3C98',
                                lw=1.6))
    ax.set_title(
        r'(b) Interfacial DMI: central difference',
        fontsize=12, pad=10)
    # Formula in the lower-right.
    ax.text(
        2.05, 2.05,
        r'$\mathbf{H}_\mathrm{DMI} = \dfrac{D}{M_s a}\,'
        r'[(m_z^{+\hat{x}}-m_z^{-\hat{x}})\,\hat{x}$'
        '\n'
        r'$\quad + (m_z^{+\hat{y}}-m_z^{-\hat{y}})\,\hat{y}'
        r' - (\partial_x m_x + \partial_y m_y)\,\hat{z}]$',
        ha='right', va='top', fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))


def main():
    out_dir = 'docs/figures'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    _panel_exchange(axes[0])
    _panel_dmi(axes[1])
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out_path = os.path.join(
            out_dir, f'stencil_schematic.{ext}')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved {out_path}')


# =============================================================================
if __name__ == '__main__':
    main()
