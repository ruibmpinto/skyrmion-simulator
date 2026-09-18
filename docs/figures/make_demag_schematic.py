"""Render the three demagnetization approaches used by the simulator.

Three panels:

(a) Local K_eff (no FFT demag): each cell sees only the
    out-of-plane self-demag of an infinite slab,
    -mu0 M_s^2 / 2 m_z, absorbed into the effective
    anisotropy K_eff = K - mu0 M_s^2 / 2. Local, per-site.

(b) Slab kernel (FFT): each layer is treated as a continuous
    slab of thickness t_Co with PBC in plane. Kernel in
    k-space is the thin-film shape factor
        f(k, t) = (1 - exp(-|k| t)) / (|k| t).
    Diagonal N_xx, N_yy, N_zz only; no inter-layer N_xz/N_yz.

(c) Newell kernel (FFT): each cell is a finite a x a x t_Co
    rectangular prism; the cell-cell tensor is built by
    Gauss-Legendre quadrature of the surface-charge
    formulation (mumax3-style). Captures inter-layer N_xz,
    N_yz cross-terms.

Output
------
docs/figures/demag_schematic.pdf (and .png)
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, \
    Rectangle, Polygon

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

plt.rcParams['figure.dpi'] = 200
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


def _clean_axes(ax, xlim, ylim):
    """Strip ticks/spines and lock limits + aspect."""
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _panel_keff(ax):
    """Local K_eff: single column with surface charges + arrow."""
    _clean_axes(ax, xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    # Layer slab cross-section (cell width a, thickness t_Co).
    top = Rectangle((-0.4, -0.4), 0.8, 0.8,
                    facecolor='#D6EAF8',
                    edgecolor='#1F618D', lw=1.5)
    ax.add_patch(top)
    # Surface magnetic charges (top + sigma, bottom - sigma).
    for x in np.linspace(-0.3, 0.3, 4):
        ax.plot(x, 0.4, '+', color='#C0392B', markersize=10,
                markeredgewidth=1.8)
        ax.plot(x, -0.4, '_', color='#1F618D', markersize=10,
                markeredgewidth=1.8)
    # Surface-charge labels.
    ax.text(0.55, 0.4, r'$+\sigma_m$', color='#C0392B',
            fontsize=13, va='center')
    ax.text(0.55, -0.4, r'$-\sigma_m$', color='#1F618D',
            fontsize=13, va='center')
    # m_z arrow inside the cell.
    ax.annotate('', xy=(0, 0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle='-|>',
                                color='black', lw=1.6))
    ax.text(-0.15, 0.0, r'$m_z$', fontsize=14, ha='right',
            va='center')
    # Formula bubble.
    ax.text(
        0.0, -1.0,
        r'$\mathbf{H}_\mathrm{demag} \approx '
        r'-\mu_0 M_s\,m_z\,\hat{z}$',
        ha='center', va='top', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))
    ax.text(
        0.0, -1.35,
        r'$K_\mathrm{eff} = K - \frac{1}{2}\mu_0 M_s^{2}$',
        ha='center', va='top', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))
    ax.set_title(r'(a) Local $K_\mathrm{eff}$ (no FFT)',
                 fontsize=15, pad=10)


def _panel_slab(ax):
    """Slab kernel: two continuous slabs separated by Ru."""
    _clean_axes(ax, xlim=(-1.6, 1.6), ylim=(-1.6, 1.6))
    # Top Co slab.
    top = Rectangle((-1.2, 0.3), 2.4, 0.35,
                    facecolor='#FADBD8',
                    edgecolor='#943126', lw=1.5)
    ax.add_patch(top)
    # Bottom Co slab.
    bot = Rectangle((-1.2, -0.65), 2.4, 0.35,
                    facecolor='#FADBD8',
                    edgecolor='#943126', lw=1.5)
    ax.add_patch(bot)
    # Ru spacer between them.
    ru = Rectangle((-1.2, -0.30), 2.4, 0.6,
                   facecolor='#F4F6F7',
                   edgecolor='#566573', lw=1.0, hatch='....')
    ax.add_patch(ru)
    # Layer labels.
    ax.text(-1.3, 0.475, r'$t_\mathrm{Co}$', fontsize=13,
            ha='right', va='center')
    ax.text(-1.3, 0.0, r'$d_\mathrm{Ru}$', fontsize=13,
            ha='right', va='center')
    ax.text(-1.3, -0.475, r'$t_\mathrm{Co}$', fontsize=13,
            ha='right', va='center')
    # Continuous in-plane charge sheets (top of each Co layer).
    for x in np.linspace(-1.0, 1.0, 7):
        ax.plot(x, 0.65, '+', color='#C0392B', markersize=8,
                markeredgewidth=1.4)
        ax.plot(x, 0.30, '_', color='#1F618D', markersize=8,
                markeredgewidth=1.4)
        ax.plot(x, -0.30, '+', color='#C0392B', markersize=8,
                markeredgewidth=1.4)
        ax.plot(x, -0.65, '_', color='#1F618D', markersize=8,
                markeredgewidth=1.4)
    # k-space wavevector annotation.
    ax.annotate('', xy=(1.1, 1.05), xytext=(-1.1, 1.05),
                arrowprops=dict(arrowstyle='<->', color='black',
                                lw=1.1))
    ax.text(0.0, 1.18, r'in-plane FFT, wavevector $\mathbf{k}$',
            ha='center', va='bottom', fontsize=9)
    # Shape factor formula.
    ax.text(
        0.0, -1.15,
        r'$f(k, t_\mathrm{Co}) = '
        r'\dfrac{1 - e^{-|k|\, t_\mathrm{Co}}}'
        r'{|k|\, t_\mathrm{Co}}$',
        ha='center', va='top', fontsize=13,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))
    ax.set_title('(b) Slab kernel (FFT)',
                 fontsize=15, pad=10)


def _panel_newell(ax):
    """Newell kernel: 3D-style prism per cell with cross-arrows."""
    _clean_axes(ax, xlim=(-1.7, 1.7), ylim=(-1.7, 1.7))
    # Upper prism (top layer): front face + extruded top/right.
    cx, cy = -0.25, 0.35
    w, h = 0.70, 0.45
    depth_dx, depth_dy = 0.45, 0.35
    front = Rectangle((cx - w / 2, cy - h / 2), w, h,
                      facecolor='#D5F5E3',
                      edgecolor='#1E8449', lw=1.5, zorder=2)
    ax.add_patch(front)
    top_face = Polygon([
        (cx - w / 2, cy + h / 2),
        (cx + w / 2, cy + h / 2),
        (cx + w / 2 + depth_dx, cy + h / 2 + depth_dy),
        (cx - w / 2 + depth_dx, cy + h / 2 + depth_dy)],
        closed=True, facecolor='#ABEBC6',
        edgecolor='#1E8449', lw=1.5, zorder=3)
    ax.add_patch(top_face)
    right_face = Polygon([
        (cx + w / 2, cy - h / 2),
        (cx + w / 2, cy + h / 2),
        (cx + w / 2 + depth_dx, cy + h / 2 + depth_dy),
        (cx + w / 2 + depth_dx, cy - h / 2 + depth_dy)],
        closed=True, facecolor='#7DCEA0',
        edgecolor='#1E8449', lw=1.5, zorder=3)
    ax.add_patch(right_face)
    # Width "a" below the prism.
    ax.annotate(
        '', xy=(cx + w / 2, cy - h / 2 - 0.10),
        xytext=(cx - w / 2, cy - h / 2 - 0.10),
        arrowprops=dict(arrowstyle='<->', color='black',
                        lw=1.0))
    ax.text(cx, cy - h / 2 - 0.22, r'$a$', ha='center',
            va='top', fontsize=13)
    # Height "t_Co" along the front-face right edge.
    ax.annotate(
        '', xy=(cx + w / 2 + 0.12, cy - h / 2),
        xytext=(cx + w / 2 + 0.12, cy + h / 2),
        arrowprops=dict(arrowstyle='<->', color='black',
                        lw=1.0))
    ax.text(cx + w / 2 + 0.20, cy, r'$t_\mathrm{Co}$',
            ha='left', va='center', fontsize=13)
    # Depth "a" along the receding top-right edge, with the
    # label placed clearly above-right to avoid the t_Co tag.
    ax.annotate(
        '', xy=(cx + w / 2 + depth_dx,
                cy + h / 2 + depth_dy),
        xytext=(cx + w / 2, cy + h / 2),
        arrowprops=dict(arrowstyle='<->', color='black',
                        lw=1.0))
    ax.text(cx + w / 2 + depth_dx + 0.10,
            cy + h / 2 + depth_dy + 0.05,
            r'$a$', ha='left', va='bottom', fontsize=13)
    # Lower companion prism (bottom layer of the SAF).
    cy2 = -0.85
    front2 = Rectangle((cx - w / 2, cy2 - h / 2), w, h,
                       facecolor='#FCF3CF',
                       edgecolor='#B7950B', lw=1.5, zorder=2)
    ax.add_patch(front2)
    top2 = Polygon([
        (cx - w / 2, cy2 + h / 2),
        (cx + w / 2, cy2 + h / 2),
        (cx + w / 2 + depth_dx, cy2 + h / 2 + depth_dy),
        (cx - w / 2 + depth_dx, cy2 + h / 2 + depth_dy)],
        closed=True, facecolor='#FAE5D3',
        edgecolor='#B7950B', lw=1.5, zorder=3)
    ax.add_patch(top2)
    right2 = Polygon([
        (cx + w / 2, cy2 - h / 2),
        (cx + w / 2, cy2 + h / 2),
        (cx + w / 2 + depth_dx, cy2 + h / 2 + depth_dy),
        (cx + w / 2 + depth_dx, cy2 - h / 2 + depth_dy)],
        closed=True, facecolor='#F5CBA7',
        edgecolor='#B7950B', lw=1.5, zorder=3)
    ax.add_patch(right2)
    # Inter-layer cross-term arrow on the left of both prisms,
    # well away from the dimension labels on the right.
    cross = FancyArrowPatch(
        (cx - w / 2 - 0.15, cy - h / 2),
        (cx - w / 2 - 0.15, cy2 + h / 2),
        arrowstyle='<->', color='#7D3C98', lw=1.6,
        mutation_scale=14)
    ax.add_patch(cross)
    ax.text(cx - w / 2 - 0.22,
            0.5 * (cy - h / 2 + cy2 + h / 2),
            r'$N_{xz},\,N_{yz}$', color='#7D3C98',
            fontsize=13, ha='right', va='center')
    # Quadrature caption below the lower prism.
    ax.text(
        0.0, -1.55,
        r'Gauss-Legendre over each cell face',
        ha='center', va='top', fontsize=15,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.95))
    ax.set_title('(c) Newell kernel (FFT)',
                 fontsize=15, pad=10)


def main():
    out_dir = 'docs/figures'
    os.makedirs(out_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.8))
    _panel_keff(axes[0])
    _panel_slab(axes[1])
    _panel_newell(axes[2])
    fig.tight_layout()
    for ext in ('pdf', 'png'):
        out_path = os.path.join(
            out_dir, f'demag_schematic.{ext}')
        fig.savefig(out_path, bbox_inches='tight')
        print(f'Saved {out_path}')


# =============================================================================
if __name__ == '__main__':
    main()
