"""Schematic of the spin Hall effect in a Pt/Co bilayer (2D
side view in the x-z plane).

Generates `she_schematic.{pdf,png}` for use in `docs/theory.tex`
section 6.1.

Layout:
  - x horizontal, z vertical, y out of the page.
  - Pt slab (grey) below, Co layer (blue) on top.
  - Charge current J along +x (horizontal arrow under the Pt).
  - Two electron trajectories: spin-+y (out of page, marker O.)
    deflects toward +z and accumulates at the Pt/Co interface;
    spin--y (into page, marker Ox) deflects toward -z.
  - Spin polarization at the interface marked by a row of
    out-of-page symbols labelled p_hat = z_hat x J_hat = +y_hat.

Functions
---------
main
    Build and save the figure.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _spin_out(ax, x, z, color, size=13):
    """Draw a 'spin out of page' marker (circle with a dot)."""
    ax.plot(x, z, marker='o', color=color, markersize=size,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.plot(x, z, marker='.', color=color, markersize=4)


def _spin_in(ax, x, z, color, size=13):
    """Draw a 'spin into page' marker (circle with a cross)."""
    ax.plot(x, z, marker='o', color=color, markersize=size,
            markerfacecolor='white', markeredgewidth=1.8)
    ax.plot(x, z, marker='x', color=color, markersize=size - 4,
            mew=1.8)


def _arrow(ax, p0, p1, color, lw=2.0):
    ax.annotate('', xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle='->', color=color,
                                lw=lw))


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Geometry (arbitrary units).
    L = 8.0
    t_pt = 2.0
    t_co = 0.5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    # Pt slab.
    ax.add_patch(Rectangle(
        (0, 0), L, t_pt,
        facecolor='lightgray', edgecolor='k', lw=1.2, alpha=0.55))
    # Co slab.
    ax.add_patch(Rectangle(
        (0, t_pt), L, t_co,
        facecolor='lightsteelblue', edgecolor='k', lw=1.2,
        alpha=0.70))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Layer labels (just outside the left edge).
    ax.text(-0.25, t_pt * 0.5, 'Pt',
            fontsize=15, ha='right', va='center')
    ax.text(-0.25, t_pt + t_co * 0.5, 'Co',
            fontsize=15, ha='right', va='center')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Charge current J along +x, drawn below the Pt slab.
    j_z = -0.65
    _arrow(ax, (0.30, j_z), (L - 0.30, j_z),
           color='crimson', lw=2.6)
    ax.text(L / 2.0, j_z - 0.45,
            r'$\mathbf{J}\,\parallel\,+\hat{x}$',
            color='crimson', fontsize=14, ha='center')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Two electron trajectories. Both drift along +x (left to
    # right in the figure); the spin-out-of-page (+y) electron
    # deflects to +z and accumulates at the Pt/Co interface;
    # the spin-into-page (-y) electron deflects to -z and
    # accumulates at the lower Pt surface. The curve is a power
    # law in x so the deflection is gradual and reads cleanly.
    s = np.linspace(0.0, 1.0, 60)
    x_traj = 0.5 + (L - 1.0) * s
    z0 = 0.5 * t_pt
    # Spin-+y trajectory.
    z_up = z0 + (0.45 * t_pt) * s ** 1.5
    ax.plot(x_traj, z_up, color='royalblue', lw=2.0)
    _arrow(ax,
           (x_traj[-3], z_up[-3]),
           (x_traj[-1], z_up[-1]),
           color='royalblue', lw=2.0)
    # Place the +y spin marker about a third along the path.
    # +y_hat points INTO the page in our right-handed frame
    # (x_hat right, z_hat up), so the marker is "into page".
    i1 = len(s) // 3
    _spin_in(ax, x_traj[i1], z_up[i1], 'royalblue')
    ax.text(x_traj[i1], z_up[i1] + 0.30,
            r'spin $+\hat{y}$ (into page)',
            color='royalblue', fontsize=11, ha='center')
    # Spin--y trajectory.
    z_dn = z0 - (0.45 * t_pt) * s ** 1.5
    ax.plot(x_traj, z_dn, color='firebrick', lw=2.0)
    _arrow(ax,
           (x_traj[-3], z_dn[-3]),
           (x_traj[-1], z_dn[-1]),
           color='firebrick', lw=2.0)
    _spin_out(ax, x_traj[i1], z_dn[i1], 'firebrick')
    ax.text(x_traj[i1], z_dn[i1] - 0.40,
            r'spin $-\hat{y}$ (out of page)',
            color='firebrick', fontsize=11, ha='center')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Spin accumulation at the Pt/Co interface: a row of
    # into-page markers along z = t_pt + small offset. p_hat is
    # along +y_hat which is INTO the page in our right-handed
    # frame (x_hat right, z_hat up).
    for x in np.linspace(L * 0.18, L * 0.82, 8):
        _spin_in(ax, x, t_pt + 0.04, 'purple', size=11)
    ax.text(L / 2.0, t_pt + t_co + 0.45,
            r'$\hat{p} = \hat{z}\times\hat{J} = +\hat{y}$',
            color='purple', fontsize=13, ha='center')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Axis triad in the lower-right corner: x_hat, z_hat as
    # arrows; y_hat as out-of-page marker.
    tx = L + 0.40
    tz = -0.50
    _arrow(ax, (tx, tz), (tx + 0.55, tz), color='k', lw=1.2)
    ax.text(tx + 0.65, tz, r'$\hat{x}$',
            fontsize=12, va='center')
    _arrow(ax, (tx, tz), (tx, tz + 0.55), color='k', lw=1.2)
    ax.text(tx, tz + 0.70, r'$\hat{z}$',
            fontsize=12, ha='center')
    _spin_in(ax, tx + 0.32, tz + 0.32, 'k', size=10)
    ax.text(tx + 0.55, tz + 0.32, r'$\hat{y}$',
            fontsize=12, va='center')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cosmetic.
    ax.set_xlim(-1.5, L + 1.5)
    ax.set_ylim(-1.7, t_pt + t_co + 1.4)
    ax.set_aspect('equal')
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig('docs/figures/she_schematic.pdf',
                bbox_inches='tight')
    fig.savefig('docs/figures/she_schematic.png',
                dpi=180, bbox_inches='tight')
    print('Saved docs/figures/she_schematic.{pdf,png}')


# =============================================================================
if __name__ == '__main__':
    main()
