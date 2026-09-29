"""Schematic phase diagram for an antiparallel SAF.

Draws the textbook chiral-magnet phase regions in reduced
units (D / D_c, H_z / H_K) adapted for the synthetic
antiferromagnet: an *antiparallel manifold* below the
spin-flop (|H_z/H_K| < H_RKKY/H_K) and a *parallel
manifold* above. Inside the antiparallel manifold the
Zeeman term cancels, so the boundaries are vertical (depend
only on D/D_c). Above the spin-flop the system reverts to
a conventional chiral ferromagnet with PMA.

Each IC from `sweep._ic_specs()` is annotated at the
position where its named structure lives in this diagram --
the seed's target phase, before any LLGS relaxation.

The boundaries drawn here are *schematic*, not computed
from the energy functional. They reflect the canonical
sequence (Bogdanov & Hubert 1994; Sampaio 2013; Buttner,
Lemesh & Beach 2018) for a chiral magnet with Q_PMA ~ 0.25.
Use it as a teaching reference and as a comparison against
the actual phase map a sweep produces.

Usage
-----
Edit the boundary positions in `main()`'s User
Configuration block (if you want to fit a different
material), then run:

    python -m studies.saf_racetrack.phase_diagram.show_phases

Output: `output/phase_diagram/schematic_phase_diagram.png`.

Functions
---------
draw_schematic
    Render the phase regions, axes, and dividing lines on
    a matplotlib axes.
mark_ics
    Overlay IC markers and labels for each entry in
    `_ic_locations`.
main
    Read the User Configuration block and write the PNG.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import sys
# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
# Local
from skyrmion_simulator.phase_diagram.plot_phase_diagram import _PHASE_COLORS

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================
# IC seed -> (D/D_c, H_z/H_K) location in the schematic.
# These are the phase regions each IC explicitly targets
# (the seed's intended basin), not where a real
# relaxation necessarily lands.
_ic_locations = {
    'fm_anti':        (0.40,  0.00),
    'fm_par+':        (0.40, +0.85),
    'fm_par-':        (0.40, -0.85),
    'skyrmion':       (0.97,  0.00),
    'sk_lattice':     (1.25,  0.00),
    'bubble_lattice': (1.05, +0.30),
    'stripe (x)':     (1.75, +0.20),
    'stripe (y)':     (1.75, -0.20),
}


def draw_schematic(ax, y_sf, x_chi, x_iSk_to_SkX,
                   x_SkX_to_SS,
                   x_lim, y_lim):
    """Render phase regions, spin-flop, and chiral threshold.

    Parameters
    ----------
    ax : matplotlib axes
        Target axes; cleared and reused.
    y_sf : float
        Spin-flop position in H_z/H_K. Antiparallel manifold
        is |y| <= y_sf.
    x_chi : float
        Bogdanov-Hubert chiral threshold in D/D_c.
    x_iSk_to_SkX, x_SkX_to_SS : float
        Boundary positions inside the antiparallel manifold
        separating iSk -> SkX -> SS as D/D_c increases.
    x_lim, y_lim : tuple of float
        Axis ranges.
    """
    ax.set_xlim(*x_lim)
    ax.set_ylim(*y_lim)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Antiparallel manifold (|y| < y_sf): vertical bands
    # because Zeeman is identically zero here.
    bands = [
        ('FM_anti', x_lim[0],     x_chi),
        ('iSk',     x_chi,        x_iSk_to_SkX),
        ('SkX',     x_iSk_to_SkX, x_SkX_to_SS),
        ('SS',      x_SkX_to_SS,  x_lim[1]),
    ]
    for label, x0, x1 in bands:
        ax.add_patch(mpatches.Rectangle(
            (x0, -y_sf), x1 - x0, 2.0 * y_sf,
            facecolor=_PHASE_COLORS[label],
            edgecolor='none', alpha=0.55,
        ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parallel manifold (|y| > y_sf): field-aligned FM in
    # the simplest schematic. (A real diagram opens narrow
    # iSk/SkX windows just above |y| = y_sf for moderate D;
    # those are below the resolution of this schematic.)
    ax.add_patch(mpatches.Rectangle(
        (x_lim[0], y_sf),
        x_lim[1] - x_lim[0], y_lim[1] - y_sf,
        facecolor=_PHASE_COLORS['FM_par+'],
        edgecolor='none', alpha=0.55,
    ))
    ax.add_patch(mpatches.Rectangle(
        (x_lim[0], y_lim[0]),
        x_lim[1] - x_lim[0], -y_sf - y_lim[0],
        facecolor=_PHASE_COLORS['FM_par-'],
        edgecolor='none', alpha=0.55,
    ))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dividers
    ax.axhline(+y_sf, color='k', lw=1.5, ls='--', alpha=0.75)
    ax.axhline(-y_sf, color='k', lw=1.5, ls='--', alpha=0.75)
    ax.axvline(x_chi, color='k', lw=1.2, ls=':', alpha=0.7)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Region labels
    band_centers = {
        'FM_anti': 0.5 * (x_lim[0] + x_chi),
        'iSk':     0.5 * (x_chi + x_iSk_to_SkX),
        'SkX':     0.5 * (x_iSk_to_SkX + x_SkX_to_SS),
        'SS':      0.5 * (x_SkX_to_SS + x_lim[1]),
    }
    for label, xc in band_centers.items():
        ax.text(xc, 0.0, label,
                ha='center', va='center',
                fontsize=11, fontweight='bold')
    ax.text(0.5 * (x_lim[0] + x_lim[1]),
            0.5 * (y_sf + y_lim[1]),
            'FM_par+ (parallel manifold)',
            ha='center', va='center',
            fontsize=11, fontweight='bold')
    ax.text(0.5 * (x_lim[0] + x_lim[1]),
            0.5 * (-y_sf + y_lim[0]),
            'FM_par- (parallel manifold)',
            ha='center', va='center',
            fontsize=11, fontweight='bold')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Boundary annotations
    ax.text(x_lim[1] - 0.02, +y_sf + 0.02,
            f'spin-flop  $|H_z/H_K|$ = {y_sf:.2f}',
            ha='right', va='bottom', fontsize=8,
            style='italic')
    ax.text(x_lim[1] - 0.02, -y_sf - 0.05,
            f'spin-flop  $|H_z/H_K|$ = {y_sf:.2f}',
            ha='right', va='top', fontsize=8,
            style='italic')
    ax.text(x_chi + 0.02, y_lim[0] + 0.05,
            r'$D = D_c$',
            ha='left', va='bottom', fontsize=8,
            style='italic')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Manifold annotations (left side)
    ax.text(x_lim[0] + 0.05, 0.5 * y_sf,
            'antiparallel\nmanifold',
            ha='left', va='center',
            fontsize=8, style='italic', alpha=0.7)
    ax.text(x_lim[0] + 0.05, 0.5 * (y_sf + y_lim[1]),
            'parallel\nmanifold (+)',
            ha='left', va='center',
            fontsize=8, style='italic', alpha=0.7)
    ax.text(x_lim[0] + 0.05, 0.5 * (-y_sf + y_lim[0]),
            'parallel\nmanifold (-)',
            ha='left', va='center',
            fontsize=8, style='italic', alpha=0.7)
    ax.set_xlabel(r'$D / D_c$', fontsize=12)
    ax.set_ylabel(r'$H_z / H_K$', fontsize=12)


# ---------------------------------------------------------------------
def mark_ics(ax):
    """Overlay IC markers and labels."""
    for name, (x, y) in _ic_locations.items():
        ax.plot(
            x, y, marker='*', markersize=16,
            markerfacecolor='white',
            markeredgecolor='black',
            markeredgewidth=1.2,
            linestyle='none',
            zorder=4,
        )
        ax.annotate(
            name, (x, y),
            textcoords='offset points',
            xytext=(10, 8),
            fontsize=9, fontweight='bold',
            bbox=dict(
                boxstyle='round,pad=0.25',
                facecolor='white', alpha=0.85,
                edgecolor='black', linewidth=0.5,
            ),
            zorder=5,
        )
    # Annotate the random ensemble as exploratory.
    ax.text(
        0.05, -0.97,
        '5 × random ICs probe every basin (no fixed target)',
        ha='left', va='top',
        fontsize=8, style='italic',
        transform=ax.transAxes,
        bbox=dict(
            boxstyle='round,pad=0.25',
            facecolor='white', alpha=0.85,
            edgecolor='grey', linewidth=0.5,
        ),
    )


# ---------------------------------------------------------------------
def main():
    """Render the schematic phase diagram."""
    # ================ User Configuration ================
    # Schematic boundary positions in reduced units. Tune
    # these to match a target material (default values are
    # the textbook chiral-magnet sequence for Q_PMA ~ 0.25
    # SAF).
    y_sf            = 0.47   # H_RKKY/H_K for K=1.6 MJ/m^3
    x_chi           = 1.0    # Bogdanov-Hubert threshold
    x_iSk_to_SkX    = 1.10   # iSk -> SkX (D/D_c)
    x_SkX_to_SS     = 1.55   # SkX -> SS (D/D_c)
    x_lim           = (0.0, 2.10)
    y_lim           = (-1.13, 1.13)
    out_path = (
        'output/phase_diagram/schematic_phase_diagram.png'
    )
    # ============ End User Configuration =================
    fig, ax = plt.subplots(figsize=(11.0, 9.5))
    ax.set_aspect('auto')
    draw_schematic(
        ax,
        y_sf=y_sf, x_chi=x_chi,
        x_iSk_to_SkX=x_iSk_to_SkX,
        x_SkX_to_SS=x_SkX_to_SS,
        x_lim=x_lim, y_lim=y_lim,
    )
    mark_ics(ax)
    ax.set_title(
        'Schematic phase diagram, antiparallel SAF '
        '(engineered PMA, $Q_{\\mathrm{PMA}} \\sim 0.25$)',
        fontsize=13,
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'Wrote {out_path}')


# =====================================================================
if __name__ == '__main__':
    sys.exit(main())
