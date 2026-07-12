"""Boundary-condition schematic: periodic vs free-y vs racetrack.

Draws, for a cross-section across the track width (y), how the
magnetostatic (demag) and the exchange/DMI couplings behave at the
top/bottom of the simulation box in three cases:
  - periodic (kind='newell'): images on all sides; demag AND exchange
    wrap top<->bottom;
  - free-y (kind='newell_freebc_y', mask=None): demag isolated across
    y, but the exchange/DMI still wraps (periodic torus);
  - racetrack: demag isolated AND exchange/DMI terminated at a free
    edge (Rohart-Thiaville tilt) over a defined width W.
The x-axis (track length) is periodic in all three.

Functions
---------
make_schematic
    Build the three-panel comparison figure.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _box_with_skyrmion(ax, y0, alpha, label):
    """Draw a unit-height box (y0..y0+2) with a skyrmion disk."""
    ax.add_patch(mpatches.Rectangle(
        (0.0, y0), 2.0, 2.0, fill=False, ec='k', lw=1.4, alpha=alpha))
    ax.add_patch(mpatches.Circle(
        (1.0, y0 + 1.0), 0.5, fc='#2c7bb6', ec='#d7191c', lw=1.2,
        alpha=alpha))
    if label is not None:
        ax.text(1.0, y0 + 1.0, label, ha='center', va='center',
                fontsize=7, color='w', alpha=alpha)


def _wrap_arrow(ax, color, style, text):
    """Curved arrow indicating top<->bottom coupling (y wrap)."""
    ax.annotate(
        '', xy=(2.35, 1.9), xytext=(2.35, 0.1),
        arrowprops=dict(arrowstyle='<->', color=color, ls=style,
                        lw=1.6, connectionstyle='arc3,rad=0.55'))
    ax.text(3.05, 1.0, text, ha='left', va='center', fontsize=7.5,
            color=color, rotation=90)


def _free_edges(ax, y_lo, y_hi):
    """Red Rohart-Thiaville free edges at y_lo and y_hi (arrows out)."""
    for ye, dy in ((y_hi, 0.30), (y_lo, -0.30)):
        ax.plot([0.0, 2.0], [ye, ye], color='#d7191c', lw=3.0)
        for xt in (0.35, 0.95, 1.55):
            ax.annotate('', xy=(xt + 0.18, ye + dy), xytext=(xt, ye),
                        arrowprops=dict(arrowstyle='->',
                                        color='#d7191c', lw=1.2))


def _width_bar(ax, y_lo, y_hi):
    """Left-side double arrow annotating the track width W."""
    ax.annotate('', xy=(-0.35, y_hi), xytext=(-0.35, y_lo),
                arrowprops=dict(arrowstyle='<->', color='k', lw=1.4))
    ax.text(-0.5, 0.5 * (y_lo + y_hi), 'W', ha='right', va='center',
            fontsize=11, fontweight='bold')


def _demag_vacuum(ax):
    """Zero-pad (isolated-demag) vacuum tiles above and below the box."""
    for y0 in (2.6, -2.6):
        ax.add_patch(mpatches.Rectangle(
            (0.0, y0), 2.0, 2.0, fill=True, fc='none',
            ec='0.7', ls=':', lw=1.0, hatch='//'))
        ax.text(1.0, y0 + 1.0, 'vacuum\n(demag\nzero-pad)',
                ha='center', va='center', fontsize=6.5, color='0.45')


def make_schematic(out_path):
    """Render the three-panel BC comparison.

    Parameters
    ----------
    out_path : str
        Output path stem (.pdf and .png are written).
    """
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 5.2))
    titles = ['periodic\n(newell)',
              'free-y demag only\n(isolated demag, exch torus)',
              'racetrack\n(racetrack: full box)',
              'masked band\n(masked_band)']
    captions = [
        'demag-y: periodic images\nexch-y: wraps '
        'top$\\leftrightarrow$bottom',
        'demag-y: isolated (zero-pad)\nexch-y: periodic (torus)',
        'demag-y: isolated (zero-pad)\nexch-y: free edge (R--T tilt)',
        'demag-y: isolated\nexch-y: free edge at band (R--T)']
    for ax, title, cap, mode in zip(
            axes, titles, captions, ['per', 'fby', 'race', 'band']):
        # Central simulation box + skyrmion.
        _box_with_skyrmion(ax, 0.0, 1.0, 'sk')
        if mode == 'per':
            # Periodic images above and below; demag + exchange wrap.
            _box_with_skyrmion(ax, 2.6, 0.22, 'image')
            _box_with_skyrmion(ax, -2.6, 0.22, 'image')
            _wrap_arrow(ax, '#444444', '-', 'demag + exch wrap')
        elif mode == 'fby':
            # Demag isolated (zero-pad vacuum) but exchange still wraps:
            # the demag-only free-y case (not a production BC), shown to
            # motivate why the racetrack must free both couplings.
            _demag_vacuum(ax)
            _wrap_arrow(ax, '#e6750a', '--', 'exchange wraps')
        elif mode == 'race':
            # Full box is the track: demag isolated (zero-pad vacuum),
            # exchange/DMI free edge at the box top/bottom; W = box.
            _demag_vacuum(ax)
            _free_edges(ax, 0.0, 2.0)
            _width_bar(ax, 0.0, 2.0)
        else:
            # masked_band: magnetic band W < box, masked vacuum margins
            # inside the box carry the free edge; demag still isolated.
            _demag_vacuum(ax)
            for y_lo, y_hi in ((0.0, 0.45), (1.55, 2.0)):
                ax.add_patch(mpatches.Rectangle(
                    (0.0, y_lo), 2.0, y_hi - y_lo, fill=True, fc='0.88',
                    ec='0.6', ls=':', lw=0.8, hatch='xx'))
            ax.text(1.0, 0.22, 'masked', ha='center', va='center',
                    fontsize=6.0, color='0.4')
            ax.text(1.0, 1.78, 'masked', ha='center', va='center',
                    fontsize=6.0, color='0.4')
            _free_edges(ax, 0.45, 1.55)
            _width_bar(ax, 0.45, 1.55)
        ax.set_title(title, fontsize=10)
        ax.text(1.0, -3.6, cap, ha='center', va='top', fontsize=8.5)
        ax.text(1.0, 5.1, 'x: periodic', ha='center', va='center',
                fontsize=7.5, color='0.4', style='italic')
        ax.set_xlim(-1.2, 3.8)
        ax.set_ylim(-4.4, 5.4)
        ax.set_aspect('equal')
        ax.axis('off')
    fig.suptitle(
        'Width (y) boundary conditions: magnetostatics vs '
        'exchange/DMI', fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    here = os.path.dirname(os.path.abspath(__file__))
    make_schematic(os.path.join(here, 'bc_schematic'))
    print('wrote bc_schematic.{pdf,png}')
