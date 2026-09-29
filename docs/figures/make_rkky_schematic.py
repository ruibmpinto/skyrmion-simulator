"""Generate the SAF RKKY-coupling schematic used in
`docs/theory.tex` (Section 5.6).

Two side-by-side panels:

  Left panel ("local view"): a single (i, j) column showing
  the top Co cell, the Ru/Pt spacer, the bottom Co cell, and
  the antiparallel spins linked by the RKKY field arrows.

  Right panel ("global view"): an x-z cross-section of the
  SAF skyrmion pair, with the top layer carrying a +z core
  and the bottom layer a -z core, AFM-locked at every
  in-plane position.

Outputs
-------
docs/figures/rkky_schematic.png
docs/figures/rkky_schematic.pdf
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import pathlib
# Third-party
import matplotlib.patches as mpatches
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
# Layer-thickness ratios for visual layout (not to scale).
t_co = 1.0
t_sp = 1.0   # Ru + Pt drawn as one block
gap_y = 0.25


def _draw_layer(ax, x0, y0, w, h, color, edge='black',
                label=None, label_color='black'):
    """Draw a single layer block with optional centered label."""
    rect = mpatches.FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle='round,pad=0.0,rounding_size=0.05',
        linewidth=1.2, edgecolor=edge, facecolor=color)
    ax.add_patch(rect)
    if label is not None:
        ax.text(x0 + w/2, y0 + h/2, label,
                ha='center', va='center',
                fontsize=10, color=label_color)


def _draw_arrow(ax, x, y0, y1, color, lw=1.6):
    """Vertical arrow from (x, y0) to (x, y1)."""
    ax.annotate(
        '', xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle='->', color=color,
                        lw=lw, mutation_scale=14),
    )


# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.6),
                         gridspec_kw={'width_ratios': [1.0, 1.6]})

# -----------------------------------------------------------------------------
# Left panel: local (i, j) view of the AFM coupling.
ax = axes[0]
ax.set_xlim(-1.2, 1.2)
ax.set_ylim(-2.4, 2.4)
ax.set_aspect('equal')
ax.axis('off')

# Top Co cell (light orange) at y in [t_sp/2 + gap_y, t_sp/2 + gap_y + t_co]
y_top0 = t_sp / 2 + gap_y
y_top1 = y_top0 + t_co
_draw_layer(ax, -0.6, y_top0, 1.2, t_co, '#ffd9b3',
            label='Top Co', label_color='black')
# Spacer (Ru + Pt) at y in [-t_sp/2, t_sp/2]
_draw_layer(ax, -0.6, -t_sp / 2, 1.2, t_sp, '#d0d0d0',
            label='Ru + Pt spacer', label_color='black')
# Bottom Co cell (light orange) at y in [-(t_sp/2 + gap_y + t_co), ...]
y_bot1 = -t_sp / 2 - gap_y
y_bot0 = y_bot1 - t_co
_draw_layer(ax, -0.6, y_bot0, 1.2, t_co, '#ffd9b3',
            label='Bottom Co', label_color='black')

# Spin arrows: top up, bottom down.
_draw_arrow(ax, 0.0, y_top0 + 0.1, y_top1 - 0.1,
            'tab:red', lw=2.4)
_draw_arrow(ax, 0.0, y_bot1 - 0.1, y_bot0 + 0.1,
            'tab:blue', lw=2.4)

# RKKY-field arrows: red field on top points down (toward
# antiparallel target), blue field on bottom points up.
arrow_x = 0.85
_draw_arrow(ax, arrow_x, y_top1 - 0.1, y_top0 + 0.1,
            'tab:blue', lw=1.4)
_draw_arrow(ax, arrow_x, y_bot0 + 0.1, y_bot1 - 0.1,
            'tab:red', lw=1.4)
ax.text(arrow_x + 0.16,
        (y_top0 + y_top1) / 2,
        r'$-H_{\rm RKKY}\,\mathbf{m}^{\rm bot}$',
        ha='left', va='center', fontsize=9,
        color='tab:blue')
ax.text(arrow_x + 0.16,
        (y_bot0 + y_bot1) / 2,
        r'$-H_{\rm RKKY}\,\mathbf{m}^{\rm top}$',
        ha='left', va='center', fontsize=9,
        color='tab:red')

ax.text(0, 2.25, 'Local view at site $(i,j)$',
        ha='center', va='center', fontsize=11,
        fontweight='bold')

# -----------------------------------------------------------------------------
# Right panel: x-z cross-section of a SAF skyrmion pair.
ax = axes[1]
nx = 24
x = np.linspace(-1, 1, nx)
# Skyrmion profile: m_z = tanh of (x - center)/Delta with sign for
# the top layer; opposite sign for the bottom layer.
R = 0.5
Delta = 0.12
mz_top = np.tanh((R - np.abs(x)) / Delta)   # +1 in core, -1 outside
mz_bot = -mz_top                            # AFM locked
# in-plane component for visual (Neel skyrmion radial)
mx_top = np.where(x > 0, -np.sqrt(np.maximum(1 - mz_top**2, 0.0)),
                  +np.sqrt(np.maximum(1 - mz_top**2, 0.0)))
mx_bot = -mx_top

ax.set_xlim(-1.25, 1.25)
ax.set_ylim(-2.4, 2.4)
ax.set_aspect('equal')
ax.axis('off')

# Layer bands (full width)
_draw_layer(ax, -1.15, y_top0, 2.3, t_co, '#ffe9d4')
_draw_layer(ax, -1.15, -t_sp / 2, 2.3, t_sp, '#e0e0e0',
            label='spacer (1.35 nm)', label_color='black')
_draw_layer(ax, -1.15, y_bot0, 2.3, t_co, '#ffe9d4')

# Top-layer arrows
y_top_mid = (y_top0 + y_top1) / 2
for i in range(0, nx, 2):
    xi = x[i]
    dz = 0.35 * mz_top[i]
    dx = 0.18 * mx_top[i]
    color = 'tab:red' if mz_top[i] > 0 else 'tab:blue'
    ax.annotate('', xy=(xi + dx, y_top_mid + dz),
                xytext=(xi - dx, y_top_mid - dz),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=1.3,
                    mutation_scale=10))
# Bottom-layer arrows
y_bot_mid = (y_bot0 + y_bot1) / 2
for i in range(0, nx, 2):
    xi = x[i]
    dz = 0.35 * mz_bot[i]
    dx = 0.18 * mx_bot[i]
    color = 'tab:red' if mz_bot[i] > 0 else 'tab:blue'
    ax.annotate('', xy=(xi + dx, y_bot_mid + dz),
                xytext=(xi - dx, y_bot_mid - dz),
                arrowprops=dict(
                    arrowstyle='->', color=color, lw=1.3,
                    mutation_scale=10))

# Q labels
ax.text(-1.05, y_top_mid + 0.6, '$Q_{\\rm top}=-1$',
        ha='left', va='center', fontsize=10)
ax.text(-1.05, y_bot_mid - 0.6, '$Q_{\\rm bot}=+1$',
        ha='left', va='center', fontsize=10)
ax.text(1.05, y_top_mid + 0.6, 'top: $+z$ core',
        ha='right', va='center', fontsize=9, color='tab:red')
ax.text(1.05, y_bot_mid - 0.6, 'bot: $-z$ core',
        ha='right', va='center', fontsize=9, color='tab:blue')

ax.text(0, 2.25, 'SAF skyrmion pair (AFM-locked)',
        ha='center', va='center', fontsize=11,
        fontweight='bold')

# -----------------------------------------------------------------------------
fig.tight_layout()
out_dir = pathlib.Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'rkky_schematic.png', dpi=180)
fig.savefig(out_dir / 'rkky_schematic.pdf')
print(f'wrote {out_dir / "rkky_schematic.png"}')
print(f'wrote {out_dir / "rkky_schematic.pdf"}')
