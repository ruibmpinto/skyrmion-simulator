"""Step-by-step breakdown of the topological-charge integrand
for theory.tex section 8.

For an isolated Neel skyrmion built from the same analytic
profile used in `make_topcharge_schematic.py`, this script
plots a 3x3 grid:

  row 1: components of d_x m  ->  (d_x m_x, d_x m_y, d_x m_z)
  row 2: components of d_y m  ->  (d_y m_x, d_y m_y, d_y m_z)
  row 3: components of cross  ->  ((d_x m x d_y m)_x,
                                   (d_x m x d_y m)_y,
                                   (d_x m x d_y m)_z)

The final scalar `m . (d_x m x d_y m)` (Pontryagin density)
already lives in panel (b) of the companion
`topcharge_schematic` figure.

Derivatives use the same central-difference / np.roll scheme
employed by `simulator.main.topological_charge`, so the panels
here show exactly the lattice fields the simulator integrates.

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

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _neel_skyrmion(nx, ny, a, R):
    """Analytic Neel skyrmion: theta(r) = pi*(1 - r/R) for r<R."""
    x = (np.arange(nx) - 0.5 * (nx - 1)) * a
    y = (np.arange(ny) - 0.5 * (ny - 1)) * a
    X, Y = np.meshgrid(x, y, indexing='xy')
    r = np.sqrt(X * X + Y * Y)
    phi = np.arctan2(Y, X)
    theta = np.where(r < R, np.pi * (1.0 - r / R), 0.0)
    m = np.empty((ny, nx, 3), dtype=float)
    m[..., 0] = np.sin(theta) * np.cos(phi)
    m[..., 1] = np.sin(theta) * np.sin(phi)
    m[..., 2] = np.cos(theta)
    return m, x, y


def _central_diffs(m, a):
    """Central differences with np.roll, matching
    simulator.main.topological_charge."""
    m_px = np.roll(m, -1, axis=1)
    m_mx = np.roll(m, +1, axis=1)
    m_py = np.roll(m, -1, axis=0)
    m_my = np.roll(m, +1, axis=0)
    dmdx = (m_px - m_mx) / (2.0 * a)
    dmdy = (m_py - m_my) / (2.0 * a)
    return dmdx, dmdy


def _draw_panel(ax, field, x_nm, y_nm, title, units, vsym=True):
    extent = (x_nm[0], x_nm[-1], y_nm[0], y_nm[-1])
    if vsym:
        vmax = float(np.max(np.abs(field)))
        if vmax == 0.0:
            vmax = 1.0
        vmin = -vmax
    else:
        vmin, vmax = float(field.min()), float(field.max())
    im = ax.imshow(field, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                   extent=extent, origin='lower')
    ax.set_title(title, fontsize=11)
    ax.set_aspect('equal')
    ax.set_xticks([-100, 0, 100])
    ax.set_yticks([-100, 0, 100])
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(units, fontsize=9)
    cb.ax.tick_params(labelsize=8)


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nx = 128
    ny = 128
    a = 2.0e-9
    R = 40.0e-9
    m, x, y = _neel_skyrmion(nx, ny, a, R)
    dmdx, dmdy = _central_diffs(m, a)
    cross = np.cross(dmdx, dmdy)
    x_nm = x * 1e9
    y_nm = y * 1e9
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(3, 3, figsize=(11.5, 11.0))
    # Row 1: d_x m
    _draw_panel(axes[0, 0], dmdx[..., 0], x_nm, y_nm,
                r'$\partial_x m_x$', r'(1/m)')
    _draw_panel(axes[0, 1], dmdx[..., 1], x_nm, y_nm,
                r'$\partial_x m_y$', r'(1/m)')
    _draw_panel(axes[0, 2], dmdx[..., 2], x_nm, y_nm,
                r'$\partial_x m_z$', r'(1/m)')
    # Row 2: d_y m
    _draw_panel(axes[1, 0], dmdy[..., 0], x_nm, y_nm,
                r'$\partial_y m_x$', r'(1/m)')
    _draw_panel(axes[1, 1], dmdy[..., 1], x_nm, y_nm,
                r'$\partial_y m_y$', r'(1/m)')
    _draw_panel(axes[1, 2], dmdy[..., 2], x_nm, y_nm,
                r'$\partial_y m_z$', r'(1/m)')
    # Row 3: cross product components
    _draw_panel(axes[2, 0], cross[..., 0], x_nm, y_nm,
                r'$(\partial_x\mathbf{m}\times'
                r'\partial_y\mathbf{m})_x$',
                r'(1/m$^2$)')
    _draw_panel(axes[2, 1], cross[..., 1], x_nm, y_nm,
                r'$(\partial_x\mathbf{m}\times'
                r'\partial_y\mathbf{m})_y$',
                r'(1/m$^2$)')
    _draw_panel(axes[2, 2], cross[..., 2], x_nm, y_nm,
                r'$(\partial_x\mathbf{m}\times'
                r'\partial_y\mathbf{m})_z$',
                r'(1/m$^2$)')
    # Common axis labels only on the outer ring.
    for ax in axes[-1, :]:
        ax.set_xlabel('$x$ (nm)')
    for ax in axes[:, 0]:
        ax.set_ylabel('$y$ (nm)')
    # Row labels on the left.
    for row, label in enumerate([
            r'$\partial_x \mathbf{m}$',
            r'$\partial_y \mathbf{m}$',
            r'$\partial_x\mathbf{m}\times\partial_y\mathbf{m}$']):
        axes[row, 0].text(-0.32, 0.5, label,
                          transform=axes[row, 0].transAxes,
                          rotation=90, va='center', ha='center',
                          fontsize=13, fontweight='bold')
    fig.tight_layout()
    fig.savefig('docs/figures/topcharge_breakdown.pdf',
                bbox_inches='tight')
    fig.savefig('docs/figures/topcharge_breakdown.png',
                dpi=160, bbox_inches='tight')
    print('Saved docs/figures/topcharge_breakdown.{pdf,png}')


# =============================================================================
if __name__ == '__main__':
    main()
