"""Figure for theory.tex section 8 ('Topological Charge').

Two-panel illustration of a Neel skyrmion and the Pontryagin
density that integrates to Q = -1:

  (a) Real-space spin texture: m_z as a colourmap, m_xy as
      quivers on a coarse subgrid.  The Neel profile used here
      has theta(0) = pi, so the core is m_z = -1 (blue),
      the m_z = 0 domain wall is the white ring, and the
      background is m_z = +1 (red).  Arrows point radially
      outward in the wall (Neel-outward chirality).

  (b) Pontryagin density
      N_xy(r) = m . (d_x m x d_y m),
      the integrand of the topological charge.  For the
      smooth linear theta(r) used here, N_xy is a single-sign
      blob concentrated within r < R rather than a thin
      ring; its sign is set by the sense of m wrapping the
      unit sphere.  The Riemann sum gives the numerical Q
      printed inside the panel (~ -1 here, as expected for a
      Q = -1 skyrmion).

The figure is generated from an analytic Neel profile with
theta(r) = pi*(1 - r/R) for r<R, 0 otherwise, so it does not
import the simulator package.

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
    """Analytic Neel skyrmion on a square lattice.

    Returns
    -------
    m : numpy.ndarray(3d), shape (ny, nx, 3)
    x, y : 1d numpy arrays of lattice site coordinates (m).
    """
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


def _pontryagin_density(m, a):
    """N_xy = m . (d_x m x d_y m) on a periodic-ish grid via
    central differences (interior points only; edges padded
    with zero)."""
    dmx_dx = np.zeros_like(m)
    dmy_dy = np.zeros_like(m)
    # Central differences in x (axis=1) and y (axis=0).
    dmx_dx[:, 1:-1, :] = (m[:, 2:, :] - m[:, :-2, :]) / (2.0 * a)
    dmy_dy[1:-1, :, :] = (m[2:, :, :] - m[:-2, :, :]) / (2.0 * a)
    cross = np.cross(dmx_dx, dmy_dy)
    return np.einsum('...i,...i', m, cross)


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    nx = 128
    ny = 128
    a = 2.0e-9
    R = 40.0e-9
    m, x, y = _neel_skyrmion(nx, ny, a, R)
    N_xy = _pontryagin_density(m, a)
    # Total topological charge by Riemann sum (interior only).
    Q = float(N_xy.sum() * a * a / (4.0 * np.pi))
    print(f'Numerical Q = {Q:+.3f} (expected +/- 1)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    # Coordinates in nm for readability.
    x_nm = x * 1e9
    y_nm = y * 1e9
    extent = (x_nm[0], x_nm[-1], y_nm[0], y_nm[-1])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel A: m_z colourmap + m_xy quivers (subsampled).
    a_ax = axes[0]
    im = a_ax.imshow(m[..., 2], cmap='RdBu_r', vmin=-1.0,
                     vmax=1.0, extent=extent, origin='lower')
    # Subsample the quiver grid to keep arrows readable.
    step = 8
    Xq, Yq = np.meshgrid(x_nm[::step], y_nm[::step], indexing='xy')
    a_ax.quiver(Xq, Yq,
                m[::step, ::step, 0], m[::step, ::step, 1],
                color='k', scale=22, width=0.004,
                headwidth=4.0, headlength=5.0)
    a_ax.set_xlabel('$x$ (nm)')
    a_ax.set_ylabel('$y$ (nm)')
    a_ax.set_title(
        r'(a) Spin texture: $m_z$ (colour) and $\mathbf{m}_{xy}$'
        r' (arrows)')
    a_ax.set_aspect('equal')
    cb = fig.colorbar(im, ax=a_ax, fraction=0.046, pad=0.04)
    cb.set_label(r'$m_z$')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel B: Pontryagin density.
    # Scale so the colourbar reads in 1/nm^2 (more legible than 1/m^2).
    N_per_nm2 = N_xy * 1e-18
    vmax = float(np.max(np.abs(N_per_nm2))) * 0.9
    b_ax = axes[1]
    im2 = b_ax.imshow(N_per_nm2, cmap='RdBu_r',
                      vmin=-vmax, vmax=vmax, extent=extent,
                      origin='lower')
    b_ax.set_xlabel('$x$ (nm)')
    b_ax.set_ylabel('$y$ (nm)')
    b_ax.set_title(
        r'(b) Pontryagin density $N_{xy} = \mathbf{m}\cdot'
        r'(\partial_x\mathbf{m}\times\partial_y\mathbf{m})$')
    b_ax.set_aspect('equal')
    cb2 = fig.colorbar(im2, ax=b_ax, fraction=0.046, pad=0.04)
    cb2.set_label(r'$N_{xy}$ (1/nm$^2$)')
    # Numerical Q annotation inside panel B.
    b_ax.text(0.04, 0.96,
              r'$Q = \dfrac{1}{4\pi}\int N_{xy}\,dA = '
              f'{Q:+.2f}$',
              transform=b_ax.transAxes,
              ha='left', va='top',
              fontsize=11,
              bbox=dict(boxstyle='round,pad=0.3',
                        facecolor='white', edgecolor='0.5',
                        alpha=0.85))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig.tight_layout()
    fig.savefig('docs/figures/topcharge_schematic.pdf',
                bbox_inches='tight')
    fig.savefig('docs/figures/topcharge_schematic.png',
                dpi=180, bbox_inches='tight')
    print('Saved docs/figures/topcharge_schematic.{pdf,png}')


# =============================================================================
if __name__ == '__main__':
    main()
