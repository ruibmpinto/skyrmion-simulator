"""Table of the final driven top-layer m_z configuration for every
(T, J) cell of the racetrack D=0.545 run.

Loads the ens0 anim dump of each cell, renders the last drive-frame
top-layer m_z on a (T rows, J columns) grid at physical scale, and
titles each panel with its stability class (from the field classifier).
Bottom layer is omitted: it mirrors the top under the SAF coupling.

Functions
---------
make_grid
    Build the (T, J) configuration table.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
# Third-party
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), '..', '..'))
# Local
from scripts.plot_track_width import _load_anim

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def make_grid(Ts, Js, cls, dx_lx, dump_dir, a, out_path):
    """Render the final top-layer m_z for every (T, J) cell.

    Parameters
    ----------
    Ts, Js : numpy.ndarray(1d)
        Grid axes (K, A/m^2).
    cls : numpy.ndarray(2d)
        Class-code grid.
    dx_lx : numpy.ndarray(2d)
        D_x / L_x per cell (for the 'p' loss-of-periodicity suffix).
    dump_dir : str
        Directory holding anim_*.npz field dumps.
    a : float
        Lattice constant (m).
    out_path : str
        Output path stem (.png is written; a 36-panel raster contact
        sheet as PDF is needlessly large, so only PNG is emitted).
    """
    a_nm = a * 1e9
    nt, nj = Ts.size, Js.size
    fig, axes = plt.subplots(
        nt, nj, figsize=(2.4 * nj, 3.4 * nt))
    norm = Normalize(vmin=-1.0, vmax=1.0)
    # Rows top-to-bottom = decreasing T; columns = increasing J, so the
    # shared axes read like the stability map: T (K) on y, J on x.
    order = np.argsort(Ts)[::-1]
    for r, i in enumerate(order):
        for k in range(nj):
            ax = axes[r, k]
            fn = os.path.join(
                dump_dir,
                f'anim_T{Ts[i]:05.1f}_j{Js[k]:.2e}.npz')
            code = str(cls[i, k])
            if (code in ('S', 'E') and np.isfinite(dx_lx[i, k])
                    and dx_lx[i, k] >= 0.5):
                code = code + 'p'
            if os.path.isfile(fn):
                na = _load_anim(fn)
                mz = na['mf_top'][..., 2]
                ny, nx = mz.shape
                ax.imshow(
                    mz, origin='lower', cmap='RdBu_r', norm=norm,
                    extent=[0.0, nx * a_nm, 0.0, ny * a_nm],
                    aspect='equal')
            ax.set_xticks([])
            ax.set_yticks([])
            # Stability class in the corner (boxed for legibility).
            ax.text(0.06, 0.94, code, transform=ax.transAxes,
                    ha='left', va='top', fontsize=24, fontweight='bold',
                    bbox=dict(facecolor='white', alpha=0.8,
                              edgecolor='none', pad=2.0))
            # Shared-axis tick labels: T value on the left column, J
            # value on the bottom row.
            if k == 0:
                ax.set_ylabel(f'{Ts[i]:.0f}', fontsize=30, rotation=0,
                              labelpad=30, va='center', ha='right')
            if r == nt - 1:
                ax.set_xlabel(f'{Js[k]*1e-11:.2g}', fontsize=30)
    fig.supxlabel(r'$J$ ($10^{11}$ A/m$^2$)', fontsize=38)
    fig.supylabel(r'$T$ (K)', fontsize=38)
    fig.tight_layout()
    fig.savefig(out_path + '.png', dpi=110, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    run_tag = 'box350x500_racetrack_D0p545'
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    dump_dir = ('/Volumes/T7/skyrmion_simulator/output/'
                'stochastic_llgs/scan_track_width/' + run_tag)
    z = np.load(os.path.join(
        repo, 'output/stochastic_llgs/scan_track_width',
        f'stability_classes_{run_tag}.npz'))
    make_grid(
        z['Ts'], z['Js'], z['cls'], z['dx_lx'], dump_dir, 2.0e-9,
        os.path.join(here, 'config_table'))
    print('wrote config_table.png')
