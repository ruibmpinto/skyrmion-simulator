"""Relaxed-skyrmion configuration snapshots and convergence figures.

Renders, for each box size of the relaxation-torque validator
(quadrature Newell demag, Set-A SAF, J=0, 250000 steps), the relaxed
top-layer m_z map at true physical scale, plus the size/torque
convergence-vs-step and the box-size scaling of the relaxed diameter
and the residual-torque floor.

Functions
---------
make_snapshots
    Build the per-box m_z configuration panel.
make_convergence
    Build the D/tau-vs-step and box-scaling panel.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def make_snapshots(boxes, in_dir, out_path):
    """Render relaxed top-layer m_z for each box at physical scale.

    Parameters
    ----------
    boxes : tuple[tuple[int]]
        (nx, ny) pairs to load as box_{nx}x{ny}.npz.
    in_dir : str
        Directory holding the per-box NPZ files.
    out_path : str
        Output path stem (.pdf and .png are written).
    """
    fig, axes = plt.subplots(
        1, len(boxes), figsize=(4.2 * len(boxes), 4.4))
    if len(boxes) == 1:
        axes = [axes]
    norm = Normalize(vmin=-1.0, vmax=1.0)
    im = None
    for ax, (nx, ny) in zip(axes, boxes):
        z = np.load(os.path.join(in_dir, f'box_{nx}x{ny}.npz'))
        a_nm = float(z['a']) * 1e9
        mz = np.asarray(z['mz'], dtype=float)
        lx = nx * a_nm
        ly = ny * a_nm
        d1 = float(z['D1'][-1]) * 1e9
        im = ax.imshow(
            mz, origin='lower', cmap='RdBu_r', norm=norm,
            extent=[0.0, lx, 0.0, ly], aspect='equal')
        ax.set_title(
            f'{nx}$\\times${ny} cells\n'
            f'{lx:.0f}$\\times${ly:.0f} nm, $D$={d1:.0f} nm',
            fontsize=11)
        ax.set_xlabel('$x$ (nm)')
        ax.set_ylabel('$y$ (nm)')
    cbar = fig.colorbar(
        im, ax=axes, fraction=0.046, pad=0.04, shrink=0.85)
    cbar.set_label('$m_z$ (top layer)')
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# -----------------------------------------------------------------------------
def make_convergence(boxes, in_dir, out_path):
    """Size/torque convergence vs step and box-size scaling.

    Parameters
    ----------
    boxes : tuple[tuple[int]]
        (nx, ny) pairs to load as box_{nx}x{ny}.npz.
    in_dir : str
        Directory holding the per-box NPZ files.
    out_path : str
        Output path stem (.pdf and .png are written).
    """
    fig, (ax_d, ax_t, ax_s) = plt.subplots(1, 3, figsize=(13.5, 4.0))
    final_d = []
    final_tau = []
    labels = []
    for nx, ny in boxes:
        z = np.load(os.path.join(in_dir, f'box_{nx}x{ny}.npz'))
        steps = np.asarray(z['steps'], dtype=float)
        d1 = np.asarray(z['D1'], dtype=float) * 1e9
        tau = np.asarray(z['tau_max'], dtype=float)
        label = f'{nx}$\\times${ny}'
        ax_d.plot(steps, d1, marker='.', ms=3, label=label)
        ax_t.semilogy(steps, tau, marker='.', ms=3, label=label)
        final_d.append(d1[-1])
        final_tau.append(tau[-1])
        labels.append(label)
    ax_d.set_xlabel('relaxation step')
    ax_d.set_ylabel('$D$ (nm)')
    ax_d.set_title('(a) diameter convergence')
    ax_d.legend(fontsize=9)
    ax_t.set_xlabel('relaxation step')
    ax_t.set_ylabel(r'$\max|\boldsymbol{\tau}|$ (arb.)')
    ax_t.set_title('(b) residual-torque floor')
    ax_t.legend(fontsize=9)
    n_cells = [nx * ny for nx, ny in boxes]
    ax_s.plot(n_cells, final_d, 'o-', label='$D$ (nm)')
    ax_s.set_xlabel('lattice cells $N_x N_y$')
    ax_s.set_ylabel('relaxed $D$ (nm)')
    ax_s.set_title('(c) box-size independence')
    ax_s.set_ylim(min(final_d) - 5.0, max(final_d) + 5.0)
    for n, d, lab in zip(n_cells, final_d, labels):
        ax_s.annotate(lab, (n, d), fontsize=8,
                      textcoords='offset points', xytext=(0, 6))
    fig.tight_layout()
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    # Run configuration: a DMI value (mJ/m^2) and a boundary-condition
    # subdirectory select the input run; both the input directory and
    # the output figure names are tagged by D so runs at different D
    # coexist. Edit these to render a different relaxation run.
    dmi_mj = 0.85
    bc = 'newell'
    boxes = ((256, 256), (350, 350), (350, 500), (500, 500))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    d_tag = f'D{dmi_mj:g}'.replace('.', 'p')
    in_dir = os.path.join(
        repo, 'output/stochastic_llgs/validation/relax_torque',
        d_tag, bc)
    stem = f'_{d_tag}_{bc}'
    make_snapshots(
        boxes, in_dir,
        os.path.join(here, 'relax_box_snapshots' + stem))
    make_convergence(
        boxes, in_dir,
        os.path.join(here, 'relax_box_convergence' + stem))
    print(f'wrote relax_box_snapshots{stem}.{{pdf,png}} and '
          f'relax_box_convergence{stem}.{{pdf,png}}')
