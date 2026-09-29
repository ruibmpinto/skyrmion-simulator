"""Free-y (mixed track BC) relaxation: the D>D_c expansion.

Renders, for the relaxation-torque validator run with periodic-x /
free-y boundaries (newell_freebc_y, Set-A, D=0.85 mJ/m^2, J=0), the
skyrmion diameter and maximum torque versus relaxation step for each
box. Unlike the periodic case (which plateaus at a box-independent
206.5 nm), the free-y skyrmion expands monotonically and never settles
within 250000 steps -- the signature of the D>D_c regime in which the
isolated skyrmion has no finite equilibrium radius.

Functions
---------
make_expansion
    Build the diameter / torque vs step panel.
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


def make_expansion(boxes, in_dir, periodic_ref_nm, out_path):
    """Diameter and max-torque vs step for the free-y relaxation.

    Parameters
    ----------
    boxes : tuple[tuple[int]]
        (nx, ny) pairs to load as box_{nx}x{ny}.npz.
    in_dir : str
        Directory with the newell_freebc_y per-box NPZ files.
    periodic_ref_nm : float
        Converged periodic-BC diameter (nm), drawn as a reference.
    out_path : str
        Output path stem (.pdf and .png are written).
    """
    fig, (ax_d, ax_t) = plt.subplots(1, 2, figsize=(12.0, 4.6))
    for nx, ny in boxes:
        z = np.load(os.path.join(in_dir, f'box_{nx}x{ny}.npz'))
        step = np.asarray(z['steps'], dtype=float)
        d1 = np.asarray(z['D1'], dtype=float) * 1e9
        tau = np.asarray(z['tau_max'], dtype=float)
        label = f'{nx}$\\times${ny}'
        ax_d.plot(step, d1, marker='.', ms=3, label=label)
        ax_t.semilogy(step, tau, marker='.', ms=3, label=label)
    ax_d.axhline(
        periodic_ref_nm, ls='--', color='k', lw=1.0,
        label=f'periodic ({periodic_ref_nm:.0f} nm)')
    ax_d.set_xlabel('relaxation step')
    ax_d.set_ylabel('$D_1$ (nm)')
    ax_d.set_title('(a) diameter: monotonic expansion')
    ax_d.legend(fontsize=8)
    ax_d.grid(True, alpha=0.3)
    ax_t.set_xlabel('relaxation step')
    ax_t.set_ylabel(r'$\max|\boldsymbol{\tau}|$ (arb.)')
    ax_t.set_title('(b) residual torque (edge-limited)')
    ax_t.legend(fontsize=8)
    ax_t.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path + '.pdf', bbox_inches='tight')
    fig.savefig(out_path + '.png', dpi=150, bbox_inches='tight')
    plt.close(fig)


# =============================================================================
if __name__ == '__main__':
    # Run configuration: DMI (mJ/m^2) and boundary-condition subdir
    # select the input run; the input directory and output figure name
    # are tagged by D so runs at different D coexist. periodic_ref_nm is
    # the converged periodic-BC diameter drawn as a reference line.
    dmi_mj = 0.85
    bc = 'newell_freebc_y_maskless'
    boxes = ((256, 256), (350, 350), (350, 500), (500, 500))
    periodic_ref_nm = 206.5
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, '..', '..'))
    d_tag = f'D{dmi_mj:g}'.replace('.', 'p')
    in_dir = os.path.join(
        repo, 'output/stochastic_llgs/validation/relax_torque',
        d_tag, bc)
    stem = f'_{d_tag}_{bc}'
    make_expansion(
        boxes, in_dir, periodic_ref_nm,
        os.path.join(here, 'relax_freebc_y_expansion' + stem))
    print(f'wrote relax_freebc_y_expansion{stem}.{{pdf,png}}')
