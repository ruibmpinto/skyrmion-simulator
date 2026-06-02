"""Plot the Güngördü 2016 / Banerjee 2014 phase-diagram benchmark.

Reads the NPZ written by `run_gungordu_sweep.py`, converts the
physical (K_top, H_z) axes back to Güngördü's dimensionless
Fig. 3 coordinates

    a_s = A_s J / D^2 = (0.5 mu0 Ms^2 - K_top) A_ex / D^2,
    h_g = H   J / D^2 = M_s H_z A_ex / D^2,

and renders the simulated phase map with Güngördü's digitised
Fig. 3 phase boundaries (FM / SkX / SP / SC) overlaid for
direct visual comparison.

Reference boundaries (Güngördü 2016, Fig. 3, pure Rashba)
---------------------------------------------------------
Digitised polylines in (a_s, h_g):
  - FM -> SkX (upper edge of the SkX lens): rises from
    (-1.25, 0) through (0, 0.8) to the apex (1.5, 2.35).
  - SP -> SkX (top of the low-field spiral sliver): ~flat at
    h_g ~ 0.2 from a_s = -1.25 to ~1.0.
  - SC wedge (bottom-right, easy-plane): a_s ~ 1.05-1.6,
    h_g < ~0.5.
  - grey line: aligned/tilted-FM split, ~ h_g = 1.5 a_s.
The simulator's phases are the coloured map; agreement of the
SkX lens, the SP sliver, and the FM region with these curves
is the benchmark.

Note on labels
--------------
The classifier emits {FM_anti, FM_par+/-, iSk, SkX, BX, SS,
Lab, ...}. Mapping to Güngördü: FM<->FM_*, SkX<->SkX, SP<->SS
(spin spiral). Güngördü's SC (square vortex-antivortex
lattice) has no dedicated classifier label (it keys on 6-fold
vs 2-fold order); a 4-fold square state typically lands in
BX/Lab there, so the small bottom-right SC region is only
qualitatively resolved.

Functions
---------
main
    Load the sweep NPZ, plot the phase map in (a_s, h_g) with
    the Fig. 3 boundaries, save the figure.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.phase_diagram.validation.banerjee_reference import (
    BAN_FM_SKX, BAN_SKX_SP, ban_fm_tilt)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Digitised Güngördü 2016 Fig. 3 phase boundaries, (a_s, h_g).
_FM_SKX = [(-1.25, 0.0), (-1.0, 0.20), (-0.5, 0.45),
           (0.0, 0.80), (0.5, 1.20), (1.0, 1.70),
           (1.5, 2.35)]
_SP_SKX = [(-1.25, 0.0), (-1.0, 0.15), (-0.5, 0.20),
           (0.0, 0.22), (0.5, 0.23), (1.0, 0.22)]
_SC_WEDGE = [(1.05, 0.0), (1.15, 0.25), (1.30, 0.45),
             (1.55, 0.30), (1.60, 0.0)]
_FM_TILT = [(0.0, 0.0), (1.6, 2.4)]   # grey aligned/tilted line


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    in_path = 'output/phase_diagram/validation/gungordu_K_H.npz'
    out_path = ('docs/figures/validation/'
                'gungordu_phase_overlay.pdf')
    if not os.path.exists(in_path):
        raise RuntimeError(
            f'plot_gungordu_overlay: input {in_path!r} not '
            f'found. Run run_gungordu_sweep.py first '
            f'(probably on the cluster) and aggregate any '
            f'partial NPZs before plotting.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    z = np.load(in_path, allow_pickle=True)
    K_values = np.asarray(z['axis_x_values'], dtype=float)
    H_values = np.asarray(z['axis_y_values'], dtype=float)
    labels = list(z['labels'])
    # Ground state = MIN-ENERGY candidate per cell, re-derived
    # from the per-IC energies. Güngördü compares the energies
    # of optimized candidate states; the stored `gs_label_idx`
    # additionally requires strict torque convergence, which
    # the slowly-relaxing textured states (SP, SkX, SC) rarely
    # reach in a finite box -- excluding them and collapsing
    # every cell to FM. Their energies are well-settled, so we
    # select on energy directly (matching the reference).
    E = np.asarray(z['E'])
    lpi = np.asarray(z['label_idx_per_ic'], dtype=int)
    Em = np.where(np.isfinite(E), E, np.inf)
    kmin = np.argmin(Em, axis=2)
    gs_label_idx = np.take_along_axis(
        lpi, kmin[:, :, None], axis=2)[:, :, 0]
    A_ex = float(z['A_ex'])
    Ms = float(z['Ms'])
    mu0 = float(z['mu0'])
    if 'D_run' not in z.files:
        raise RuntimeError(
            f'plot_gungordu_overlay: input NPZ {in_path!r} '
            f'does not contain the `D_run` field. Re-run '
            f'run_gungordu_sweep.py (which writes D_run).')
    D = float(z['D_run'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Back-convert physical axes to Güngördü dimensionless
    # coords. Inverse of the runner mapping (demag off, bare K,
    # K = -A_s_phys):
    #   a_s = -K A_ex / D^2
    #   h_g = M_s H_z A_ex / D^2
    a_s_axis = -K_values * A_ex / (D * D)
    h_g_axis = Ms * H_values * A_ex / (D * D)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_x, n_y = gs_label_idx.shape
    a_s_2d, h_g_2d = np.meshgrid(a_s_axis, h_g_axis,
                                 indexing='ij')
    cmap = plt.get_cmap('tab10', len(labels))
    fig, ax = plt.subplots(figsize=(7.5, 6.0), dpi=160)
    pcm = ax.pcolormesh(
        a_s_2d, h_g_2d, gs_label_idx,
        cmap=cmap, vmin=-0.5, vmax=len(labels) - 0.5,
        shading='nearest')
    cbar = fig.colorbar(pcm, ax=ax, ticks=range(len(labels)))
    cbar.ax.set_yticklabels(labels)
    cbar.set_label('phase (simulator)')
    # Güngördü Fig. 3 reference boundaries.
    for poly, lbl, style in [
            (_FM_SKX, 'FM-SkX (Güngördü)', 'k-'),
            (_SP_SKX, 'SP-SkX (Güngördü)', 'k--'),
            (_SC_WEDGE, 'SC wedge (Güngördü)', 'k-.'),
            (_FM_TILT, 'FM aligned/tilted', '-')]:
        xs = [p[0] for p in poly]
        ys = [p[1] for p in poly]
        if style == '-':
            ax.plot(xs, ys, color='0.6', lw=2.0, label=lbl)
        else:
            ax.plot(xs, ys, style, lw=1.8, label=lbl)
    # Banerjee 2014 Fig. 1(b) variational boundaries (same
    # (a_s, h_g) plane). Dotted, distinct colour, so the
    # simulator map is compared to BOTH references at once.
    bx = [p[0] for p in BAN_FM_SKX]
    by = [p[1] for p in BAN_FM_SKX]
    ax.plot(bx, by, ':', color='tab:red', lw=1.8,
            label='FM-SkX (Banerjee)')
    sx = [p[0] for p in BAN_SKX_SP]
    sy = [p[1] for p in BAN_SKX_SP]
    ax.plot(sx, sy, ':', color='tab:blue', lw=1.8,
            label='SkX-SP (Banerjee)')
    a_line = np.linspace(-0.5, 0.75, 20)
    ax.plot(a_line, [ban_fm_tilt(av) for av in a_line],
            ':', color='0.4', lw=1.2,
            label='H=2A FM-tilt (Banerjee)')
    ax.set_xlim(-1.5, 1.7)
    ax.set_ylim(0.0, 2.4)
    ax.axvline(0.0, color='0.8', lw=0.8)
    ax.set_xlabel(r'$a_s = A_s J / D^2$')
    ax.set_ylabel(r'$h_g = H J / D^2$')
    ax.set_title(
        f'Güngördü Fig. 3 phase diagram '
        f'(D = {D*1e3:.2f} mJ/m$^2$, {n_x}x{n_y} cells, '
        f'64x64 lattice, demag off)')
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    fig.savefig(out_path.replace('.pdf', '.png'))
    print(f'Saved {out_path}')


# =============================================================================
if __name__ == '__main__':
    main()
