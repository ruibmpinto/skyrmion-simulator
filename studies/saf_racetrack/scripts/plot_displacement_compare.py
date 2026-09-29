"""Final-stage displacement: best vs worst pulse shape, side by side.

For the L_x = 1400 nm (700x500) box, compares the worst and a best
shape --- square (least efficient, most charge) and the sharp-fall
triangle (most stable) --- by their final driven configuration. Each
panel shows the final top-layer m_z of the ens0 realization with a
light-green arrow from the skyrmion's start to its end position (an
open circle at the end), so the net displacement is read directly; the
panel title gives that displacement in nm. Square sits on the top row
and the triangle on the bottom, aligned column by column, so the two
shapes can be compared at the same current. One figure per
temperature, on a temperature-specific current ladder (T = 10 K at
J = 2, 4, 6; T = 100 K at the lower J = 1, 2, 3, where the narrower
stability window still holds both shapes).

Any cell beyond a shape's stability cap (no production data) is drawn
as a labelled placeholder.

No argparse; configure via the variables at the top of main().

Run with:
    python -m studies.saf_racetrack.scripts.plot_displacement_compare

Functions
---------
main
    Write the comparison figure for each configured temperature.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import json
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
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


def _decode_config(npz):
    """Decode the per-file config record (ASCII-code JSON) to a dict.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.

    Returns
    -------
    config : dict
        The parsed configuration.
    """
    codes = np.atleast_1d(npz['meta_config_repr']).ravel()
    return json.loads(''.join(chr(int(c)) for c in codes))
# -------------------------------------------------------------------------


def _draw_cell(ax, path):
    """Draw one final-config panel with a start-to-end arrow.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axis.
    path : str
        Trajectory file for the cell.

    Returns
    -------
    d_nm : float
        Net displacement (nm) annotated on the panel.
    """
    npz = np.load(path, allow_pickle=True)
    config = _decode_config(npz)
    nx = int(config['nx'])
    ny = int(config['ny'])
    l_x = float(npz['L_x']) * 1e9
    l_y = float(npz['L_y']) * 1e9
    mz = np.asarray(npz['mz_final_top'], dtype=float).reshape(ny, nx)
    ax.imshow(mz, origin='lower', cmap='RdBu_r', vmin=-1.0, vmax=1.0,
              extent=[0.0, l_x, 0.0, l_y], aspect='equal')
    x0 = float(npz['cx_wrapped'][0]) * 1e9
    y0 = float(npz['cy_wrapped'][0]) * 1e9
    x1 = float(npz['cx_wrapped'][-1]) * 1e9
    y1 = float(npz['cy_wrapped'][-1]) * 1e9
    green = '#66ff66'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=green, lw=2.0))
    ax.plot([x1], [y1], 'o', mfc='none', mec=green, ms=8, mew=2.0)
    cx = npz['cx_unwrapped']
    cy = npz['cy_unwrapped']
    return float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0])) * 1e9
# -------------------------------------------------------------------------


def _plot_temperature(prod_dir, box_tag, t_sub, shapes, currents,
                      out_path):
    """Two-row comparison figure at one temperature.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    box_tag : str
        Box directory name (title only).
    t_sub : float
        Substrate temperature (K).
    shapes : list[str]
        Two shapes, drawn top and bottom.
    currents : list[float]
        Peak currents for the columns (A/m^2).
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        len(shapes), len(currents),
        figsize=(4.2 * len(currents), 3.3 * len(shapes)),
        squeeze=False)
    for r, shape in enumerate(shapes):
        for c, peak_j in enumerate(currents):
            ax = axes[r][c]
            path = os.path.join(
                prod_dir, '%s_T%05.1f_j%.2e_ens000.npz'
                % (shape, t_sub, peak_j))
            if os.path.isfile(path):
                d_nm = _draw_cell(ax, path)
                ax.set_title(r'$J=%.0f$: $d=%.0f$ nm'
                             % (peak_j / 1e11, d_nm))
            else:
                ax.text(0.5, 0.5, 'beyond\nstability cap',
                        ha='center', va='center', fontsize=13,
                        transform=ax.transAxes)
                ax.set_title(r'$J=%.0f$' % (peak_j / 1e11))
                ax.set_xticks([])
                ax.set_yticks([])
            if c == 0:
                ax.set_ylabel('%s\n$y$ (nm)' % shape)
            if r == len(shapes) - 1:
                ax.set_xlabel(r'$x$ (nm)')
    fig.suptitle('%s --- final displacement, $T=%g$ K'
                 % (box_tag, t_sub))
    fig.tight_layout()
    fig.subplots_adjust(top=0.90)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Write the comparison figure for each configured temperature."""
    plt.rcParams.update({
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 18,
    })
    # =========================== User Configuration =========================
    prod_root = ('output/'
                 'sweeps_driving_T/pulse_shape/production')
    box_tag = 'hk36_D0p72_700x500'
    out_dir = 'output/figures_driving_T/ranking'
    shapes = ['square', 'tri_sharpfall']
    # Currents differ by temperature: the stability window narrows with
    # T, so T = 100 K is shown on its own (lower) current ladder.
    currents_by_t = {
        10.0: [2.0e11, 4.0e11, 6.0e11],
        100.0: [1.0e11, 2.0e11, 3.0e11],
    }
    # ======================= End User Configuration =========================
    prod_dir = os.path.join(prod_root, box_tag)
    if not os.path.isdir(prod_dir):
        raise RuntimeError(
            'main: production directory not found: %r.' % prod_dir)
    os.makedirs(out_dir, exist_ok=True)
    for t_sub, currents in currents_by_t.items():
        _plot_temperature(
            prod_dir, box_tag, t_sub, shapes, currents,
            os.path.join(out_dir,
                         'dcompare_T%03d_%s.png' % (int(t_sub), box_tag)))


# =============================================================================
if __name__ == '__main__':
    main()
