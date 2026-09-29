"""Energy-efficient and fastest operating points per temperature.

From the pulse-shape production of the L_x = 1400 nm (700x500) box,
finds for each temperature the two extremal operating points over all
surviving (shape, peak current) cells:

- energy-efficient: the cell maximizing displacement per delivered
  action d / int J^2 dt (the least energy dissipated per unit travel);
- fastest: the cell maximizing the average pulse speed.

Emits a printed and a LaTeX table of the two points per temperature,
and a scatter of speed versus efficiency (every surviving cell, one
colour per temperature) with the two operating points of each
temperature marked, which shows the speed/efficiency trade-off.

No argparse; configure via the variables at the top of main().

Run with:
    python -m scripts.plot_operating_points

Functions
---------
main
    Load the box once, find the operating points, write table + figure.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
# Local
from scripts.plot_pulse_ranking import _load_box, _make_pulse
from src.simulator.pulse_metrics import analytic_action

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _cells(cache, shapes, t_sub):
    """Surviving cells of one temperature with efficiency and speed.

    Parameters
    ----------
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shapes : list[str]
        Shape names.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    rows : list[dict]
        One dict per surviving cell: shape, peak_j (A/m^2), eff
        (d / action), speed (average pulse speed, m/s).
    """
    rows = []
    for shape in shapes:
        for peak_j, stats in cache[shape][t_sub]:
            if stats['n_loc'] == 0:
                continue
            pulse = _make_pulse(shape, peak_j, 0.5e-9, 0.25e-9)
            eff = stats['d_loc'] / analytic_action(pulse)
            rows.append({'shape': shape, 'peak_j': peak_j,
                         'eff': eff, 'speed': stats['vavg']})
    return rows
# -------------------------------------------------------------------------


def _operating_points(cache, shapes, temperatures):
    """Best-efficiency and best-speed cell per temperature.

    Parameters
    ----------
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    points : list[dict]
        Per temperature: t_sub, 'best_eff' cell, 'best_speed' cell,
        and 'rows' (all surviving cells).
    """
    points = []
    for t_sub in temperatures:
        rows = _cells(cache, shapes, t_sub)
        if not rows:
            continue
        best_eff = max(rows, key=lambda r: r['eff'])
        best_speed = max(rows, key=lambda r: r['speed'])
        points.append({'t_sub': t_sub, 'best_eff': best_eff,
                       'best_speed': best_speed, 'rows': rows})
    return points
# -------------------------------------------------------------------------


def _print_tables(points):
    """Print the operating points as text and as a LaTeX table.

    Parameters
    ----------
    points : list[dict]
        Output of `_operating_points`.
    """
    print('\n=== operating points per temperature ===')
    print('%-6s | %-28s | %-28s'
          % ('T (K)', 'energy-efficient', 'fastest'))
    print('%-6s | %-14s %-13s | %-14s %-13s'
          % ('', 'shape (J)', 'd/action', 'shape (J)', 'v_avg'))
    for p in points:
        e, s = p['best_eff'], p['best_speed']
        print('%-6g | %-10s %4.1f  %.3e | %-10s %4.1f  %6.1f m/s'
              % (p['t_sub'], e['shape'], e['peak_j'] / 1e11, e['eff'],
                 s['shape'], s['peak_j'] / 1e11, s['speed']))
    print('\n=== LaTeX ===')
    print(r'\begin{tabular}{rllll}')
    print(r'\toprule')
    print(r'$T$ (K) & \multicolumn{2}{c}{energy-efficient} '
          r'& \multicolumn{2}{c}{fastest}\\')
    print(r'\cmidrule(lr){2-3}\cmidrule(lr){4-5}')
    print(r' & shape ($J$) & $v_{\mathrm{avg}}$ '
          r'& shape ($J$) & $v_{\mathrm{avg}}$\\')
    print(r'\midrule')
    for p in points:
        e, s = p['best_eff'], p['best_speed']
        print(r'%g & %s ($%.1f$) & $%.0f$ & %s ($%.1f$) & $%.0f$\\'
              % (p['t_sub'],
                 e['shape'].replace('_', r'\_'), e['peak_j'] / 1e11,
                 e['speed'],
                 s['shape'].replace('_', r'\_'), s['peak_j'] / 1e11,
                 s['speed']))
    print(r'\bottomrule')
    print(r'\end{tabular}')
# -------------------------------------------------------------------------


def _plot_points(box_tag, points, temperatures, out_path):
    """Scatter of speed versus efficiency with the operating points.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    points : list[dict]
        Output of `_operating_points`.
    temperatures : list[float]
        Substrate temperatures (K), for the colour scale.
    out_path : str
        Destination PNG.
    """
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(8.0, 6.0))
    t_min, t_max = min(temperatures), max(temperatures)

    def _color(t_sub):
        return cmap((t_sub - t_min) / max(t_max - t_min, 1.0))

    for p in points:
        color = _color(p['t_sub'])
        sp = [r['speed'] for r in p['rows']]
        ef = [r['eff'] for r in p['rows']]
        ax.scatter(sp, ef, s=18, color=color, alpha=0.5,
                   label=r'$T=%g$ K' % p['t_sub'])
        e, s = p['best_eff'], p['best_speed']
        ax.scatter([e['speed']], [e['eff']], s=190, marker='*',
                   color=color, edgecolor='k', zorder=5)
        ax.scatter([s['speed']], [s['eff']], s=110, marker='D',
                   color=color, edgecolor='k', zorder=5)
    ax.set_xlabel(r'average pulse speed $v_{\mathrm{avg}}$ (m/s)')
    ax.set_ylabel(r'energy efficiency $d/\!\int J^2\,dt$')
    star = mlines.Line2D([], [], color='0.4', marker='*', ls='none',
                         markersize=13, markeredgecolor='k',
                         label='most efficient')
    diam = mlines.Line2D([], [], color='0.4', marker='D', ls='none',
                         markersize=9, markeredgecolor='k',
                         label='fastest')
    leg1 = ax.legend(loc='upper right', frameon=False)
    ax.add_artist(leg1)
    ax.legend(handles=[star, diam], loc='lower left', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the box once, find the operating points, write outputs."""
    plt.rcParams.update({
        'axes.labelsize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 12,
    })
    # =========================== User Configuration =========================
    prod_root = ('/Volumes/T7/skyrmion_simulator/output/'
                 'sweeps_driving_T/pulse_shape/production')
    box_tag = 'hk36_D0p72_700x500'
    out_dir = 'output/figures_driving_T/ranking'
    shapes = ['square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
              'tri_symmetric', 'gaussian']
    temperatures = [0.0, 10.0, 50.0, 100.0]
    # ======================= End User Configuration =========================
    prod_dir = os.path.join(prod_root, box_tag)
    if not os.path.isdir(prod_dir):
        raise RuntimeError(
            'main: production directory not found: %r.' % prod_dir)
    os.makedirs(out_dir, exist_ok=True)
    print('loading %s ...' % box_tag)
    cache = _load_box(prod_dir, shapes, temperatures)
    points = _operating_points(cache, shapes, temperatures)
    _print_tables(points)
    _plot_points(
        box_tag, points, temperatures,
        os.path.join(out_dir, 'operating_points_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
