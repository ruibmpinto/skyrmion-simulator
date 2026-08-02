"""Pulse-width analysis of the 700x500 width sweep.

For each shape, at that shape's cap current (the strongest usable
drive) for each temperature, tracks how the driven response depends on
pulse duration t_pulse in {100, 200, 300, 400, 600, 700} ps (the 500 ps
point is the production pulse and is not repeated here). Every quantity
is an ensemble average over the x-localized survivors of the cell.

Figures (output/figures_driving_T/width/):

width_displacement_<box>.png   net displacement vs t_pulse
width_speed_<box>.png          average pulse speed vs t_pulse
width_deformation_<box>.png    max ellipse major axis D1 vs t_pulse
width_survival_<box>.png       surviving-skyrmion fraction vs t_pulse

each one curve per shape, one panel per temperature, on a shared
y-range.

No argparse; configure via the variables at the top of main().

Run with:
    python -m scripts.plot_pulse_width

Functions
---------
main
    Load the width sweep and write the four width-dependence figures.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# Local
from scripts.plot_pulse_ranking import (_classify, _decode_config,
                                        _displacement, _pulse_speed)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _cap_current(width_dir, shape, t_sub):
    """Largest peak current present for one (shape, T), the cap.

    Parameters
    ----------
    width_dir : str
        Width-sweep directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    peak_j : {float, None}
        The cap current (A/m^2), or None if the shape is absent.
    """
    js = set()
    for path in glob.glob(os.path.join(
            width_dir, '%s_tp*_T%05.1f_j*_ens000.npz' % (shape, t_sub))):
        tag = os.path.basename(path).split('_j')[1].split('_ens')[0]
        js.add(float(tag))
    return max(js) if js else None
# -------------------------------------------------------------------------


def _width_cell(width_dir, shape, t_sub, peak_j, width_ps):
    """Ensemble statistics of one (shape, T, J, width) cell.

    Parameters
    ----------
    width_dir : str
        Width-sweep directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).
    width_ps : int
        Pulse duration (ps).

    Returns
    -------
    stats : {dict, None}
        Mean and SE of displacement (nm), average speed (m/s), max D1
        (nm), min D2 (nm) over x-localized survivors; plus P_surv and
        n_total. None if the cell is absent.
    """
    paths = sorted(glob.glob(os.path.join(
        width_dir, '%s_tp%04d_T%05.1f_j%.2e_ens*.npz'
        % (shape, width_ps, t_sub, peak_j))))
    if not paths:
        return None
    d_nm, v_avg, d1_max, d2_min = [], [], [], []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        code, metrics = _classify(npz, config)
        if not (code in ('S', 'E')
                or (code == 'P' and metrics['dx_lx'] < 0.5)):
            continue
        d_nm.append(_displacement(npz) * 1e9)
        _vmax, va = _pulse_speed(npz, float(config['t_pulse']))
        v_avg.append(va)
        d1_max.append(float(np.max(npz['D1_top'])) * 1e9)
        d2_min.append(float(np.min(npz['D2_top'])) * 1e9)
    n_surv = len(d_nm)

    def _ms(vals):
        return ((float(np.mean(vals)),
                 float(np.std(vals) / np.sqrt(len(vals))))
                if vals else (float('nan'), float('nan')))

    d_m, d_se = _ms(d_nm)
    v_m, v_se = _ms(v_avg)
    d1_m, d1_se = _ms(d1_max)
    d2_m, d2_se = _ms(d2_min)
    return {
        'd': d_m, 'd_se': d_se, 'v': v_m, 'v_se': v_se,
        'd1': d1_m, 'd1_se': d1_se, 'd2': d2_m, 'd2_se': d2_se,
        'p_surv': n_surv / len(paths), 'n_total': len(paths),
    }
# -------------------------------------------------------------------------


def _load(width_dir, shapes, temperatures, widths):
    """Cache of per-(shape, T, width) statistics at each shape's cap.

    Parameters
    ----------
    width_dir : str
        Width-sweep directory for the box.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K).
    widths : list[int]
        Pulse durations (ps).

    Returns
    -------
    cache : dict
        cache[shape][t_sub] = list of (width_ps, stats) with stats
        present, ascending in width.
    """
    cache = {}
    for shape in shapes:
        cache[shape] = {}
        for t_sub in temperatures:
            peak_j = _cap_current(width_dir, shape, t_sub)
            row = []
            if peak_j is not None:
                for width_ps in widths:
                    stats = _width_cell(
                        width_dir, shape, t_sub, peak_j, width_ps)
                    if stats is not None:
                        row.append((width_ps, stats))
            cache[shape][t_sub] = row
    return cache
# -------------------------------------------------------------------------


def _shape_colors(shapes):
    """Fixed colour per shape.

    Parameters
    ----------
    shapes : list[str]
        Shape names.

    Returns
    -------
    colors : dict
        Shape -> RGBA.
    """
    cmap = plt.get_cmap('tab10')
    return {s: cmap(i) for i, s in enumerate(shapes)}
# -------------------------------------------------------------------------


def _plot_metric(box_tag, cache, shapes, temperatures, key, se_key,
                 ylabel, ylim, out_path):
    """One metric versus pulse duration, per shape, per temperature.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `_load`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    key : str
        Stats key to plot ('d', 'v', 'd1', 'p_surv').
    se_key : {str, None}
        Matching SE key, or None for no error bar.
    ylabel : str
        Y-axis label.
    ylim : tuple[float]
        Shared (low, high) y-limits.
    out_path : str
        Destination PNG.
    """
    colors = _shape_colors(shapes)
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.8),
        squeeze=False)
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        for shape in shapes:
            row = cache[shape][t_sub]
            if not row:
                continue
            ws = [w for w, _ in row]
            ys = [s[key] for _, s in row]
            es = ([s[se_key] for _, s in row] if se_key
                  else [0.0] * len(ws))
            ax.errorbar(ws, ys, yerr=es, fmt='o-', color=colors[shape],
                        capsize=2, label=shape)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'pulse duration $t_{\mathrm{pulse}}$ (ps)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
    axes[0][0].legend(loc='best', frameon=False, fontsize=9)
    fig.suptitle('%s -- %s (at each shape cap current)'
                 % (box_tag, ylabel))
    fig.tight_layout()
    fig.subplots_adjust(top=0.85)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the width sweep and write the four width figures."""
    plt.rcParams.update({
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'figure.titlesize': 17,
    })
    # =========================== User Configuration =========================
    width_root = ('/Volumes/T7/skyrmion_simulator/output/'
                  'sweeps_driving_T/pulse_shape/width')
    box_tag = 'hk36_D0p72_700x500'
    out_dir = 'output/figures_driving_T/width'
    shapes = ['square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
              'tri_symmetric', 'gaussian']
    temperatures = [0.0, 10.0, 50.0, 100.0]
    widths = [100, 200, 300, 400, 600, 700]
    # ======================= End User Configuration =========================
    width_dir = os.path.join(width_root, box_tag)
    if not os.path.isdir(width_dir):
        raise RuntimeError(
            'main: width directory not found: %r.' % width_dir)
    os.makedirs(out_dir, exist_ok=True)
    print('loading %s ...' % box_tag)
    cache = _load(width_dir, shapes, temperatures, widths)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared y-limits per metric across every panel.
    def _span(key, floor_zero):
        vals = [s[key] for sh in shapes for t in temperatures
                for _, s in cache[sh][t] if np.isfinite(s[key])]
        lo, hi = min(vals), max(vals)
        pad = 0.05 * (hi - lo)
        return (0.0 if floor_zero else lo - pad, hi + pad)

    _plot_metric(
        box_tag, cache, shapes, temperatures, 'd', 'd_se',
        r'displacement (nm)', _span('d', True),
        os.path.join(out_dir, 'width_displacement_%s.png' % box_tag))
    _plot_metric(
        box_tag, cache, shapes, temperatures, 'v', 'v_se',
        r'average speed (m/s)', _span('v', True),
        os.path.join(out_dir, 'width_speed_%s.png' % box_tag))
    _plot_metric(
        box_tag, cache, shapes, temperatures, 'd1', 'd1_se',
        r'max $D_1$ (nm)', _span('d1', True),
        os.path.join(out_dir, 'width_deformation_%s.png' % box_tag))
    _plot_metric(
        box_tag, cache, shapes, temperatures, 'p_surv', None,
        r'$P_{\mathrm{surv}}$', (0.0, 1.05),
        os.path.join(out_dir, 'width_survival_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
