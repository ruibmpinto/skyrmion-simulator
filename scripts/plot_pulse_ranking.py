"""Pulse-shape ranking by the three driving figures of merit.

Ranks the six drive shapes (square, half-sine, three triangle
asymmetries, Gaussian) on the finite-T pulse-shape production data
(500 ps pulse + 500 ps settle, racetrack, D = 0.72 mJ/m^2, Set A) by:

1. displacement per delivered charge   d / int J dt,
2. displacement per delivered action   d / int J^2 dt   (Ohmic
   dissipation, i.e. energy),
3. maximum displacement while the skyrmion stays compact.

The displacement d is the net centroid travel over the trajectory,
ensemble-averaged over surviving realizations of each cell; the charge
and action are the analytic integrals of the shape at its peak
amplitude, so they are exact and shape-specific. The three triangles
share both integrals, so their ranking isolates rise/fall asymmetry.

Figures (output/figures_driving_T/ranking/), one per box:

rank_d_per_charge_<box>.png, rank_d_per_action_<box>.png
    Efficiency versus peak current, one curve per shape, per T.
rank_maxd_compact_<box>.png
    Maximum displacement reached while compact (S), grouped bar over
    shape and temperature.
speed_max_<box>.png, speed_avg_<box>.png
    Peak and average pulse speed versus peak current, one curve per
    shape, per T.

All three boxes share one y-axis range per figure type, so panels are
comparable across temperature and across track length. Members are
gated to single-winding, x-localized skyrmions (S, E, and seam-sitting
P with D_x/L_x < 0.5) for the efficiency figures, and to compact (S)
members for the maximum-compact-displacement figure. The short box
carries no periodic wrap-around: the largest displacement (~218 nm) is
far below its L_x = 700 nm, so the net centroid travel is unambiguous.

No argparse; configure via the variables at the top of main().

Run with:
    python -m scripts.plot_pulse_ranking

Functions
---------
main
    Load all boxes once and write the three ranking figures per box.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import json
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.simulator.pulse_metrics import analytic_action, analytic_charge
from src.simulator.pulses import (GaussianPulse, HalfSinePulse,
                                   SquarePulse, TrianglePulse)
from src.stochastic_llgs.stability import classify_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
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
    if 'meta_config_repr' not in npz:
        raise RuntimeError(
            '_decode_config: trajectory has no meta_config_repr.')
    codes = np.atleast_1d(npz['meta_config_repr']).ravel()
    return json.loads(''.join(chr(int(c)) for c in codes))
# -------------------------------------------------------------------------


def _make_pulse(shape, j0, t_pulse, gauss_fwhm):
    """Reconstruct a shape's pulse object at peak amplitude.

    Parameters
    ----------
    shape : str
        Pulse-shape name.
    j0 : float
        Peak current density (A/m^2).
    t_pulse : float
        Pulse duration (s).
    gauss_fwhm : float
        Gaussian FWHM (s), used only for the Gaussian shape.

    Returns
    -------
    pulse : object
        The pulse primitive for charge/action integrals.
    """
    if shape == 'square':
        return SquarePulse(j0, 0.0, t_pulse)
    if shape == 'halfsine':
        return HalfSinePulse(j0, 0.0, t_pulse)
    if shape == 'tri_sharprise':
        return TrianglePulse(j0, 0.0, 0.0, t_pulse)
    if shape == 'tri_sharpfall':
        return TrianglePulse(j0, 0.0, t_pulse, t_pulse)
    if shape == 'tri_symmetric':
        return TrianglePulse(j0, 0.0, t_pulse / 2.0, t_pulse)
    if shape == 'gaussian':
        return GaussianPulse(j0, t_pulse / 2.0, gauss_fwhm)
    raise RuntimeError('_make_pulse: unknown shape %r.' % shape)
# -------------------------------------------------------------------------


def _classify(npz, config):
    """Class code and metrics of a member's final frame.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.
    config : dict
        Its parsed configuration (for nx, ny).

    Returns
    -------
    code : str
        Classifier verdict.
    metrics : dict
        Classifier metrics.
    """
    nx = int(config['nx'])
    ny = int(config['ny'])
    mz = np.asarray(npz['mz_final_top'], dtype=float).reshape(ny, nx)
    return classify_field(
        mz, abs(float(npz['Q'][-1])), float(npz['D1_top'][-1]),
        float(npz['D2_top'][-1]), float(npz['L_x']))
# -------------------------------------------------------------------------


def _displacement(npz):
    """Net centroid travel of one trajectory (m).

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.

    Returns
    -------
    d : float
        |r(end) - r(0)| from the unwrapped top-layer centroid.
    """
    cx = npz['cx_unwrapped']
    cy = npz['cy_unwrapped']
    return float(np.hypot(cx[-1] - cx[0], cy[-1] - cy[0]))
# -------------------------------------------------------------------------


def _pulse_speed(npz, t_pulse):
    """Maximum and average speed over the drive window (m/s).

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.
    t_pulse : float
        Pulse duration (s); the window is [0, t_pulse].

    Returns
    -------
    v_max : float
        Peak instantaneous speed during the pulse.
    v_avg : float
        Mean instantaneous speed during the pulse.
    """
    t = npz['t_sample']
    v = np.hypot(np.gradient(npz['cx_unwrapped'], t),
                 np.gradient(npz['cy_unwrapped'], t))
    window = t <= t_pulse
    return float(v[window].max()), float(v[window].mean())
# -------------------------------------------------------------------------


def _cell_stats(prod_dir, shape, t_sub, peak_j):
    """Displacement statistics of one (shape, T, J) cell.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    stats : dict
        'd_loc' / 'd_loc_se' (mean and SE of displacement over
        x-localized survivors, nm), 'n_loc'; 'd_cmp' (mean over
        compact members, nm), 'n_cmp', 'frac_cmp'; 'n_total'. NaN
        means no qualifying members.
    """
    paths = sorted(glob.glob(os.path.join(
        prod_dir, '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))))
    d_loc, d_cmp, vmax_loc, vavg_loc = [], [], [], []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        code, metrics = _classify(npz, config)
        localized = code in ('S', 'E') or (
            code == 'P' and metrics['dx_lx'] < 0.5)
        d_nm = _displacement(npz) * 1e9
        if localized:
            d_loc.append(d_nm)
            v_max, v_avg = _pulse_speed(npz, float(config['t_pulse']))
            vmax_loc.append(v_max)
            vavg_loc.append(v_avg)
        if code == 'S':
            d_cmp.append(d_nm)
    n_total = len(paths)

    def _ms(vals):
        return ((float(np.mean(vals)),
                 float(np.std(vals) / np.sqrt(len(vals))))
                if vals else (float('nan'), float('nan')))

    d_loc_m, d_loc_se = _ms(d_loc)
    vmax_m, vmax_se = _ms(vmax_loc)
    vavg_m, vavg_se = _ms(vavg_loc)
    return {
        'd_loc': d_loc_m, 'd_loc_se': d_loc_se, 'n_loc': len(d_loc),
        'vmax': vmax_m, 'vmax_se': vmax_se,
        'vavg': vavg_m, 'vavg_se': vavg_se,
        'd_cmp': float(np.mean(d_cmp)) if d_cmp else float('nan'),
        'n_cmp': len(d_cmp),
        'frac_cmp': len(d_cmp) / n_total if n_total else 0.0,
        'n_total': n_total,
    }
# -------------------------------------------------------------------------


def _currents(prod_dir, shape, t_sub):
    """Ascending peak currents present for one (shape, T).

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    currents : list[float]
        Peak current densities (A/m^2), sorted ascending.
    """
    pattern = os.path.join(
        prod_dir, '%s_T%05.1f_j*_ens000.npz' % (shape, t_sub))
    currents = set()
    for path in glob.glob(pattern):
        tag = os.path.basename(path).split('_j')[1].split('_ens')[0]
        currents.add(float(tag))
    return sorted(currents)
# -------------------------------------------------------------------------


def _load_box(prod_dir, shapes, temperatures):
    """Load every cell of one box into a stats cache.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    cache : dict
        cache[shape][t_sub] = list of (peak_j, stats), ascending J.
    """
    cache = {}
    for shape in shapes:
        cache[shape] = {}
        for t_sub in temperatures:
            row = []
            for peak_j in _currents(prod_dir, shape, t_sub):
                row.append(
                    (peak_j, _cell_stats(prod_dir, shape, t_sub, peak_j)))
            cache[shape][t_sub] = row
    return cache
# -------------------------------------------------------------------------


def _efficiency_series(cache, shape, t_sub, kind):
    """Efficiency curve d / (charge or action) for one (shape, T).

    Parameters
    ----------
    cache : dict
        Output of `_load_box`.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    kind : {'charge', 'action'}
        Normalisation.

    Returns
    -------
    js : list[float]
        Peak currents (1e11 A/m^2).
    eff : list[float]
        d / normalisation.
    err : list[float]
        SE of the above.
    """
    js, eff, err = [], [], []
    for peak_j, stats in cache[shape][t_sub]:
        if stats['n_loc'] == 0:
            continue
        pulse = _make_pulse(shape, peak_j, 0.5e-9, 0.25e-9)
        norm = (analytic_charge(pulse) if kind == 'charge'
                else analytic_action(pulse))
        js.append(peak_j / 1e11)
        eff.append(stats['d_loc'] / norm)
        err.append(stats['d_loc_se'] / norm)
    return js, eff, err
# -------------------------------------------------------------------------


def _velocity_series(cache, shape, t_sub, which):
    """Speed curve versus peak current for one (shape, T).

    Parameters
    ----------
    cache : dict
        Output of `_load_box`.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    which : {'vmax', 'vavg'}
        Peak or average pulse speed.

    Returns
    -------
    js : list[float]
        Peak currents (1e11 A/m^2).
    vals : list[float]
        Speed (m/s).
    err : list[float]
        SE of the above.
    """
    js, vals, err = [], [], []
    for peak_j, stats in cache[shape][t_sub]:
        if stats['n_loc'] == 0:
            continue
        js.append(peak_j / 1e11)
        vals.append(stats[which])
        err.append(stats[which + '_se'])
    return js, vals, err
# -------------------------------------------------------------------------


def _max_compact(cache, shape, t_sub, frac_min):
    """Largest compact-member displacement over qualifying cells.

    Parameters
    ----------
    cache : dict
        Output of `_load_box`.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    frac_min : float
        Minimum compact fraction for a cell to qualify.

    Returns
    -------
    d : float
        Maximum mean compact displacement (nm), or 0 if none qualify.
    """
    best = 0.0
    for _peak_j, stats in cache[shape][t_sub]:
        if stats['frac_cmp'] >= frac_min and np.isfinite(stats['d_cmp']):
            best = max(best, stats['d_cmp'])
    return best
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


def _plot_efficiency(box_tag, cache, shapes, temperatures, kind, ylim,
                     out_path):
    """d / charge or d / action versus peak current, per shape and T.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `_load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    kind : {'charge', 'action'}
        Normalisation shown.
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
            js, eff, err = _efficiency_series(cache, shape, t_sub, kind)
            if js:
                ax.errorbar(js, eff, yerr=err, fmt='o-',
                            color=colors[shape], capsize=2,
                            label=shape)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'peak $J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylim(ylim)
        if col == 0:
            unit = (r'$\int J\,dt$' if kind == 'charge'
                    else r'$\int J^2 dt$')
            ax.set_ylabel(r'$d\,/\,$%s (nm per unit)' % unit)
        ax.set_box_aspect(1)
    # Legend on the T = 0 column, which is populated for every box
    # (the short box has no T = 100 survivors, so its last panel is
    # empty and cannot host the legend).
    axes[0][0].legend(loc='best', frameon=False)
    fig.suptitle('%s --- displacement per %s' % (box_tag, kind))
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_velocity(box_tag, cache, shapes, temperatures, which, ylabel,
                   ylim, out_path):
    """Peak or average pulse speed versus peak current, per shape and T.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `_load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    which : {'vmax', 'vavg'}
        Peak or average pulse speed.
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
            js, vals, err = _velocity_series(cache, shape, t_sub, which)
            if js:
                ax.errorbar(js, vals, yerr=err, fmt='o-',
                            color=colors[shape], capsize=2,
                            label=shape)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'peak $J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(ylabel)
        ax.set_box_aspect(1)
    axes[0][0].legend(loc='best', frameon=False)
    fig.suptitle('%s --- %s' % (box_tag, ylabel))
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_max_compact(box_tag, cache, shapes, temperatures, frac_min,
                      ylim, out_path):
    """Maximum displacement while compact, grouped bar over shape, T.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `_load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the groups (K).
    frac_min : float
        Minimum compact fraction for a cell to qualify.
    ylim : tuple[float]
        Shared (low, high) y-limits.
    out_path : str
        Destination PNG.
    """
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(9, 4.0))
    n_t = len(temperatures)
    width = 0.8 / n_t
    x = np.arange(len(shapes))
    for ti, t_sub in enumerate(temperatures):
        heights = [_max_compact(cache, shape, t_sub, frac_min)
                   for shape in shapes]
        ax.bar(x + (ti - (n_t - 1) / 2.0) * width, heights, width,
               color=cmap(ti / max(n_t - 1, 1)),
               label=r'$T = %g$ K' % t_sub)
    ax.set_xticks(x)
    ax.set_xticklabels(shapes, rotation=20, ha='right')
    ax.set_ylabel(r'max compact displacement (nm)')
    ax.set_ylim(ylim)
    ax.legend(loc='best', frameon=False, ncol=len(temperatures))
    fig.suptitle('%s --- reach while compact' % box_tag)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load all boxes once and write the three ranking figures per box."""
    plt.rcParams.update({
        'axes.labelsize': 16,
        'axes.titlesize': 16,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 19,
    })
    # =========================== User Configuration =========================
    prod_root = ('/Volumes/T7/skyrmion_simulator/output/'
                 'sweeps_driving_T/pulse_shape/production')
    box_tags = ['hk36_D0p72_350x500', 'hk36_D0p72_700x500',
                'hk36_D0p72_1400x500']
    out_dir = 'output/figures_driving_T/ranking'
    shapes = ['square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
              'tri_symmetric', 'gaussian']
    temperatures = [0.0, 10.0, 50.0, 100.0]
    frac_min = 0.5             # compact-fraction floor for FoM 3
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # One load pass per box; every figure is drawn from the cache.
    caches = {}
    for box_tag in box_tags:
        prod_dir = os.path.join(prod_root, box_tag)
        if not os.path.isdir(prod_dir):
            raise RuntimeError(
                'main: production directory not found: %r.' % prod_dir)
        print('loading %s ...' % box_tag)
        caches[box_tag] = _load_box(prod_dir, shapes, temperatures)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared y-limits per figure type, across every box, T and shape.
    charge_vals, action_vals, compact_vals = [], [], []
    vmax_vals, vavg_vals = [], []
    for box_tag in box_tags:
        cache = caches[box_tag]
        for shape in shapes:
            for t_sub in temperatures:
                charge_vals += _efficiency_series(
                    cache, shape, t_sub, 'charge')[1]
                action_vals += _efficiency_series(
                    cache, shape, t_sub, 'action')[1]
                vmax_vals += _velocity_series(
                    cache, shape, t_sub, 'vmax')[1]
                vavg_vals += _velocity_series(
                    cache, shape, t_sub, 'vavg')[1]
                compact_vals.append(
                    _max_compact(cache, shape, t_sub, frac_min))
    def _span(vals):
        lo, hi = min(vals), max(vals)
        pad = 0.05 * (hi - lo)
        return (max(0.0, lo - pad), hi + pad)

    ylim_charge = _span(charge_vals)
    ylim_action = _span(action_vals)
    ylim_compact = (0.0, max(compact_vals) * 1.10)
    # One shared speed scale, so the peak and average panels compare.
    ylim_speed = (0.0, max(vmax_vals + vavg_vals) * 1.05)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for box_tag in box_tags:
        cache = caches[box_tag]
        _plot_efficiency(
            box_tag, cache, shapes, temperatures, 'charge', ylim_charge,
            os.path.join(out_dir, 'rank_d_per_charge_%s.png' % box_tag))
        _plot_efficiency(
            box_tag, cache, shapes, temperatures, 'action', ylim_action,
            os.path.join(out_dir, 'rank_d_per_action_%s.png' % box_tag))
        _plot_velocity(
            box_tag, cache, shapes, temperatures, 'vmax',
            r'max pulse speed (m/s)', ylim_speed,
            os.path.join(out_dir, 'speed_max_%s.png' % box_tag))
        _plot_velocity(
            box_tag, cache, shapes, temperatures, 'vavg',
            r'average pulse speed (m/s)', ylim_speed,
            os.path.join(out_dir, 'speed_avg_%s.png' % box_tag))
        _plot_max_compact(
            box_tag, cache, shapes, temperatures, frac_min, ylim_compact,
            os.path.join(out_dir, 'rank_maxd_compact_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
