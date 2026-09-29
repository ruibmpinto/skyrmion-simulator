"""Skyrmion mobility mu(shape, T, J) for the 700x500 box.

The mobility is the velocity per unit current, v = mu J. Since the net
displacement is d = int v dt = mu int J dt, it equals the displacement
per delivered charge,

    mu = d / int J dt,

which is exactly the y-value of the displacement-per-charge ranking
figure. This script evaluates it for the L_x = 1400 nm (700x500) box
from the pulse-shape production, per shape, ensemble-averaged over the
x-localized survivors of each (T, J) cell, and plots mu versus peak
current, one curve per shape, per temperature. Charge is expressed in
10^11 A/m^2 times ns, so mu reads in nm per (10^11 A/m^2 . ns): at low
drive every shape collapses to the intrinsic mobility, which rises
with temperature and falls at high drive as the skyrmion deforms.

No argparse; configure via the variables at the top of main().

Run with:
    python -m studies.saf_racetrack.scripts.plot_mobility

Functions
---------
main
    Load the 700x500 production once and write the mobility figure.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# Local
from studies.saf_racetrack.scripts.plot_pulse_ranking import (
    _load_box, _make_pulse, _shape_colors)
from skyrmion_simulator.simulator.pulse_metrics import analytic_charge

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _mobility_series(cache, shape, t_sub):
    """Mobility curve mu = d / charge for one (shape, T).

    Parameters
    ----------
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    js : list[float]
        Peak currents (1e11 A/m^2).
    mu : list[float]
        Mobility (nm per 1e11 A/m^2 per ns).
    err : list[float]
        SE of the mobility.
    """
    js, mu, err = [], [], []
    for peak_j, stats in cache[shape][t_sub]:
        if stats['n_loc'] == 0:
            continue
        pulse = _make_pulse(shape, peak_j, 0.5e-9, 0.25e-9)
        # Charge in (1e11 A/m^2) . ns, so mu reads in nm per that.
        charge = analytic_charge(pulse) / (1.0e11 * 1.0e-9)
        js.append(peak_j / 1.0e11)
        mu.append(stats['d_loc'] / charge)
        err.append(stats['d_loc_se'] / charge)
    return js, mu, err
# -------------------------------------------------------------------------


def _plot_mobility(box_tag, cache, shapes, temperatures, ylim, out_path):
    """Mobility versus peak current, one curve per shape, per T.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
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
            js, mu, err = _mobility_series(cache, shape, t_sub)
            if js:
                ax.errorbar(js, mu, yerr=err, fmt='o-',
                            color=colors[shape], capsize=2,
                            label=shape)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'peak $J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(
                r'$\mu = d/\!\int J\,dt$'
                r' (nm per $10^{11}$ A/m$^2$/ns)')
        ax.set_box_aspect(1)
    axes[0][0].legend(loc='best', frameon=False)
    fig.suptitle('%s --- skyrmion mobility' % box_tag)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _mobility_vs_t_series(cache, shape, peak_j, temperatures):
    """Mobility versus temperature at fixed current for one shape.

    Parameters
    ----------
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shape : str
        Pulse-shape name.
    peak_j : float
        Peak current density held fixed (A/m^2).
    temperatures : list[float]
        Substrate temperatures to sample (K).

    Returns
    -------
    ts : list[float]
        Temperatures with a surviving cell at this current (K).
    mu : list[float]
        Mobility (nm per 1e11 A/m^2 per ns).
    err : list[float]
        SE of the mobility.
    """
    pulse = _make_pulse(shape, peak_j, 0.5e-9, 0.25e-9)
    charge = analytic_charge(pulse) / (1.0e11 * 1.0e-9)
    ts, mu, err = [], [], []
    for t_sub in temperatures:
        stats = dict(cache[shape][t_sub]).get(peak_j)
        if stats is None or stats['n_loc'] == 0:
            continue
        ts.append(t_sub)
        mu.append(stats['d_loc'] / charge)
        err.append(stats['d_loc_se'] / charge)
    return ts, mu, err
# -------------------------------------------------------------------------


def _plot_mobility_vs_t(box_tag, cache, shapes, temperatures, peak_js,
                        ylim, out_path):
    """Mobility versus temperature, one panel per current.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures for the x-axis (K).
    peak_js : list[float]
        Peak currents, one panel each (A/m^2).
    ylim : tuple[float]
        Shared (low, high) y-limits.
    out_path : str
        Destination PNG.
    """
    colors = _shape_colors(shapes)
    fig, axes = plt.subplots(
        1, len(peak_js), figsize=(5 * len(peak_js), 4.8),
        squeeze=False)
    for col, peak_j in enumerate(peak_js):
        ax = axes[0][col]
        for shape in shapes:
            ts, mu, err = _mobility_vs_t_series(
                cache, shape, peak_j, temperatures)
            if ts:
                ax.errorbar(ts, mu, yerr=err, fmt='o-',
                            color=colors[shape], capsize=2,
                            label=shape)
        ax.set_title(r'$J = %.0f\times10^{11}$ A/m$^2$' % (peak_j / 1e11))
        ax.set_xlabel(r'$T$ (K)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(
                r'$\mu = d/\!\int J\,dt$'
                r' (nm per $10^{11}$ A/m$^2$/ns)')
        ax.set_box_aspect(1)
    axes[0][0].legend(loc='best', frameon=False)
    fig.suptitle('%s --- mobility vs temperature' % box_tag)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _mu_at(cache, shape, t_sub, peak_j):
    """Mobility of one (shape, T, J) cell, or None if absent.

    Parameters
    ----------
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    mu : {float, None}
        Mobility (nm per 1e11 A/m^2 per ns), or None if the cell has
        no surviving members.
    """
    stats = dict(cache[shape][t_sub]).get(peak_j)
    if stats is None or stats['n_loc'] == 0:
        return None
    pulse = _make_pulse(shape, peak_j, 0.5e-9, 0.25e-9)
    charge = analytic_charge(pulse) / (1.0e11 * 1.0e-9)
    return stats['d_loc'] / charge
# -------------------------------------------------------------------------


def _plot_mobility_range(box_tag, cache, shapes, temperatures, peak_js,
                         out_path):
    """Shape-spread mobility band versus temperature, one band per J.

    At each (T, J) the mobility is taken over all shapes that survive;
    the shaded band spans their min to max and the line is their mean.
    A narrow band means the mobility is shape-independent there.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `plot_pulse_ranking._load_box`.
    shapes : list[str]
        Shape names to span.
    temperatures : list[float]
        Substrate temperatures for the x-axis (K).
    peak_js : list[float]
        Peak currents, one band each (A/m^2).
    out_path : str
        Destination PNG.
    """
    cmap = plt.get_cmap('viridis')
    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    for k, peak_j in enumerate(peak_js):
        color = cmap(k / max(len(peak_js) - 1, 1))
        ts, lo, hi, mid = [], [], [], []
        for t_sub in temperatures:
            vals = [_mu_at(cache, shape, t_sub, peak_j)
                    for shape in shapes]
            vals = [v for v in vals if v is not None]
            if not vals:
                continue
            ts.append(t_sub)
            lo.append(min(vals))
            hi.append(max(vals))
            mid.append(float(np.mean(vals)))
        if not ts:
            continue
        ax.fill_between(ts, lo, hi, color=color, alpha=0.25,
                        linewidth=0)
        ax.plot(ts, mid, 'o-', color=color,
                label=r'$J = %.0f\times10^{11}$ A/m$^2$'
                % (peak_j / 1e11))
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$\mu = d/\!\int J\,dt$'
                  r' (nm per $10^{11}$ A/m$^2$/ns)')
    ax.legend(loc='best', frameon=False)
    ax.set_xlim(0.0, 100.0)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the 700x500 production once and write the mobility figure."""
    plt.rcParams.update({
        'axes.labelsize': 19,
        'axes.titlesize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'figure.titlesize': 19,
    })
    # =========================== User Configuration =========================
    prod_root = ('output/'
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
    vs_t_currents = [1.0e11, 2.0e11, 3.0e11]
    print('loading %s ...' % box_tag)
    cache = _load_box(prod_dir, shapes, temperatures)
    mu_vals = []
    for shape in shapes:
        for t_sub in temperatures:
            mu_vals += _mobility_series(cache, shape, t_sub)[1]
    lo, hi = min(mu_vals), max(mu_vals)
    pad = 0.05 * (hi - lo)
    ylim = (max(0.0, lo - pad), hi + pad)
    _plot_mobility(
        box_tag, cache, shapes, temperatures, ylim,
        os.path.join(out_dir, 'mobility_vs_J_%s.png' % box_tag))
    _plot_mobility_vs_t(
        box_tag, cache, shapes, temperatures, vs_t_currents, ylim,
        os.path.join(out_dir, 'mobility_vs_T_%s.png' % box_tag))
    _plot_mobility_range(
        box_tag, cache, shapes, temperatures,
        [1.0e11, 3.0e11, 5.0e11, 8.0e11],
        os.path.join(out_dir, 'mobility_range_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
