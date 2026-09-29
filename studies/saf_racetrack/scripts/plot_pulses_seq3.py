"""Post-processing of the three-pulse-train driving study (T3).

Drives the campaign seeds with a train of three identical pulses back to
back (period = t_pulse, a 4*t_pulse window) on the full production grid,
and characterises the response. Mirrors the single-pulse analysis
(efficiency, speed, deformation, config table) and adds the metrics that
only a pulse TRAIN exposes: the per-pulse displacement increments, the
velocity waveform across the three pulses, and the deformation that
accumulates pulse by pulse.

Charge and action are the single-pulse analytic values times the pulse
count, since the three pulses are identical. The drive speed is measured
over the whole train (t <= n_pulses * t_pulse), not one pulse.

Figures and GIFs (output/figures_driving_T/pulses_seq3/):

seq3_rank_d_per_charge_<box>.png   displacement per delivered charge
seq3_rank_d_per_action_<box>.png   displacement per Ohmic action
seq3_speed_max_<box>.png           peak drive speed vs current
seq3_speed_avg_<box>.png           mean drive speed vs current
seq3_deformation_<box>.png         max major axis D1 vs current
seq3_config_<shape>_<box>.png      ens0 final m_z on the T x J grid
seq3_perpulse_disp_<box>.png       displacement after pulse 1, 2, 3
seq3_vt_train_<box>.png            ensemble-mean |v|(t) over the train
seq3_d1t_train_<box>.png           ensemble-mean D1(t) over the train
seq3_<cell>.gif                    field animations of selected cells

No argparse; configure via the variables at the top of main().

Run with:
    python -m studies.saf_racetrack.scripts.plot_pulses_seq3

Functions
---------
main
    Load the train dataset and write the figures and GIFs.
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
import matplotlib.animation as manim
import matplotlib.pyplot as plt
import numpy as np
# Local
from studies.saf_racetrack.scripts.plot_pulse_ranking import (
    _classify, _decode_config, _displacement, _make_pulse, _pulse_speed)
from skyrmion_simulator.simulator.pulse_metrics import analytic_action, \
    analytic_charge

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _localized(code, metrics):
    """True if the final frame is an x-localized moving skyrmion.

    Parameters
    ----------
    code : str
        Classifier verdict.
    metrics : dict
        Classifier metrics (uses 'dx_lx').

    Returns
    -------
    keep : bool
        Compact or elongated, or a periodic-seam skyrmion with
        D_x / L_x < 0.5; a genuine spanning stripe is rejected.
    """
    return (code in ('S', 'E')
            or (code == 'P' and metrics['dx_lx'] < 0.5))
# -------------------------------------------------------------------------


def _currents(root, shape, t_sub):
    """Peak currents present for one (shape, T), ascending.

    Parameters
    ----------
    root : str
        Dataset directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    js : list[float]
        Peak current densities (A/m^2), sorted ascending.
    """
    js = set()
    for path in glob.glob(os.path.join(
            root, '%s_T%05.1f_j*_ens000.npz' % (shape, t_sub))):
        tag = os.path.basename(path).split('_j')[1].split('_ens')[0]
        js.add(float(tag))
    return sorted(js)
# -------------------------------------------------------------------------


def _pulse_charge_action(shape, peak_j, config):
    """Delivered charge and action of the whole train.

    The three pulses are identical, so the train totals are the
    single-pulse analytic values scaled by the pulse count.

    Parameters
    ----------
    shape : str
        Pulse-shape name.
    peak_j : float
        Peak current density (A/m^2).
    config : dict
        Parsed per-file configuration (t_pulse, gauss_fwhm, n_pulses).

    Returns
    -------
    charge : float
        Total delivered charge (A s / m^2).
    action : float
        Total Ohmic action (A^2 s / m^4).
    """
    pulse = _make_pulse(shape, peak_j, float(config['t_pulse']),
                        float(config['gauss_fwhm']))
    n = int(config['n_pulses'])
    return n * analytic_charge(pulse), n * analytic_action(pulse)
# -------------------------------------------------------------------------


def _mean_se(vals):
    """Mean and standard error of a list, NaN if empty.

    Parameters
    ----------
    vals : list[float]
        Sample values.

    Returns
    -------
    mean : float
        Sample mean.
    se : float
        Standard error of the mean.
    """
    if not vals:
        return float('nan'), float('nan')
    arr = np.asarray(vals, dtype=float)
    return (float(arr.mean()),
            float(arr.std(ddof=1) / np.sqrt(arr.size))
            if arr.size > 1 else 0.0)
# -------------------------------------------------------------------------


def _cell_stats(root, shape, t_sub, peak_j):
    """Ensemble statistics of one (shape, T, J) train cell.

    Parameters
    ----------
    root : str
        Dataset directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    stats : {dict, None}
        Means and SEs over x-localized survivors: displacement d (nm),
        peak/mean drive speed (m/s), max D1 (nm), per-pulse cumulative
        displacement d_k (nm, k = 1..n_pulses), plus p_surv, n_total,
        charge and action of the train. None if the cell is absent.
    """
    paths = sorted(glob.glob(os.path.join(
        root, '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))))
    if not paths:
        return None
    d_nm, vmax, vavg, d1_max = [], [], [], []
    cum = None
    n_pulses = None
    charge = action = float('nan')
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        code, metrics = _classify(npz, config)
        if not _localized(code, metrics):
            continue
        n_pulses = int(config['n_pulses'])
        t_p = float(config['t_pulse'])
        charge, action = _pulse_charge_action(shape, peak_j, config)
        d_nm.append(_displacement(npz) * 1e9)
        v_hi, v_bar = _pulse_speed(npz, n_pulses * t_p)
        vmax.append(v_hi)
        vavg.append(v_bar)
        d1_max.append(float(np.max(npz['D1_top'])) * 1e9)
        # Cumulative displacement at each pulse boundary k*t_p.
        t = np.asarray(npz['t_sample'], dtype=float)
        cx = np.asarray(npz['cx_unwrapped'], dtype=float)
        cy = np.asarray(npz['cy_unwrapped'], dtype=float)
        rows = []
        for k in range(1, n_pulses + 1):
            idx = int(np.searchsorted(t, k * t_p))
            idx = min(idx, t.size - 1)
            rows.append(np.hypot(cx[idx] - cx[0], cy[idx] - cy[0]) * 1e9)
        if cum is None:
            cum = [[] for _ in range(n_pulses)]
        for k in range(n_pulses):
            cum[k].append(rows[k])
    n_surv = len(d_nm)
    if n_pulses is None:
        n_pulses = 1
        cum = [[]]
    d_m, d_se = _mean_se(d_nm)
    vmax_m, vmax_se = _mean_se(vmax)
    vavg_m, vavg_se = _mean_se(vavg)
    d1_m, d1_se = _mean_se(d1_max)
    d_k = [_mean_se(cum[k])[0] for k in range(n_pulses)]
    d_k_se = [_mean_se(cum[k])[1] for k in range(n_pulses)]
    return {
        'd': d_m, 'd_se': d_se, 'vmax': vmax_m, 'vmax_se': vmax_se,
        'vavg': vavg_m, 'vavg_se': vavg_se, 'd1': d1_m, 'd1_se': d1_se,
        'd_k': d_k, 'd_k_se': d_k_se, 'n_pulses': n_pulses,
        'p_surv': n_surv / len(paths), 'n_total': len(paths),
        'charge': charge, 'action': action,
    }
# -------------------------------------------------------------------------


def _load(root, shapes, temperatures):
    """Cache of per-(shape, T) cell statistics over the current ladder.

    Parameters
    ----------
    root : str
        Dataset directory for the box.
    shapes : list[str]
        Shape names.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    cache : dict
        cache[shape][t_sub] = list of (peak_j, stats) with stats
        present, ascending in current.
    """
    cache = {}
    for shape in shapes:
        cache[shape] = {}
        for t_sub in temperatures:
            row = []
            for peak_j in _currents(root, shape, t_sub):
                stats = _cell_stats(root, shape, t_sub, peak_j)
                if stats is not None:
                    row.append((peak_j, stats))
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


def _plot_vs_current(box_tag, cache, shapes, temperatures, fn_value,
                     ylabel, title, out_path):
    """One scalar metric versus current, per shape, per temperature.

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
    fn_value : callable
        Maps (peak_j, stats) -> (y, y_err); NaN y is skipped.
    ylabel : str
        Y-axis label.
    title : str
        Figure suptitle.
    out_path : str
        Destination PNG.
    """
    colors = _shape_colors(shapes)
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.8),
        squeeze=False)
    y_hi = 0.0
    for shape in shapes:
        for t_sub in temperatures:
            for peak_j, stats in cache[shape][t_sub]:
                y, e = fn_value(peak_j, stats)
                if np.isfinite(y):
                    y_hi = max(y_hi, y + (e if np.isfinite(e) else 0.0))
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        for shape in shapes:
            row = cache[shape][t_sub]
            xs, ys, es = [], [], []
            for peak_j, stats in row:
                y, e = fn_value(peak_j, stats)
                if np.isfinite(y):
                    xs.append(peak_j * 1e-11)
                    ys.append(y)
                    es.append(e if np.isfinite(e) else 0.0)
            if xs:
                ax.errorbar(xs, ys, yerr=es, fmt='o-',
                            color=colors[shape], capsize=2, label=shape)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylim(0.0, 1.05 * y_hi if y_hi > 0 else 1.0)
        ax.set_box_aspect(1)
        if col == 0:
            ax.set_ylabel(ylabel)
    axes[0][0].legend(loc='best', frameon=False, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_perpulse(box_tag, cache, temperatures, shape, out_path):
    """Cumulative displacement after each pulse, per current, per T.

    A clean ratchet gives evenly spaced steps (each pulse adds the same
    travel); saturating steps mean the skyrmion elongates and later
    pulses move it less.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    cache : dict
        Output of `_load`.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    shape : str
        Shape to draw (the reference square).
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.8),
        squeeze=False)
    y_hi = 0.0
    for t_sub in temperatures:
        for _peak_j, stats in cache[shape][t_sub]:
            fin = [v for v in stats['d_k'] if np.isfinite(v)]
            if fin:
                y_hi = max(y_hi, max(fin))
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        row = cache[shape][t_sub]
        cmap = plt.get_cmap('viridis')
        js = [pj for pj, _ in row]
        j_lo = min(js) if js else 0.0
        j_hi = max(js) if js else 1.0
        for peak_j, stats in row:
            ks = list(range(1, stats['n_pulses'] + 1))
            color = cmap((peak_j - j_lo) / max(j_hi - j_lo, 1e-30))
            ax.errorbar(ks, stats['d_k'], yerr=stats['d_k_se'],
                        fmt='o-', color=color, capsize=2,
                        label='%.1f' % (peak_j * 1e-11))
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel('pulse number')
        ax.set_xticks(list(range(1, 4)))
        ax.set_ylim(0.0, 1.05 * y_hi if y_hi > 0 else 1.0)
        ax.set_box_aspect(1)
        if col == 0:
            ax.set_ylabel('cumulative displacement (nm)')
    axes[0][0].legend(title=r'$J$ ($10^{11}$)', loc='best',
                      frameon=False, fontsize=8, ncol=2)
    fig.suptitle('%s -- %s: displacement after each pulse'
                 % (box_tag, shape))
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _series(root, shape, t_sub, peak_j, key):
    """Ensemble-mean time series over x-localized survivors.

    Parameters
    ----------
    root : str
        Dataset directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).
    key : {'speed', 'd1'}
        Which per-frame quantity to average.

    Returns
    -------
    t : numpy.ndarray(1d)
        Sample times (ns), or None if no survivors.
    mean : numpy.ndarray(1d)
        Ensemble mean of the quantity.
    """
    paths = sorted(glob.glob(os.path.join(
        root, '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))))
    t_ref, stack = None, []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        code, metrics = _classify(npz, config)
        if not _localized(code, metrics):
            continue
        t = np.asarray(npz['t_sample'], dtype=float)
        if key == 'speed':
            cx = np.asarray(npz['cx_unwrapped'], dtype=float)
            cy = np.asarray(npz['cy_unwrapped'], dtype=float)
            y = np.hypot(np.gradient(cx, t), np.gradient(cy, t))
        else:
            y = np.asarray(npz['D1_top'], dtype=float) * 1e9
        if t_ref is None:
            t_ref = t
        if y.shape == t_ref.shape:
            stack.append(y)
    if not stack:
        return None, None
    return t_ref * 1e9, np.mean(np.vstack(stack), axis=0)
# -------------------------------------------------------------------------


def _plot_waveform(box_tag, root, temperatures, shape, key, ylabel,
                   title, out_path):
    """Ensemble-mean |v|(t) or D1(t) over the train, per current, per T.

    The pulse boundaries are marked so the per-pulse repeat (or drift)
    is visible against the drive.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    root : str
        Dataset directory for the box.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    shape : str
        Shape to draw.
    key : {'speed', 'd1'}
        Series to average.
    ylabel : str
        Y-axis label.
    title : str
        Figure suptitle.
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.8),
        squeeze=False)
    cmap = plt.get_cmap('viridis')
    y_hi = 0.0
    drawn = {}
    for t_sub in temperatures:
        js = _currents(root, shape, t_sub)
        drawn[t_sub] = js
        for peak_j in js:
            t, y = _series(root, shape, t_sub, peak_j, key)
            if y is not None:
                y_hi = max(y_hi, float(np.nanmax(y)))
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        js = drawn[t_sub]
        j_lo = min(js) if js else 0.0
        j_hi = max(js) if js else 1.0
        n_pulses = 3
        t_p = None
        for peak_j in js:
            t, y = _series(root, shape, t_sub, peak_j, key)
            if y is None:
                continue
            color = cmap((peak_j - j_lo) / max(j_hi - j_lo, 1e-30))
            ax.plot(t, y, color=color,
                    label='%.1f' % (peak_j * 1e-11))
            if t_p is None:
                t_p = float(t[-1]) / (n_pulses + 1)
        if t_p is not None:
            for k in range(1, n_pulses + 1):
                ax.axvline(k * t_p, color='0.7', lw=0.8, ls=':')
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'$t$ (ns)')
        ax.set_ylim(0.0, 1.05 * y_hi if y_hi > 0 else 1.0)
        ax.set_box_aspect(1)
        if col == 0:
            ax.set_ylabel(ylabel)
    axes[0][0].legend(title=r'$J$ ($10^{11}$)', loc='best',
                      frameon=False, fontsize=8, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    fig.subplots_adjust(top=0.86)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_config_table(root, shape, temperatures, out_path):
    """Final ens0 m_z on the (T, J) grid for one shape.

    Parameters
    ----------
    root : str
        Dataset directory for the box.
    shape : str
        Pulse-shape name.
    temperatures : list[float]
        Substrate temperatures, one row each (K).
    out_path : str
        Destination PNG.
    """
    x_gate = 0.5
    norm = plt.Normalize(vmin=-1.0, vmax=1.0)
    order = sorted(range(len(temperatures)),
                   key=lambda k: temperatures[k], reverse=True)
    all_j = sorted({j for t in temperatures
                    for j in _currents(root, shape, t)})
    nt, nj = len(temperatures), len(all_j)
    fig, axes = plt.subplots(nt, nj, figsize=(2.2 * nj, 2.0 * nt),
                             squeeze=False)
    for r, i in enumerate(order):
        t_sub = temperatures[i]
        for k, peak_j in enumerate(all_j):
            ax = axes[r][k]
            ax.set_xticks([])
            ax.set_yticks([])
            paths = sorted(glob.glob(os.path.join(
                root, '%s_T%05.1f_j%.2e_ens000.npz'
                % (shape, t_sub, peak_j))))
            if not paths:
                ax.text(0.5, 0.5, '--', transform=ax.transAxes,
                        ha='center', va='center', color='0.6')
            else:
                npz = np.load(paths[0], allow_pickle=True)
                config = _decode_config(npz)
                code, metrics = _classify(npz, config)
                nx, ny = int(config['nx']), int(config['ny'])
                mz = np.asarray(npz['mz_final_top'],
                                dtype=float).reshape(ny, nx)
                a_nm = float(npz['L_x']) / nx * 1e9
                ax.imshow(mz, origin='lower', cmap='RdBu_r', norm=norm,
                          extent=[0.0, nx * a_nm, 0.0, ny * a_nm],
                          aspect='equal')
                if metrics['dx_lx'] >= x_gate:
                    code = code + 'p'
                ax.text(0.03, 0.93, code, transform=ax.transAxes,
                        ha='left', va='top', fontsize=11,
                        fontweight='bold',
                        bbox=dict(facecolor='white', alpha=0.8,
                                  edgecolor='none', pad=1.2))
            if k == 0:
                ax.set_ylabel('%.0f' % t_sub, fontsize=12, rotation=0,
                              labelpad=16, va='center', ha='right')
            if r == nt - 1:
                ax.set_xlabel('%.1f' % (peak_j * 1e-11), fontsize=11)
    fig.supxlabel(r'$J$ ($10^{11}$ A/m$^2$)', fontsize=14)
    fig.supylabel(r'$T$ (K)', fontsize=14)
    fig.suptitle('%s -- final configuration (3 consecutive pulses)' % shape,
                 fontsize=13)
    fig.subplots_adjust(left=0.08, right=0.99, top=0.93, bottom=0.08,
                        hspace=0.3, wspace=0.06)
    fig.savefig(out_path, dpi=130, bbox_inches='tight')
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _render_gif(anim_path, out_path, fps):
    """Render an anim_*.npz top-layer m_z stream to a GIF.

    Parameters
    ----------
    anim_path : str
        Path to the field-dump NPZ (keys m_top, time_s).
    out_path : str
        Destination GIF.
    fps : int
        Frames per second.
    """
    z = np.load(anim_path, allow_pickle=True)
    m = np.asarray(z['m_top'], dtype=float)
    ts = np.asarray(z['time_s'], dtype=float)
    n_frames, ny, nx, _ = m.shape
    fig, ax = plt.subplots(figsize=(6.0, 6.0 * ny / nx))
    im = ax.imshow(m[0, :, :, 2], origin='lower', cmap='RdBu_r',
                   vmin=-1.0, vmax=1.0, aspect='equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ttl = ax.set_title('t = %.2f ns' % (ts[0] * 1e9))

    def _update(k):
        im.set_data(m[k, :, :, 2])
        ttl.set_text('t = %.2f ns' % (ts[k] * 1e9))
        return im, ttl

    anim = manim.FuncAnimation(fig, _update, frames=n_frames, blit=False)
    anim.save(out_path, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_train_schematic(shapes, t_p, gauss_fwhm, n_pulses, out_path):
    """Schematic of the applied drive J(t) for the pulse train.

    The three identical pulses are placed back to back at offsets
    k * t_p (k = 0..n_pulses-1) and summed, exactly as the driver
    builds them; the amplitude is normalised to the peak so only the
    shape sequence is shown.

    Parameters
    ----------
    shapes : list[str]
        Shape names, one row each.
    t_p : float
        Single-pulse duration (s).
    gauss_fwhm : float
        Gaussian FWHM (s), used only by the gaussian shape.
    n_pulses : int
        Number of pulses in the train.
    out_path : str
        Destination PNG.
    """
    t = np.linspace(0.0, (n_pulses + 1) * t_p, 2000)
    fig, axes = plt.subplots(
        len(shapes), 1, figsize=(7.0, 1.9 * len(shapes)),
        sharex=True, squeeze=False)
    for r, shape in enumerate(shapes):
        base = _make_pulse(shape, 1.0, t_p, gauss_fwhm)
        j = np.array([sum(base(ti - k * t_p) for k in range(n_pulses))
                      for ti in t])
        ax = axes[r][0]
        ax.plot(t * 1e9, j, color='C0')
        ax.fill_between(t * 1e9, 0.0, j, color='C0', alpha=0.2)
        for k in range(1, n_pulses + 1):
            ax.axvline(k * t_p * 1e9, color='0.7', lw=0.7, ls=':')
        ax.set_ylabel(shape, fontsize=10)
        ax.set_ylim(0.0, 1.1)
        ax.set_yticks([0.0, 1.0])
    axes[-1][0].set_xlabel(r'$t$ (ns)')
    fig.suptitle(r'Applied drive $J(t)/J_{\mathrm{peak}}$: '
                 r'three consecutive pulses')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the train dataset and write the figures and GIFs."""
    plt.rcParams.update({
        'axes.labelsize': 14, 'axes.titlesize': 14,
        'xtick.labelsize': 11, 'ytick.labelsize': 11,
        'figure.titlesize': 16})
    # =========================== User Configuration =========================
    root_base = ('output/'
                 'sweeps_driving_T/pulse_shape/pulses_seq3')
    box_tag = 'hk36_D0p72_700x500'
    out_dir = 'output/figures_driving_T/pulses_seq3'
    shapes = ['square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
              'tri_symmetric', 'gaussian']
    temperatures = [0.0, 10.0, 50.0, 100.0]
    gif_cells = [('square', 10.0, 3.0e11), ('square', 100.0, 3.0e11),
                 ('tri_sharpfall', 10.0, 6.0e11),
                 ('square', 10.0, 6.0e11)]
    gif_fps = 8
    # ======================= End User Configuration =========================
    root = os.path.join(root_base, box_tag)
    if not os.path.isdir(root):
        raise RuntimeError('main: dataset not found: %r.' % root)
    os.makedirs(out_dir, exist_ok=True)
    _plot_train_schematic(
        ['gaussian', 'tri_sharpfall'], 500.0e-12,
        250.0e-12, 3,
        os.path.join(out_dir, 'seq3_train_schematic_%s.png' % box_tag))
    print('loading %s ...' % box_tag)
    cache = _load(root, shapes, temperatures)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Efficiency: displacement per delivered charge and per action.
    _plot_vs_current(
        box_tag, cache, shapes, temperatures,
        lambda pj, s: (s['d'] / (s['charge'] / 1e11 / 1e-9),
                       s['d_se'] / (s['charge'] / 1e11 / 1e-9)),
        r'$d/\!\int\! J\,dt$ (nm per $10^{11}$A/m$^2\cdot$ns)',
        '%s -- displacement per charge (3 consecutive pulses)' % box_tag,
        os.path.join(out_dir, 'seq3_rank_d_per_charge_%s.png' % box_tag))
    _plot_vs_current(
        box_tag, cache, shapes, temperatures,
        lambda pj, s: (s['d'] / (s['action'] / 1e22 / 1e-9),
                       s['d_se'] / (s['action'] / 1e22 / 1e-9)),
        r'$d/\!\int\! J^2 dt$ (nm per $10^{22}\cdot$ns)',
        '%s -- displacement per action (3 consecutive pulses)' % box_tag,
        os.path.join(out_dir, 'seq3_rank_d_per_action_%s.png' % box_tag))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Speed and deformation.
    _plot_vs_current(
        box_tag, cache, shapes, temperatures,
        lambda pj, s: (s['vmax'], s['vmax_se']),
        r'$v_{\mathrm{max}}$ (m/s)',
        '%s -- peak drive speed (3 consecutive pulses)' % box_tag,
        os.path.join(out_dir, 'seq3_speed_max_%s.png' % box_tag))
    _plot_vs_current(
        box_tag, cache, shapes, temperatures,
        lambda pj, s: (s['vavg'], s['vavg_se']),
        r'$v_{\mathrm{avg}}$ (m/s)',
        '%s -- mean drive speed (3 consecutive pulses)' % box_tag,
        os.path.join(out_dir, 'seq3_speed_avg_%s.png' % box_tag))
    _plot_vs_current(
        box_tag, cache, shapes, temperatures,
        lambda pj, s: (s['d1'], s['d1_se']),
        r'max $D_1$ (nm)',
        '%s -- deformation (3 consecutive pulses)' % box_tag,
        os.path.join(out_dir, 'seq3_deformation_%s.png' % box_tag))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Train-specific: per-pulse increments and the waveforms. Shown for
    # square (which, being contiguous constant current, reads as one
    # long pulse) and for two shaped pulses (gaussian, sharp-fall
    # triangle) whose amplitude returns toward each pulse edge, so the
    # three pulses appear as three distinct velocity humps.
    for shape in ('square', 'gaussian', 'tri_sharpfall'):
        _plot_perpulse(
            box_tag, cache, temperatures, shape,
            os.path.join(out_dir,
                         'seq3_perpulse_disp_%s_%s.png'
                         % (shape, box_tag)))
        _plot_waveform(
            box_tag, root, temperatures, shape, 'speed',
            r'$\langle|v|\rangle$ (m/s)',
            '%s -- %s velocity waveform (3 consecutive pulses)'
            % (box_tag, shape),
            os.path.join(out_dir,
                         'seq3_vt_train_%s_%s.png' % (shape, box_tag)))
        _plot_waveform(
            box_tag, root, temperatures, shape, 'd1',
            r'$\langle D_1\rangle$ (nm)',
            '%s -- %s deformation waveform (3 consecutive pulses)'
            % (box_tag, shape),
            os.path.join(out_dir,
                         'seq3_d1t_train_%s_%s.png' % (shape, box_tag)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Config tables for the reference shapes.
    for shape in ('square', 'tri_sharpfall'):
        _plot_config_table(
            root, shape, temperatures,
            os.path.join(out_dir, 'seq3_config_%s_%s.png'
                         % (shape, box_tag)))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # GIFs of selected cells.
    for shape, t_sub, peak_j in gif_cells:
        anim = os.path.join(
            root, 'anim_%s_T%05.1f_j%.2e.npz' % (shape, t_sub, peak_j))
        if not os.path.isfile(anim):
            print('skip gif (absent): %s' % os.path.basename(anim))
            continue
        out = os.path.join(
            out_dir, 'seq3_%s_T%03.0f_j%.1f_%s.gif'
            % (shape, t_sub, peak_j * 1e-11, box_tag))
        _render_gif(anim, out, gif_fps)


# =============================================================================
if __name__ == '__main__':
    main()
