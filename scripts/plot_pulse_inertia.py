"""Skyrmion inertia (Pham S48) recast onto the pulse-shape campaign.

Reads the square-pulse production trajectories of the finite-T
pulse-shape study and characterises the velocity transient at pulse
turn-on and turn-off. Unlike the original Pham S48 -- which swept the
RKKY field and found an underdamped, ringing response in some regimes
-- the SAF racetrack at D = 0.72 mJ/m^2, H_k,top = 36 mT is
OVERDAMPED: the rise is a saturating ramp and the fall a monotonic
exponential decay to a small residual drift, with no overshoot and no
inertial oscillation even in the deterministic T = 0 member. The
inertial-frequency panel of the paper is therefore replaced by the
temperature dependence of the decay rate.

Panels (output/figures_driving_T/inertia/):

S48_A_vt_full.png
    Complete v(t) with the pulse window shaded, one trace per peak
    current, T = 0 -- the overview.
S48_B_vt_rise.png
    v(t) on the rising edge (0 -> 200 ps), one trace per peak
    current, T = 0 (deterministic, cleanest).
S48_C_vt_fall.png
    v(t) on the falling edge (t_pulse -> t_pulse + 250 ps), one
    trace per peak current, T = 0.
S48_D_invtau.png
    Inverse decay time 1/tau of the rise and fall exponential fits
    versus peak current, T = 0.
S48_E_invtau_T.png
    Inverse decay time versus substrate temperature at a common
    peak current, rise and fall pooled, with the ensemble standard
    deviation as the error bar. Finite-T members that do not end as
    a compact/elongated skyrmion are excluded via the campaign
    field classifier.

The rise time is fit over a fixed early window (0 -> 90 ps) that
ends before the mild velocity overshoot seen at the highest currents
-- a single saturating exponential cannot represent that overshoot,
and a wider window otherwise injects a spurious current dependence
into 1/tau_rise. The falling edge is a clean monotone exponential and
uses the full 250 ps window.

The pulse timing (t_pulse) is read from each file's config record,
never assumed. No argparse; configure via the variables at the top of
main().

Run with:
    python -m scripts.plot_pulse_inertia

Functions
---------
main
    Load the square-pulse production cells and write the five panels.
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
from scipy.optimize import curve_fit
# Local
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
    """Decode the per-file config record into a dict.

    The C++ driver stores `meta_config_repr` as an array of ASCII
    codes of a JSON string. Decode it to characters and parse.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.

    Returns
    -------
    config : dict
        The parsed configuration (shape, t_pulse, nx, ny, ...).
    """
    if 'meta_config_repr' not in npz:
        raise RuntimeError(
            '_decode_config: trajectory has no meta_config_repr.')
    codes = np.atleast_1d(npz['meta_config_repr']).ravel()
    text = ''.join(chr(int(c)) for c in codes)
    return json.loads(text)
# -------------------------------------------------------------------------


def _speed(t, cx, cy):
    """Centred-difference speed magnitude of a centroid stream.

    Parameters
    ----------
    t : numpy.ndarray(1d)
        Sample times (s).
    cx, cy : numpy.ndarray(1d)
        Unwrapped centroid coordinates (m).

    Returns
    -------
    v : numpy.ndarray(1d)
        Speed |dr/dt| (m/s).
    """
    vx = np.gradient(cx, t)
    vy = np.gradient(cy, t)
    return np.sqrt(vx * vx + vy * vy)
# -------------------------------------------------------------------------


def _load_member(path):
    """Load one trajectory: time, speed and config.

    Parameters
    ----------
    path : str
        Path to a trajectory NPZ.

    Returns
    -------
    t : numpy.ndarray(1d)
        Sample times (s).
    v : numpy.ndarray(1d)
        Top-layer speed (m/s).
    config : dict
        Parsed per-file configuration.
    npz : numpy.lib.npyio.NpzFile
        The opened archive (for classification fields).
    """
    npz = np.load(path, allow_pickle=True)
    t = npz['t_sample']
    v = _speed(t, npz['cx_unwrapped'], npz['cy_unwrapped'])
    return t, v, _decode_config(npz), npz
# -------------------------------------------------------------------------


def _cell_paths(prod_dir, shape, t_sub, peak_j):
    """Sorted ensemble-member paths for one (shape, T, J) cell.

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
    paths : list[str]
        Every realization path; raises if none are found.
    """
    pattern = os.path.join(
        prod_dir,
        '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError(
            '_cell_paths: no members match %r.' % pattern)
    return paths
# -------------------------------------------------------------------------


def _available_currents(prod_dir, shape, t_sub):
    """Peak currents present for one (shape, T), ascending.

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
    if not currents:
        raise RuntimeError(
            '_available_currents: none for %s T=%g.' % (shape, t_sub))
    return sorted(currents)
# -------------------------------------------------------------------------


def _rise_model(t, amp, tau):
    """Saturating-exponential rise v(t) = amp (1 - exp(-t / tau)).

    Parameters
    ----------
    t : numpy.ndarray(1d)
        Time from pulse start (s).
    amp : float
        Plateau amplitude (m/s).
    tau : float
        Rise time constant (s).

    Returns
    -------
    v : numpy.ndarray(1d)
        Model speed (m/s).
    """
    return amp * (1.0 - np.exp(-t / tau))
# -------------------------------------------------------------------------


def _fall_model(dt, vp, vres, tau):
    """Exponential decay v = vres + (vp - vres) exp(-dt / tau).

    Parameters
    ----------
    dt : numpy.ndarray(1d)
        Time since pulse turn-off (s).
    vp : float
        Speed at turn-off (m/s).
    vres : float
        Residual speed at long time (m/s).
    tau : float
        Fall time constant (s).

    Returns
    -------
    v : numpy.ndarray(1d)
        Model speed (m/s).
    """
    return vres + (vp - vres) * np.exp(-dt / tau)
# -------------------------------------------------------------------------


def _fit_rise(t, v, t_end):
    """Fit v(t) = A (1 - exp(-t / tau)) on the rising edge.

    The fit window must end before the mild velocity overshoot that
    appears at the highest currents; a single saturating exponential
    cannot represent the overshoot, and including its tail biases tau.

    Parameters
    ----------
    t : numpy.ndarray(1d)
        Sample times from pulse start (s).
    v : numpy.ndarray(1d)
        Speed (m/s).
    t_end : float
        Upper edge of the fit window (s), before the overshoot.

    Returns
    -------
    tau : float
        Rise time constant (s); NaN if the fit fails.
    """
    window = (t >= 0.0) & (t <= t_end)
    tt = t[window]
    yy = v[window]
    if tt.size < 4:
        return float('nan')
    a0 = float(np.max(yy))
    tau0 = 0.25 * float(t_end)
    bounds = ([0.0, 1.0e-12], [1.0e4, 1.0e-8])
    try:
        popt, _ = curve_fit(
            _rise_model, tt, yy, p0=(a0, tau0), bounds=bounds,
            maxfev=20000)
        return float(popt[1])
    except Exception:
        return float('nan')
# -------------------------------------------------------------------------


def _fit_fall(t, v, t_pulse, t_end):
    """Fit v(t) = vres + (vp - vres) exp(-(t - t_pulse)/tau).

    Parameters
    ----------
    t : numpy.ndarray(1d)
        Sample times (s).
    v : numpy.ndarray(1d)
        Speed (m/s).
    t_pulse : float
        Pulse turn-off time (s).
    t_end : float
        Upper edge of the fit window (s).

    Returns
    -------
    tau : float
        Fall time constant (s); NaN if the fit fails.
    """
    window = (t >= t_pulse) & (t <= t_end)
    tt = t[window] - t_pulse
    yy = v[window]
    if tt.size < 4:
        return float('nan')
    vp0 = float(yy[0])
    vres0 = float(np.min(yy))
    tau0 = 0.25 * float(t_end - t_pulse)
    bounds = ([0.0, -1.0e3, 1.0e-12], [1.0e4, 1.0e4, 1.0e-8])
    try:
        popt, _ = curve_fit(
            _fall_model, tt, yy, p0=(vp0, vres0, tau0),
            bounds=bounds, maxfev=20000)
        return float(popt[2])
    except Exception:
        return float('nan')
# -------------------------------------------------------------------------


def _member_class(npz, config):
    """Class code of a member's final drive frame.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.
    config : dict
        Its parsed configuration (for nx, ny).

    Returns
    -------
    code : str
        'S', 'E', 'L' or 'A' from the campaign classifier.
    """
    nx = int(config['nx'])
    ny = int(config['ny'])
    mz = np.asarray(npz['mz_final_top'], dtype=float).reshape(ny, nx)
    q_abs = abs(float(npz['Q'][-1]))
    d1 = float(npz['D1_top'][-1])
    d2 = float(npz['D2_top'][-1])
    l_x = float(npz['L_x'])
    code, _ = classify_field(mz, q_abs, d1, d2, l_x)
    return code
# -------------------------------------------------------------------------


def _plot_edge(prod_dir, shape, t_sub, currents, edge, t_win,
               out_path):
    """Plot the velocity transient on one edge, one trace per J.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Temperature to draw (K); use the deterministic T = 0.
    currents : list[float]
        Peak currents to overlay (A/m^2).
    edge : {'rise', 'fall'}
        Which transient to draw.
    t_win : float
        Window length past the edge (s).
    out_path : str
        Destination PNG.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    j_min = float(min(currents))
    j_max = float(max(currents))
    for peak_j in currents:
        path = _cell_paths(prod_dir, shape, t_sub, peak_j)[0]
        t, v, config, _ = _load_member(path)
        t_pulse = float(config['t_pulse'])
        if edge == 'rise':
            window = (t >= 0.0) & (t <= t_win)
            t_ref = 0.0
        else:
            window = (t >= t_pulse) & (t <= t_pulse + t_win)
            t_ref = t_pulse
        color = cmap(
            (peak_j - j_min) / max(j_max - j_min, 1.0e-30))
        ax.plot((t[window] - t_ref) * 1e12, v[window],
                color=color, marker='o', markersize=3,
                label='%.1f' % (peak_j / 1.0e11))
    ax.set_xlabel(r'$t - t_{\mathrm{%s}}$ (ps)'
                  % ('rise' if edge == 'rise' else 'fall'))
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.legend(title=r'$J$ ($10^{11}$ A/m$^2$)', loc='best',
              frameon=False, fontsize=9, ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_full(prod_dir, shape, t_sub, currents, out_path):
    """Plot the complete v(t) with the pulse window shaded.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Temperature to draw (K).
    currents : list[float]
        Peak currents to overlay (A/m^2).
    out_path : str
        Destination PNG.
    """
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    j_min = float(min(currents))
    j_max = float(max(currents))
    t_pulse_shade = None
    for peak_j in currents:
        path = _cell_paths(prod_dir, shape, t_sub, peak_j)[0]
        t, v, config, _ = _load_member(path)
        t_pulse_shade = float(config['t_pulse'])
        color = cmap(
            (peak_j - j_min) / max(j_max - j_min, 1.0e-30))
        ax.plot(t * 1e12, v, color=color,
                label='%.1f' % (peak_j / 1.0e11))
    ax.axvspan(0.0, t_pulse_shade * 1e12, color='violet', alpha=0.12,
               label='pulse on')
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.legend(title=r'$J$ ($10^{11}$ A/m$^2$)', loc='best',
              frameon=False, fontsize=9, ncol=2)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_invtau_vs_j(prod_dir, shape, t_sub, currents, rise_fit_win,
                      fall_win, out_path):
    """Plot 1/tau_rise and 1/tau_fall versus peak current.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Temperature to draw (K); the deterministic T = 0.
    currents : list[float]
        Peak currents (A/m^2).
    rise_fit_win, fall_win : float
        Fit-window lengths for the rise and fall edges (s). The rise
        window ends before the high-current overshoot.
    out_path : str
        Destination PNG.
    """
    j_arr, inv_rise, inv_fall = [], [], []
    for peak_j in currents:
        path = _cell_paths(prod_dir, shape, t_sub, peak_j)[0]
        t, v, config, _ = _load_member(path)
        t_pulse = float(config['t_pulse'])
        tau_r = _fit_rise(t, v, rise_fit_win)
        tau_f = _fit_fall(t, v, t_pulse, t_pulse + fall_win)
        if not np.isfinite(tau_r):
            print('  WARN rise fit failed: %s J=%.2e'
                  % (shape, peak_j))
        if not np.isfinite(tau_f):
            print('  WARN fall fit failed: %s J=%.2e'
                  % (shape, peak_j))
        j_arr.append(peak_j / 1.0e11)
        inv_rise.append(1.0 / tau_r if np.isfinite(tau_r)
                        else float('nan'))
        inv_fall.append(1.0 / tau_f if np.isfinite(tau_f)
                        else float('nan'))
    fig, ax = plt.subplots()
    ax.plot(j_arr, np.array(inv_rise) * 1e-9, 'o-', color='C0',
            label='rise')
    ax.plot(j_arr, np.array(inv_fall) * 1e-9, 's-', color='C3',
            label='fall')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$1/\tau$ (1/ns)')
    ax.legend(loc='best', frameon=False)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_invtau_vs_t(prod_dir, shape, temperatures, common_j,
                      rise_fit_win, fall_win, out_path):
    """Plot 1/tau versus temperature at a common peak current.

    Pools the rise and fall time constants over all surviving
    ensemble members; the error bar is their standard deviation.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    temperatures : list[float]
        Substrate temperatures to draw (K).
    common_j : float
        Peak current present at every temperature (A/m^2).
    rise_fit_win, fall_win : float
        Fit-window lengths for the rise and fall edges (s). The rise
        window ends before the high-current overshoot.
    out_path : str
        Destination PNG.
    """
    t_arr, mean_inv, std_inv = [], [], []
    for t_sub in temperatures:
        paths = _cell_paths(prod_dir, shape, t_sub, common_j)
        taus = []
        n_skip = 0
        for path in paths:
            t, v, config, npz = _load_member(path)
            if t_sub > 0.0 and _member_class(npz, config) not in (
                    'S', 'E'):
                n_skip += 1
                continue
            t_pulse = float(config['t_pulse'])
            tau_r = _fit_rise(t, v, rise_fit_win)
            tau_f = _fit_fall(t, v, t_pulse, t_pulse + fall_win)
            for tau in (tau_r, tau_f):
                if np.isfinite(tau):
                    taus.append(1.0 / tau)
        if not taus:
            print('  WARN no usable fits: %s T=%g' % (shape, t_sub))
            continue
        inv = np.array(taus) * 1e-9
        t_arr.append(t_sub)
        mean_inv.append(float(np.mean(inv)))
        std_inv.append(float(np.std(inv)))
        print('  T=%3.0f K: 1/tau = %.3f +/- %.3f 1/ns '
              '(n_fit=%d, n_skip=%d)'
              % (t_sub, mean_inv[-1], std_inv[-1], len(taus), n_skip))
    fig, ax = plt.subplots()
    ax.errorbar(t_arr, mean_inv, yerr=std_inv, fmt='o-', color='C2',
                capsize=4)
    ax.set_xlabel(r'$T$ (K)')
    ax.set_ylabel(r'$1/\tau$ (1/ns)')
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_fit_diagnostic(prod_dir, shape, t_sub, peak_j, rise_fit_win,
                         fall_win, out_path):
    """Illustrate the 1/tau extraction on one representative cell.

    Draws the measured v(t) (ground truth), shades the rise and fall
    fit windows, and overlays the fitted exponential curves -- solid
    inside each window, dashed where extrapolated -- so the fit and
    its window can be checked against the data by eye.

    Parameters
    ----------
    prod_dir : str
        Production directory for the box.
    shape : str
        Pulse-shape name.
    t_sub : float
        Temperature of the drawn member (K); use T = 0.
    peak_j : float
        Peak current density of the drawn cell (A/m^2).
    rise_fit_win : float
        Rising-edge fit-window length (s).
    fall_win : float
        Falling-edge fit-window length (s).
    out_path : str
        Destination PNG.
    """
    path = _cell_paths(prod_dir, shape, t_sub, peak_j)[0]
    t, v, config, _ = _load_member(path)
    t_pulse = float(config['t_pulse'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Refit both edges keeping the full parameter vectors for drawing.
    rise_m = (t >= 0.0) & (t <= rise_fit_win)
    rp, _ = curve_fit(
        _rise_model, t[rise_m], v[rise_m],
        p0=(float(np.max(v[rise_m])), 0.25 * rise_fit_win),
        bounds=([0.0, 1.0e-12], [1.0e4, 1.0e-8]), maxfev=20000)
    fall_m = (t >= t_pulse) & (t <= t_pulse + fall_win)
    fp, _ = curve_fit(
        _fall_model, t[fall_m] - t_pulse, v[fall_m],
        p0=(float(v[fall_m][0]), float(np.min(v[fall_m])),
            0.25 * fall_win),
        bounds=([0.0, -1.0e3, 1.0e-12], [1.0e4, 1.0e4, 1.0e-8]),
        maxfev=20000)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(t * 1e12, v, 'o', color='0.35', markersize=3,
            label='data (ground truth)')
    ax.axvspan(0.0, rise_fit_win * 1e12, color='C0', alpha=0.12,
               label='rise fit window')
    ax.axvspan(t_pulse * 1e12, (t_pulse + fall_win) * 1e12,
               color='C3', alpha=0.12, label='fall fit window')
    # Rise fit: solid in the window, dashed extrapolation past it.
    t_in = np.linspace(0.0, rise_fit_win, 100)
    t_ex = np.linspace(rise_fit_win, 0.30e-9, 100)
    ax.plot(t_in * 1e12, _rise_model(t_in, *rp), '-', color='C0',
            linewidth=2, label='rise fit')
    ax.plot(t_ex * 1e12, _rise_model(t_ex, *rp), '--', color='C0',
            linewidth=1)
    # Fall fit: solid in the window, dashed extrapolation past it.
    d_in = np.linspace(0.0, fall_win, 100)
    d_ex = np.linspace(fall_win, fall_win + 0.15e-9, 100)
    ax.plot((t_pulse + d_in) * 1e12, _fall_model(d_in, *fp), '-',
            color='C3', linewidth=2, label='fall fit')
    ax.plot((t_pulse + d_ex) * 1e12, _fall_model(d_ex, *fp), '--',
            color='C3', linewidth=1)
    ax.set_xlabel(r'$t$ (ps)')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.set_title(
        r'$J = %.1f\times10^{11}$ A/m$^2$, $T=0$:  '
        r'$1/\tau_{\mathrm{rise}} = %.2f$/ns,  '
        r'$1/\tau_{\mathrm{fall}} = %.2f$/ns'
        % (peak_j / 1e11, 1e-9 / rp[1], 1e-9 / fp[2]), fontsize=11)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Load the square-pulse cells and write the five S48 panels."""
    # =========================== User Configuration =========================
    prod_root = ('/Volumes/T7/skyrmion_simulator/output/'
                 'sweeps_driving_T/pulse_shape/production')
    box_tag = 'hk36_D0p72_700x500'
    shape = 'square'
    out_dir = 'output/figures_driving_T/inertia'
    rise_plot_win = 0.20e-9    # s, rising-edge plot window (shows peak)
    rise_fit_win = 0.09e-9     # s, rising-edge fit window (pre-overshoot)
    fall_win = 0.25e-9         # s, falling-edge fit/plot window
    temperatures = [0.0, 10.0, 50.0, 100.0]
    common_j = 3.0e11          # A/m^2, present at every temperature
    # ======================= End User Configuration =========================
    prod_dir = os.path.join(prod_root, box_tag)
    if not os.path.isdir(prod_dir):
        raise RuntimeError(
            'main: production directory not found: %r.' % prod_dir)
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # T = 0 deterministic ladder drives the transient panels A/B/C/D.
    # Order: full-trace overview (A), then the rise (B) and fall (C)
    # zooms, then the 1/tau-vs-J summary (D).
    currents = _available_currents(prod_dir, shape, 0.0)
    print('S48: %s %s, T=0 currents (1e11) = %s'
          % (box_tag, shape,
             ', '.join('%.1f' % (j / 1e11) for j in currents)))
    _plot_full(prod_dir, shape, 0.0, currents,
               os.path.join(out_dir, 'S48_A_vt_full.png'))
    _plot_edge(prod_dir, shape, 0.0, currents, 'rise', rise_plot_win,
               os.path.join(out_dir, 'S48_B_vt_rise.png'))
    _plot_edge(prod_dir, shape, 0.0, currents, 'fall', fall_win,
               os.path.join(out_dir, 'S48_C_vt_fall.png'))
    _plot_invtau_vs_j(prod_dir, shape, 0.0, currents, rise_fit_win,
                      fall_win,
                      os.path.join(out_dir, 'S48_D_invtau.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Temperature dependence at the common current (panel E).
    print('S48_E: 1/tau vs T at J = %.2e A/m^2' % common_j)
    _plot_invtau_vs_t(prod_dir, shape, temperatures, common_j,
                      rise_fit_win, fall_win,
                      os.path.join(out_dir, 'S48_E_invtau_T.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Appendix diagnostic: fit windows, ground truth and fit on one
    # high-current cell where the rise overshoot is visible.
    diag_j = 5.0e11
    _plot_fit_diagnostic(prod_dir, shape, 0.0, diag_j, rise_fit_win,
                         fall_win,
                         os.path.join(out_dir, 'invtau_fit_windows.png'))


# =============================================================================
if __name__ == '__main__':
    main()
