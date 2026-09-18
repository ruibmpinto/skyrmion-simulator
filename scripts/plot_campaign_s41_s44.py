"""S41 velocity and S44 deformation from the DC track-width campaign.

Recasts Pham et al. (2024) figures S41 and S44 onto the finite-T DC
campaign (2 ns square drive, racetrack, D = 0.72 mJ/m^2, Set A) with
no new simulations. Every panel is an ensemble average over the 100
thermal realizations of each (T, J) cell, gated on the field
classifier so only members ending as a compact (S) or elongated (E)
skyrmion contribute -- labyrinth and reversed members never pollute a
mean. The shaded band or error bar is the standard error.

Panels, one figure per box, columns are T in {10, 50, 100} K:

S41_vt_<box>.png
    Ensemble-mean |v|(t) with a +/- SE band, one trace per peak
    current. The DC drive runs the whole 2 ns, so this is the drift
    speed, not a pulse transient.
S44_A_D1D2_t_<box>.png
    Ensemble-mean ellipse axes D1(t) (major) and D2(t) (minor) of
    the top layer at the largest surviving current, showing the
    stretch building up under drive.
S44_B_vD_J_<box>.png
    Drive-average speed v_avg (left axis) and the extremal axes
    max D1, min D2 (right axis) versus peak current, with SE bars.

The paper's S44_C (domain-wall angle psi versus J) is omitted: the
campaign driver does not record psi, only the ellipse orientation.

No argparse; configure via the variables at the top of main().

Run with:
    python -m scripts.plot_campaign_s41_s44

Functions
---------
main
    Write the S41 and S44 panels for every configured box.
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
        The parsed configuration (nx, ny, ...).
    """
    if 'meta_config_repr' not in npz:
        raise RuntimeError(
            '_decode_config: trajectory has no meta_config_repr.')
    codes = np.atleast_1d(npz['meta_config_repr']).ravel()
    return json.loads(''.join(chr(int(c)) for c in codes))
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
    return np.sqrt(np.gradient(cx, t) ** 2 + np.gradient(cy, t) ** 2)
# -------------------------------------------------------------------------


def _survives(npz, config):
    """True if the member is a single-winding, x-localized skyrmion.

    Keeps compact (S) and elongated (E) skyrmions, and spanning-flagged
    (P) members that are still x-localized (D_x/L_x < 0.5). The P flag
    fires when the reversed domain touches both periodic-x edges; on a
    periodic track that also catches a skyrmion merely sitting on the
    seam, which is a valid localized bit (a translation off the seam).
    Only a domain that genuinely spans the track (D_x/L_x >= 0.5) is
    excluded, along with labyrinth (L), annihilated (A) and
    reversed-background (R) members.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        An opened trajectory archive.
    config : dict
        Its parsed configuration (for nx, ny).

    Returns
    -------
    ok : bool
        Whether the member is a surviving, x-localized skyrmion.
    """
    nx = int(config['nx'])
    ny = int(config['ny'])
    mz = np.asarray(npz['mz_final_top'], dtype=float).reshape(ny, nx)
    code, metrics = classify_field(
        mz, abs(float(npz['Q'][-1])), float(npz['D1_top'][-1]),
        float(npz['D2_top'][-1]), float(npz['L_x']))
    if code in ('S', 'E'):
        return True
    return code == 'P' and metrics['dx_lx'] < 0.5
# -------------------------------------------------------------------------


def _cell_currents(campaign_dir, t_sub):
    """Ascending peak currents present for one (box, T).

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    t_sub : float
        Substrate temperature (K).

    Returns
    -------
    currents : list[float]
        Peak current densities (A/m^2), sorted ascending.
    """
    pattern = os.path.join(
        campaign_dir, 'T%05.1f_j*_ens000.npz' % t_sub)
    currents = set()
    for path in glob.glob(pattern):
        tag = os.path.basename(path).split('_j')[1].split('_ens')[0]
        currents.add(float(tag))
    return sorted(currents)
# -------------------------------------------------------------------------


def _load_cell(campaign_dir, t_sub, peak_j):
    """Load the surviving members of one (T, J) cell.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    cell : dict
        't' (times), 'v' (n_surv x n_t speed), 'd1', 'd2'
        (n_surv x n_t axes, m), 'vavg' (n_surv drive speeds),
        'n_surv' and 'n_total'. Raises if no member files exist.
    """
    pattern = os.path.join(
        campaign_dir, 'T%05.1f_j%.2e_ens*.npz' % (t_sub, peak_j))
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise RuntimeError('_load_cell: no members for %r.' % pattern)
    t_ref = None
    v_rows, d1_rows, d2_rows, vavg = [], [], [], []
    for path in paths:
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        if not _survives(npz, config):
            continue
        t = npz['t_sample']
        if t_ref is None:
            t_ref = t
        v_rows.append(_speed(t, npz['cx_unwrapped'],
                             npz['cy_unwrapped']))
        d1_rows.append(np.asarray(npz['D1_top'], dtype=float))
        d2_rows.append(np.asarray(npz['D2_top'], dtype=float))
        vavg.append(float(npz['velocity']))
    return {
        't': t_ref,
        'v': np.array(v_rows),
        'd1': np.array(d1_rows),
        'd2': np.array(d2_rows),
        'vavg': np.array(vavg),
        'n_surv': len(v_rows),
        'n_total': len(paths),
    }
# -------------------------------------------------------------------------


def _mean_se(rows):
    """Column-wise mean and standard error of stacked rows.

    Parameters
    ----------
    rows : numpy.ndarray(2d)
        Shape (n_member, n_sample).

    Returns
    -------
    mean : numpy.ndarray(1d)
        Per-column mean.
    se : numpy.ndarray(1d)
        Per-column standard error (std / sqrt(n_member)).
    """
    n = rows.shape[0]
    return rows.mean(axis=0), rows.std(axis=0) / np.sqrt(n)
# -------------------------------------------------------------------------


def _s41_data(campaign_dir, temperatures):
    """Ensemble-mean |v|(t) per surviving cell, and the box's peak.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    data : dict
        'all_j' (union of currents, for colour), 'per_t' (t_sub ->
        list of (peak_j, t, mean, se) for surviving cells), and
        'v_max' (largest mean + SE seen, for the shared y-range).
    """
    all_j = sorted({j for t in temperatures
                    for j in _cell_currents(campaign_dir, t)})
    per_t = {}
    v_max = 0.0
    for t_sub in temperatures:
        rows = []
        for peak_j in _cell_currents(campaign_dir, t_sub):
            cell = _load_cell(campaign_dir, t_sub, peak_j)
            if cell['n_surv'] == 0:
                continue
            mean, se = _mean_se(cell['v'])
            rows.append((peak_j, cell['t'], mean, se))
            v_max = max(v_max, float(np.max(mean + se)))
        per_t[t_sub] = rows
    return {'all_j': all_j, 'per_t': per_t, 'v_max': v_max}
# -------------------------------------------------------------------------


def _plot_s41(box_tag, data, temperatures, ylim, out_path):
    """Ensemble-mean |v|(t) per current, one column per temperature.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    data : dict
        Output of `_s41_data` for this box.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    ylim : tuple[float]
        Shared (low, high) y-limits, common to every box.
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.6),
        squeeze=False)
    cmap = plt.get_cmap('viridis')
    all_j = data['all_j']
    j_min, j_max = all_j[0], all_j[-1]
    handles = {}
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        for peak_j, t, mean, se in data['per_t'][t_sub]:
            color = cmap((peak_j - j_min) / max(j_max - j_min, 1e-30))
            line, = ax.plot(t * 1e9, mean, color=color,
                            label='%.1f' % (peak_j / 1e11))
            ax.fill_between(t * 1e9, mean - se, mean + se,
                            color=color, alpha=0.25, linewidth=0)
            handles.setdefault(peak_j, line)
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'$t$ (ns)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(r'$\langle|v|\rangle$ (m/s)')
        ax.set_box_aspect(1)
    if handles:
        ordered = [handles[j] for j in sorted(handles)]
        axes[0][0].legend(
            ordered, [h.get_label() for h in ordered],
            title=r'$J$ ($10^{11}$ A/m$^2$)', loc='best',
            frameon=False, ncol=2)
    fig.suptitle(box_tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _top_surviving_current(campaign_dir, t_sub, min_surv):
    """Largest current with at least min_surv surviving members.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    t_sub : float
        Substrate temperature (K).
    min_surv : int
        Minimum surviving-member count required.

    Returns
    -------
    peak_j : {float, None}
        The largest qualifying current, or None if none qualify.
    """
    chosen = None
    for peak_j in _cell_currents(campaign_dir, t_sub):
        if _load_cell(campaign_dir, t_sub, peak_j)['n_surv'] >= min_surv:
            chosen = peak_j
    return chosen
# -------------------------------------------------------------------------


def _s44a_data(campaign_dir, temperatures, min_surv):
    """D1(t), D2(t) at the top surviving current, and the axis span.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    temperatures : list[float]
        Substrate temperatures (K).
    min_surv : int
        Minimum survivors for a current to qualify.

    Returns
    -------
    data : dict
        'per_t' (t_sub -> dict with tt, peak_j, and D1/D2 mean, SE, or
        None when no cell qualifies) and 'lo'/'hi' (axis extent with
        the SE bands, for the shared y-range; inf/-inf if all empty).
    """
    per_t = {}
    lo, hi = float('inf'), float('-inf')
    for t_sub in temperatures:
        peak_j = _top_surviving_current(campaign_dir, t_sub, min_surv)
        if peak_j is None:
            per_t[t_sub] = None
            continue
        cell = _load_cell(campaign_dir, t_sub, peak_j)
        d1_m, d1_se = _mean_se(cell['d1'] * 1e9)
        d2_m, d2_se = _mean_se(cell['d2'] * 1e9)
        per_t[t_sub] = {
            'tt': cell['t'] * 1e9, 'peak_j': peak_j,
            'd1_m': d1_m, 'd1_se': d1_se,
            'd2_m': d2_m, 'd2_se': d2_se,
        }
        lo = min(lo, float(np.min(d2_m - d2_se)))
        hi = max(hi, float(np.max(d1_m + d1_se)))
    return {'per_t': per_t, 'lo': lo, 'hi': hi}
# -------------------------------------------------------------------------


def _plot_s44a(box_tag, data, temperatures, ylim, out_path):
    """Ensemble-mean D1(t), D2(t) at the top surviving current.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    data : dict
        Output of `_s44a_data` for this box.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    ylim : tuple[float]
        Shared (low, high) y-limits, common to every box.
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.6),
        squeeze=False)
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        cell = data['per_t'][t_sub]
        if cell is None:
            ax.set_title(r'$T = %g$ K (no cell)' % t_sub)
            ax.set_ylim(ylim)
            ax.set_box_aspect(1)
            continue
        tt = cell['tt']
        d1_m, d1_se = cell['d1_m'], cell['d1_se']
        d2_m, d2_se = cell['d2_m'], cell['d2_se']
        ax.plot(tt, d1_m, color='C3', label=r'$D_1$ (major)')
        ax.fill_between(tt, d1_m - d1_se, d1_m + d1_se, color='C3',
                        alpha=0.25, linewidth=0)
        ax.plot(tt, d2_m, color='C0', label=r'$D_2$ (minor)')
        ax.fill_between(tt, d2_m - d2_se, d2_m + d2_se, color='C0',
                        alpha=0.25, linewidth=0)
        ax.set_title(r'$T = %g$ K, $J = %.1f$'
                     % (t_sub, cell['peak_j'] / 1e11))
        ax.set_xlabel(r'$t$ (ns)')
        ax.set_ylim(ylim)
        if col == 0:
            ax.set_ylabel(r'axis (nm)')
        ax.legend(loc='best', frameon=False)
        ax.set_box_aspect(1)
    fig.suptitle(box_tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _s44b_data(campaign_dir, temperatures):
    """v_avg, max D1, min D2 versus current, and the two axis spans.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    temperatures : list[float]
        Substrate temperatures (K).

    Returns
    -------
    data : dict
        'per_t' (t_sub -> dict with j_plot and the mean/SE of v_avg,
        max D1, min D2 over surviving currents), plus 'v_lo'/'v_hi'
        (left-axis span) and 'd_lo'/'d_hi' (right-axis span), each
        widened by the SE bars.
    """
    per_t = {}
    v_lo, v_hi = float('inf'), float('-inf')
    d_lo, d_hi = float('inf'), float('-inf')
    for t_sub in temperatures:
        j_plot, v_m, v_se = [], [], []
        d1_m, d1_se, d2_m, d2_se = [], [], [], []
        for peak_j in _cell_currents(campaign_dir, t_sub):
            cell = _load_cell(campaign_dir, t_sub, peak_j)
            if cell['n_surv'] == 0:
                continue
            n = cell['n_surv']
            j_plot.append(peak_j / 1e11)
            v_m.append(float(cell['vavg'].mean()))
            v_se.append(float(cell['vavg'].std() / np.sqrt(n)))
            d1x = cell['d1'].max(axis=1) * 1e9
            d2n = cell['d2'].min(axis=1) * 1e9
            d1_m.append(float(d1x.mean()))
            d1_se.append(float(d1x.std() / np.sqrt(n)))
            d2_m.append(float(d2n.mean()))
            d2_se.append(float(d2n.std() / np.sqrt(n)))
        per_t[t_sub] = {
            'j_plot': j_plot, 'v_m': v_m, 'v_se': v_se,
            'd1_m': d1_m, 'd1_se': d1_se,
            'd2_m': d2_m, 'd2_se': d2_se,
        }
        for m, se in zip(v_m, v_se):
            v_lo, v_hi = min(v_lo, m - se), max(v_hi, m + se)
        for m, se in zip(d1_m, d1_se):
            d_lo, d_hi = min(d_lo, m - se), max(d_hi, m + se)
        for m, se in zip(d2_m, d2_se):
            d_lo, d_hi = min(d_lo, m - se), max(d_hi, m + se)
    return {'per_t': per_t, 'v_lo': v_lo, 'v_hi': v_hi,
            'd_lo': d_lo, 'd_hi': d_hi}
# -------------------------------------------------------------------------


def _plot_s44b(box_tag, data, temperatures, ylim_v, ylim_d, out_path):
    """v_avg, max D1, min D2 versus current, one column per T.

    Parameters
    ----------
    box_tag : str
        Box directory name (title only).
    data : dict
        Output of `_s44b_data` for this box.
    temperatures : list[float]
        Substrate temperatures for the columns (K).
    ylim_v : tuple[float]
        Shared left-axis (v_avg) limits, common to every box.
    ylim_d : tuple[float]
        Shared right-axis (D1/D2) limits, common to every box.
    out_path : str
        Destination PNG.
    """
    fig, axes = plt.subplots(
        1, len(temperatures), figsize=(5 * len(temperatures), 4.6),
        squeeze=False)
    for col, t_sub in enumerate(temperatures):
        ax = axes[0][col]
        ax2 = ax.twinx()
        cell = data['per_t'][t_sub]
        ax.errorbar(cell['j_plot'], cell['v_m'], yerr=cell['v_se'],
                    fmt='o-', color='C0', capsize=3,
                    label=r'$v_{\mathrm{avg}}$')
        ax2.errorbar(cell['j_plot'], cell['d1_m'], yerr=cell['d1_se'],
                     fmt='s--', color='C3', capsize=3, label=r'max $D_1$')
        ax2.errorbar(cell['j_plot'], cell['d2_m'], yerr=cell['d2_se'],
                     fmt='^--', color='C2', capsize=3, label=r'min $D_2$')
        ax.set_title(r'$T = %g$ K' % t_sub)
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylim(ylim_v)
        ax2.set_ylim(ylim_d)
        if col == 0:
            ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)', color='C0')
        if col == len(temperatures) - 1:
            ax2.set_ylabel(r'axis (nm)')
        ax.tick_params(axis='y', labelcolor='C0')
        ax.set_box_aspect(1)
    axes[0][0].legend(loc='upper left', frameon=False)
    fig.suptitle(box_tag)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def _plot_class_check(campaign_dir, box_tag, t_sub, out_path):
    """Final top m_z of the ens0 member at each current, T fixed.

    One row per peak current, labelled with the classifier verdict and
    the metrics behind it, so the compact -> elongated -> spanning (P)
    progression can be verified by eye. Included for reference: it
    shows what the spanning class the S41/S44 gate excludes actually
    looks like -- a reversed domain bridging the periodic track.

    Parameters
    ----------
    campaign_dir : str
        Campaign directory for the box.
    box_tag : str
        Box directory name (title only).
    t_sub : float
        Substrate temperature (K).
    out_path : str
        Destination PNG.
    """
    currents = _cell_currents(campaign_dir, t_sub)
    fig, axes = plt.subplots(
        len(currents), 1, figsize=(7.0, 1.5 * len(currents)),
        squeeze=False)
    for row, peak_j in enumerate(currents):
        ax = axes[row][0]
        path = os.path.join(
            campaign_dir, 'T%05.1f_j%.2e_ens000.npz' % (t_sub, peak_j))
        npz = np.load(path, allow_pickle=True)
        config = _decode_config(npz)
        mz = np.asarray(npz['mz_final_top'], dtype=float).reshape(
            int(config['ny']), int(config['nx']))
        code, m = classify_field(
            mz, abs(float(npz['Q'][-1])), float(npz['D1_top'][-1]),
            float(npz['D2_top'][-1]), float(npz['L_x']))
        ax.imshow(mz, origin='lower', cmap='RdBu_r', vmin=-1.0,
                  vmax=1.0, aspect='equal')
        ax.set_ylabel(r'$J=%.1f$' % (peak_j / 1e11), fontsize=13,
                      rotation=0, ha='right', va='center')
        ax.set_title(
            r'%s:  $D_x/L_x=%.2f$,  $D_1/D_2=%.2f$,  sol$=%.2f$'
            % (code, m['dx_lx'], m['ratio'], m['solidity']),
            fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle('%s, T=%g K --- ens0 final top $m_z$ (class check)'
                 % (box_tag, t_sub))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print('Saved %s' % out_path)
# -------------------------------------------------------------------------


def main():
    """Write the S41 and S44 panels for every configured box."""
    plt.rcParams.update({
        'axes.labelsize': 18,
        'axes.titlesize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'legend.title_fontsize': 14,
        'figure.titlesize': 22,
    })
    # =========================== User Configuration =========================
    campaign_root = ('/Volumes/T7/skyrmion_simulator/output/'
                     'stochastic_llgs/scan_track_width/campaign')
    box_tags = ['hk36_D0p72_350x500', 'hk36_D0p72_700x500',
                'hk36_D0p72_1400x500']
    out_dir = 'output/figures_driving_T/campaign_dc'
    temperatures = [10.0, 50.0, 100.0]
    min_surv = 20              # survivors needed to fix the S44_A cell
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load S41, S44A and S44B for every box first, then fix one common
    # y-range per panel type so the figures share a scale across boxes
    # and temperature columns.
    s41, s44a, s44b = {}, {}, {}
    for box_tag in box_tags:
        campaign_dir = os.path.join(campaign_root, box_tag)
        if not os.path.isdir(campaign_dir):
            raise RuntimeError(
                'main: campaign directory not found: %r.'
                % campaign_dir)
        print('=== %s (loading) ===' % box_tag)
        s41[box_tag] = _s41_data(campaign_dir, temperatures)
        s44a[box_tag] = _s44a_data(campaign_dir, temperatures, min_surv)
        s44b[box_tag] = _s44b_data(campaign_dir, temperatures)

    def _padded(lo, hi, floor_zero):
        pad = 0.05 * (hi - lo) if hi > lo else 1.0
        return (0.0 if floor_zero else lo - pad, hi + pad)

    ylim_s41 = (0.0, max(s41[b]['v_max'] for b in box_tags) * 1.05)
    ylim_s44a = _padded(
        min(s44a[b]['lo'] for b in box_tags),
        max(s44a[b]['hi'] for b in box_tags), floor_zero=False)
    ylim_s44b_v = _padded(
        min(s44b[b]['v_lo'] for b in box_tags),
        max(s44b[b]['v_hi'] for b in box_tags), floor_zero=True)
    ylim_s44b_d = _padded(
        min(s44b[b]['d_lo'] for b in box_tags),
        max(s44b[b]['d_hi'] for b in box_tags), floor_zero=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    for box_tag in box_tags:
        campaign_dir = os.path.join(campaign_root, box_tag)
        print('=== %s ===' % box_tag)
        _plot_s41(box_tag, s41[box_tag], temperatures, ylim_s41,
                  os.path.join(out_dir, 'S41_vt_%s.png' % box_tag))
        _plot_s44a(box_tag, s44a[box_tag], temperatures, ylim_s44a,
                   os.path.join(out_dir,
                                'S44_A_D1D2_t_%s.png' % box_tag))
        _plot_s44b(box_tag, s44b[box_tag], temperatures, ylim_s44b_v,
                   ylim_s44b_d,
                   os.path.join(out_dir,
                                'S44_B_vD_J_%s.png' % box_tag))
        _plot_class_check(
            campaign_dir, box_tag, 10.0,
            os.path.join(out_dir, 'class_check_T10_%s.png' % box_tag))


# =============================================================================
if __name__ == '__main__':
    main()
