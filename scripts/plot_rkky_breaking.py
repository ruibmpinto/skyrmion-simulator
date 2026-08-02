"""RKKY interlayer-lock diagnostics from the track-width campaign.

Answers, from data already on disk, whether and where the AFM
interlayer lock of the SAF pair breaks. Three observables:

- Interlayer alignment <m_top . m_bot> over the box. Computed from the
  stored full fields: at the end of the j = 0 thermal equilibration for
  every (T, realization) via `m_thermal_T*_ens*.npz`, and as a per-frame
  time series for the ens0 realization of each (T, j) cell via
  `anim_T*.npz`.
- Centroid separation |r_top - r_bot| of the two layers during the
  drive, from the per-layer tracked centres already written into every
  trajectory file (`cx_unwrapped`/`cx_unwrapped_bot`, ...), so all
  realizations contribute.
- Charge imbalance |Q_top + Q_bot|, which is ~0 for a locked
  antiparallel pair and grows when one layer decouples.

The pair coordinates are only meaningful where a single object exists,
so the drive-phase maps are masked to cells whose ensemble survival
exceeds a floor.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m scripts.plot_rkky_breaking
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.lines as mlines
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _set_style():
    """Enlarge default text sizes for figures reproduced on a page."""
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 15,
        'xtick.labelsize': 13,
        'ytick.labelsize': 13,
        'legend.fontsize': 13,
        'figure.titlesize': 18,
    })


# -----------------------------------------------------------------------------
def _alignment_of_field(path):
    """Box-averaged interlayer alignment of one stored two-layer state.

    Parameters
    ----------
    path : str
        Path to an NPZ holding `m_top` and `m_bot` as (n_frames, ny, nx,
        3) arrays.

    Returns
    -------
    align : numpy.ndarray(1d)
        <m_top . m_bot> averaged over the box, one entry per frame.
    """
    with np.load(path, allow_pickle=True) as d:
        if 'm_top' not in d.files or 'm_bot' not in d.files:
            raise RuntimeError(
                f'_alignment_of_field: {path!r} lacks m_top/m_bot; '
                f'keys={list(d.files)}.')
        m_top = d['m_top']
        m_bot = d['m_bot']
        align = (m_top*m_bot).sum(axis=-1).mean(axis=(-2, -1))
        del m_top, m_bot
    return np.atleast_1d(np.asarray(align, dtype=float))


# -----------------------------------------------------------------------------
def _alignment_vs_t_j0(case_dir, t_tokens, n_sample):
    """Interlayer alignment at the end of the j = 0 equilibration.

    Parameters
    ----------
    case_dir : str
        Campaign case directory.
    t_tokens : list[str]
        Temperature filename tokens, e.g. '010.0'.
    n_sample : int
        Realizations to average per temperature.

    Returns
    -------
    t_values : numpy.ndarray(1d)
        Temperatures (K).
    mean, std : numpy.ndarray(1d)
        Ensemble mean and spread of the alignment.
    counts : numpy.ndarray(1d)
        Realizations used per temperature.
    """
    t_values, mean, std, counts = [], [], [], []
    for tok in t_tokens:
        paths = sorted(glob.glob(
            os.path.join(case_dir, f'm_thermal_T{tok}_ens*.npz')))
        if not paths:
            raise RuntimeError(
                f'_alignment_vs_t_j0: no m_thermal for T={tok} in '
                f'{case_dir!r}.')
        vals = [float(_alignment_of_field(p)[-1])
                for p in paths[:n_sample]]
        t_values.append(float(tok))
        mean.append(float(np.mean(vals)))
        std.append(float(np.std(vals)))
        counts.append(len(vals))
    return (np.array(t_values), np.array(mean), np.array(std),
            np.array(counts))


# -----------------------------------------------------------------------------
def _pair_maps(case_dir, Ts, Js):
    """Drive-phase pair separation and charge imbalance over the grid.

    Reads every trajectory of every cell and reduces the per-layer
    tracked centres to a final-time separation, plus the final charge
    imbalance. Both come from series already stored per realization, so
    no field is re-read.

    Parameters
    ----------
    case_dir : str
        Campaign case directory.
    Ts, Js : numpy.ndarray(1d)
        Grid axes (K, A/m^2).

    Returns
    -------
    sep_mean, sep_std : numpy.ndarray(2d)
        Mean and spread of |r_top - r_bot| at the drive end (nm).
    q_sum : numpy.ndarray(2d)
        Mean |Q_top + Q_bot| at the drive end.
    n_used : numpy.ndarray(2d)
        Realizations with a finite separation per cell.
    """
    shape = (Ts.size, Js.size)
    sep_mean = np.full(shape, np.nan)
    sep_std = np.full(shape, np.nan)
    q_sum = np.full(shape, np.nan)
    n_used = np.zeros(shape, dtype=int)
    for i, T in enumerate(Ts):
        for k, J in enumerate(Js):
            paths = sorted(glob.glob(os.path.join(
                case_dir, f'T{T:05.1f}_j{J:.2e}_ens*.npz')))
            if not paths:
                continue
            sep, qs = [], []
            for p in paths:
                with np.load(p, allow_pickle=True) as z:
                    dx = (np.asarray(z['cx_unwrapped'], float)
                          - np.asarray(z['cx_unwrapped_bot'], float))
                    dy = (np.asarray(z['cy_unwrapped'], float)
                          - np.asarray(z['cy_unwrapped_bot'], float))
                    sep.append(float(np.hypot(dx, dy)[-1]))
                    qs.append(abs(float(z['Q'][-1])
                                  + float(z['Q_bot'][-1])))
            sep = np.asarray(sep)*1e9
            qs = np.asarray(qs)
            good = np.isfinite(sep)
            n_used[i, k] = int(good.sum())
            if good.any():
                sep_mean[i, k] = float(sep[good].mean())
                sep_std[i, k] = float(sep[good].std())
            if np.isfinite(qs).any():
                q_sum[i, k] = float(np.nanmean(qs))
    return sep_mean, sep_std, q_sum, n_used


# -----------------------------------------------------------------------------
def _plot_alignment(series, out_path):
    """Interlayer alignment versus temperature at zero current.

    Parameters
    ----------
    series : list[tuple]
        One (label, t_values, mean, std) per case.
    out_path : str
        Output PNG path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))
    for label, t_values, mean, std in series:
        axes[0].errorbar(t_values, mean, yerr=std, marker='o', ms=5,
                         capsize=3, label=label)
        axes[1].errorbar(t_values, 1.0 + mean, yerr=std, marker='s',
                         ms=5, capsize=3, label=label)
    axes[0].axhline(-1.0, color='k', ls=':', lw=1.0,
                    label='perfect AFM lock')
    axes[0].axhline(0.0, color='0.5', ls='--', lw=1.0,
                    label='fully unlocked')
    axes[0].set_ylim(-1.05, 0.12)
    axes[0].set_ylabel(r'$\langle m_{\rm top}\cdot m_{\rm bot}\rangle$')
    axes[0].set_title('interlayer alignment at $j=0$', fontsize=16)
    axes[1].set_ylabel(r'$1+\langle m_{\rm top}\cdot m_{\rm bot}\rangle$')
    axes[1].set_title('deviation from the locked state', fontsize=16)
    for ax in axes:
        ax.set_xlabel(r'$T_{\rm sub}$ (K)')
        ax.set_box_aspect(1)
        ax.legend(frameon=False, fontsize=12)
    fig.suptitle('RKKY interlayer lock versus temperature', fontsize=18)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_alignment_series(series, out_path):
    """Per-frame interlayer alignment during the drive, ens0.

    Parameters
    ----------
    series : list[tuple]
        One (label, t_ns, align) per plotted cell.
    out_path : str
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(11.2, 5.8))
    cmap = plt.get_cmap('plasma')
    for idx, (label, t_ns, align) in enumerate(series):
        ax.plot(t_ns, align, color=cmap(idx/max(len(series) - 1, 1)),
                label=label, lw=1.4)
    ax.axhline(-1.0, color='k', ls=':', lw=1.0,
               label='perfect AFM lock')
    ax.set_xlabel(r'$t$ (ns)')
    ax.set_ylabel(r'$\langle m_{\rm top}\cdot m_{\rm bot}\rangle$')
    ax.set_title('interlayer alignment during the drive (ens0)',
                 fontsize=16)
    # Legend outside the axes: twelve traces leave no clear interior.
    ax.legend(frameon=False, fontsize=11, ncol=1,
              loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _plot_pair_maps(Ts, Js, sep_mean, q_sum, p_surv, p_min, label,
                    out_path):
    """Separation and charge-imbalance maps over the (T, j) grid.

    Parameters
    ----------
    Ts, Js : numpy.ndarray(1d)
        Grid axes (K, A/m^2).
    sep_mean : numpy.ndarray(2d)
        Mean final |r_top - r_bot| (nm).
    q_sum : numpy.ndarray(2d)
        Mean final |Q_top + Q_bot|.
    p_surv : numpy.ndarray(2d)
        Ensemble survival, used to mask meaningless cells.
    p_min : float
        Survival floor below which the pair coordinate is not defined.
    label : str
        Case label for the title.
    out_path : str
        Output PNG path.
    """
    keep = p_surv >= p_min
    panels = ((np.where(keep, sep_mean, np.nan),
               r'$|r_{\rm top}-r_{\rm bot}|$ (nm)', 'viridis'),
              (np.where(keep, q_sum, np.nan),
               r'$|Q_{\rm top}+Q_{\rm bot}|$', 'magma'))
    fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.0))
    for ax, (z, name, cmap) in zip(axes, panels):
        im = ax.imshow(z, origin='lower', aspect='auto', cmap=cmap)
        ax.set_xticks(np.arange(Js.size))
        ax.set_xticklabels([f'{j*1e-11:.2g}' for j in Js])
        ax.set_yticks(np.arange(Ts.size))
        ax.set_yticklabels([f'{t:.0f}' for t in Ts])
        ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
        ax.set_ylabel('T (K)')
        ax.set_title(name, fontsize=15)
        for i in range(Ts.size):
            for k in range(Js.size):
                v = z[i, k]
                txt = f'{v:.1f}' if name.startswith('$|r') else \
                    (f'{v:.3f}' if np.isfinite(v) else '')
                ax.text(k, i, txt if np.isfinite(v) else '-',
                        ha='center', va='center', fontsize=10,
                        color='w')
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f'interlayer pair coordinates, drive end -- {label} '
                 f'(cells with $P_{{\\rm surv}}\\geq{p_min}$)',
                 fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _separation_series(case_dir, Ts, Js, p_surv, p_min):
    """Ensemble-mean separation time series of the surviving cells.

    Also reports, per cell, the steady offset reached in the second half
    of the drive and a coherence ratio for that window: the largest FFT
    amplitude of the detrended series over its mean amplitude. A
    sustained oscillation gives a ratio of order half the sample count;
    incoherent thermal wander stays at a few.

    Parameters
    ----------
    case_dir : str
        Campaign case directory.
    Ts, Js : numpy.ndarray(1d)
        Grid axes (K, A/m^2).
    p_surv : numpy.ndarray(2d)
        Ensemble survival per cell.
    p_min : float
        Survival floor below which the pair coordinate is undefined.

    Returns
    -------
    curves : list[tuple]
        One (T, J, t_ns, d_mean, offset, ratio) per surviving cell.
    """
    curves = []
    for i, T in enumerate(Ts):
        for k, J in enumerate(Js):
            if not (p_surv[i, k] >= p_min):
                continue
            paths = sorted(glob.glob(os.path.join(
                case_dir, f'T{T:05.1f}_j{J:.2e}_ens*.npz')))
            if not paths:
                continue
            stack, t_s = [], None
            for p in paths:
                with np.load(p, allow_pickle=True) as z:
                    dx = (np.asarray(z['cx_unwrapped'], float)
                          - np.asarray(z['cx_unwrapped_bot'], float))
                    dy = (np.asarray(z['cy_unwrapped'], float)
                          - np.asarray(z['cy_unwrapped_bot'], float))
                    stack.append(np.hypot(dx, dy)*1e9)
                    if t_s is None:
                        t_s = np.asarray(z['t_sample'], float)
            d_mean = np.nanmean(np.stack(stack), axis=0)
            half = d_mean[d_mean.size//2:]
            offset = float(np.nanmean(half))
            idx = np.arange(half.size)
            trend = np.polyval(np.polyfit(idx, half, 1), idx)
            spec = np.abs(np.fft.rfft(half - trend))[1:]
            ratio = (float(spec.max()/spec.mean())
                     if spec.size and spec.mean() > 0 else np.nan)
            curves.append((float(T), float(J), t_s*1e9, d_mean, offset,
                           ratio))
    return curves


# -----------------------------------------------------------------------------
def _plot_separation(curves, label, out_path):
    """Separation time series and its steady offset versus current.

    Parameters
    ----------
    curves : list[tuple]
        Output of `_separation_series`.
    label : str
        Case label for the title.
    out_path : str
        Output PNG path.
    """
    t_values = sorted({c[0] for c in curves})
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.4))
    cmap = plt.get_cmap('plasma')
    for T, J, t_ns, d_mean, offset, ratio in curves:
        col = cmap(t_values.index(T)/max(len(t_values) - 1, 1))
        axes[0].plot(t_ns, d_mean, color=col, lw=1.2)
    for idx, T in enumerate(t_values):
        sel = [(c[1], c[4]) for c in curves if c[0] == T]
        sel.sort()
        axes[1].plot([s[0]*1e-11 for s in sel], [s[1] for s in sel],
                     marker='o', ms=5,
                     color=cmap(idx/max(len(t_values) - 1, 1)),
                     label=f'T={T:.0f} K')
    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$d = |r_{\rm top}-r_{\rm bot}|$ (nm)')
    axes[0].set_title('separation during the drive (colour = $T$)',
                      fontsize=15)
    axes[1].set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    axes[1].set_ylabel(r'steady $d$ (nm)')
    axes[1].set_title('steady offset versus current', fontsize=15)
    axes[1].legend(frameon=False, fontsize=12)
    for ax in axes:
        ax.set_box_aspect(1)
    fig.suptitle(f'interlayer separation -- {label}', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    plt.close(fig)


# -----------------------------------------------------------------------------
def _stretch_table(curves, v_mean, v_se, Ts, Js):
    """Pair the steady interlayer offset with the drift speed per cell.

    Parameters
    ----------
    curves : list[tuple]
        Output of `_separation_series`.
    v_mean, v_se : numpy.ndarray(2d)
        Ensemble drift speed and its standard error (m/s).
    Ts, Js : numpy.ndarray(1d)
        Grid axes (K, A/m^2).

    Returns
    -------
    rows : list[dict]
        Keys T, J, d, v, v_se, lag (ps), for cells with a finite speed.
    """
    rows = []
    for T, J, _t, _d, offset, ratio in curves:
        i = int(np.argmin(np.abs(Ts - T)))
        k = int(np.argmin(np.abs(Js - J)))
        v = float(v_mean[i, k])
        if not np.isfinite(v) or v <= 0.0:
            continue
        rows.append({'T': T, 'J': J, 'd': offset, 'v': v,
                     'v_se': float(v_se[i, k]),
                     'lag': offset*1e-9/v*1e12, 'ratio': ratio})
    return rows


# -----------------------------------------------------------------------------
def _plot_stretch(tables, labels, styles, out_path):
    """Current-driven interlayer stretch across boxes.

    Left: steady offset versus current, one curve per temperature, one
    line style per box. Centre: the same offsets against the drift
    speed, where a spring dragged at constant velocity collapses onto a
    single line d = v tau. Right: the implied lag tau = d / v.

    Parameters
    ----------
    tables : list[list[dict]]
        One `_stretch_table` output per box.
    labels : list[str]
        Box labels, one per table.
    styles : list[str]
        Line styles, one per box.
    out_path : str
        Output PNG path.
    """
    t_values = sorted({r['T'] for tab in tables for r in tab})
    cmap = plt.get_cmap('plasma')
    fig, axes = plt.subplots(1, 3, figsize=(17.4, 5.4))
    for tab, lab, st in zip(tables, labels, styles):
        for idx, T in enumerate(t_values):
            sel = sorted((r for r in tab if r['T'] == T),
                         key=lambda r: r['J'])
            if not sel:
                continue
            col = cmap(idx/max(len(t_values) - 1, 1))
            axes[0].plot([r['J']*1e-11 for r in sel],
                         [r['d'] for r in sel], st, color=col,
                         marker='o', ms=5,
                         label=(f'T={T:.0f} K' if st == '-' else None))
            axes[1].plot([r['v'] for r in sel], [r['d'] for r in sel],
                         st, color=col, marker='s', ms=5)
            axes[2].plot([r['J']*1e-11 for r in sel],
                         [r['lag'] for r in sel], st, color=col,
                         marker='^', ms=5)
    # Reference: proportional drag, fitted through the origin over all
    # boxes and temperatures.
    v_all = np.array([r['v'] for tab in tables for r in tab])
    d_all = np.array([r['d'] for tab in tables for r in tab])
    tau = float((v_all*d_all).sum()/(v_all*v_all).sum())
    v_line = np.linspace(0.0, v_all.max()*1.05, 50)
    axes[1].plot(v_line, tau*v_line, color='k', ls=':', lw=1.4,
                 label=rf'$d=v\tau$, $\tau={tau*1e-9*1e12:.1f}$ ps')
    axes[0].set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    axes[0].set_ylabel(r'steady $d$ (nm)')
    axes[0].set_title('interlayer stretch vs current', fontsize=15)
    axes[0].legend(frameon=False, fontsize=11)
    axes[1].set_xlabel(r'$v$ (m/s)')
    axes[1].set_ylabel(r'steady $d$ (nm)')
    axes[1].set_title('stretch vs drift speed', fontsize=15)
    axes[1].legend(frameon=False, fontsize=11)
    axes[2].set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    axes[2].set_ylabel(r'$\tau = d/v$ (ps)')
    axes[2].set_title('implied interlayer lag', fontsize=15)
    for ax in axes:
        ax.set_box_aspect(1)
    box_handles = [mlines.Line2D([], [], color='0.3', ls=st,
                                 label=lab)
                   for lab, st in zip(labels, styles)]
    axes[2].legend(handles=box_handles, frameon=False, fontsize=11,
                   loc='upper left', title='track length')
    fig.suptitle('current-driven interlayer stretch', fontsize=17)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, dpi=160)
    print(f'Saved: {out_path}')
    print(f'   global proportional-drag fit: tau = '
          f'{tau*1e-9*1e12:.2f} ps')
    plt.close(fig)


# -----------------------------------------------------------------------------
def main():
    """Assess the RKKY interlayer lock from the stored campaign data.

    Config is variables at the top of main() (no argparse).
    """
    # =========================== User Configuration =========================
    campaign_root = ('/Volumes/T7/skyrmion_simulator/output/'
                     'stochastic_llgs/scan_track_width/campaign')
    out_dir = 'output/figures_sllg/track_width/rkky_breaking'
    tags = ['hk36_D0p72_350x500',
            'hk36_D0p72_700x500',
            'hk36_D0p72_1400x500']
    labels = [r'$L_x=700$ nm', r'$L_x=1400$ nm', r'$L_x=2800$ nm']
    styles = ['-', '--', ':']
    t_tokens = ['010.0', '050.0', '100.0', '130.0', '160.0', '200.0']
    # The alignment spread across realizations is ~3e-4, so a modest
    # sample resolves the mean; the fields are multi-GB per box.
    n_sample = 25              # realizations per T for the j=0 average
    p_min = 0.5                # survival floor for the pair maps
    series_tag = 'hk36_D0p72_350x500'
    series_js = ['5.00e+10', '5.00e+11']
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    _set_style()
    align_series = []
    for tag, label in zip(tags, labels):
        case_dir = os.path.join(campaign_root, tag)
        t_values, mean, std, counts = _alignment_vs_t_j0(
            case_dir, t_tokens, n_sample)
        align_series.append((label, t_values, mean, std))
        print(f'== {tag}: alignment at j=0 ==')
        for T, m, s, n in zip(t_values, mean, std, counts):
            print(f'   T={T:5.0f} K  <m.m> = {m:+.5f} +- {s:.5f}  '
                  f'(n={n})  1+<m.m> = {1.0 + m:.5f}')
    _plot_alignment(align_series,
                    os.path.join(out_dir, 'alignment_vs_T.png'))
    # Per-frame alignment during the drive, ens0 of selected cells.
    case_dir = os.path.join(campaign_root, series_tag)
    drive_series = []
    for tok in t_tokens:
        for j_tok in series_js:
            path = os.path.join(case_dir, f'anim_T{tok}_j{j_tok}.npz')
            if not os.path.isfile(path):
                print(f'   no dump for T={tok} j={j_tok}; skipped')
                continue
            with np.load(path, allow_pickle=True) as d:
                phase = np.asarray(d['phase_id'])
                t_s = np.asarray(d['time_s'], float)
            drive_idx = np.flatnonzero(phase == 1)
            align = _alignment_of_field(path)[drive_idx]
            t_ns = (t_s[drive_idx] - t_s[drive_idx][0])*1e9
            j_val = float(j_tok)
            drive_series.append(
                (f'T={float(tok):.0f} K, J={j_val*1e-11:.2g}',
                 t_ns, align))
            print(f'   drive T={float(tok):5.0f} K J={j_val:.1e}: '
                  f'<m.m> {align[0]:+.4f} -> {align[-1]:+.4f}')
    if drive_series:
        _plot_alignment_series(
            drive_series, os.path.join(out_dir, 'alignment_vs_t_drive.png'))
    # Pair separation / charge imbalance over the whole grid.
    tables = []
    for tag, label in zip(tags, labels):
        case_dir = os.path.join(campaign_root, tag)
        agg = np.load(os.path.join(case_dir, 'aggregate.npz'))
        Ts = np.asarray(agg['Ts'], float)
        Js = np.asarray(agg['Js'], float)
        sep_mean, sep_std, q_sum, n_used = _pair_maps(case_dir, Ts, Js)
        print(f'== {tag}: pair coordinates at drive end ==')
        for i, T in enumerate(Ts):
            row = '  '.join(
                f'{sep_mean[i, k]:7.1f}' if np.isfinite(sep_mean[i, k])
                else '    nan' for k in range(Js.size))
            print(f'   T={T:5.0f} K sep(nm): {row}')
        p_surv = np.asarray(agg['P_surv'], float)
        _plot_pair_maps(
            Ts, Js, sep_mean, q_sum, p_surv, p_min, label,
            os.path.join(out_dir, f'pair_maps_{tag}.png'))
        curves = _separation_series(case_dir, Ts, Js, p_surv, p_min)
        rows = _stretch_table(curves, np.asarray(agg['v_mean'], float),
                              np.asarray(agg['v_se'], float), Ts, Js)
        tables.append(rows)
        print(f'== {tag}: stretch (surviving cells) ==')
        print('     T     J   steady d (nm)    v (m/s)   tau (ps)'
              '   FFT peak/mean')
        for r in rows:
            print(f'   {r["T"]:5.0f} {r["J"]*1e-11:5.2g}   '
                  f'{r["d"]:8.2f}     {r["v"]:8.1f}   {r["lag"]:7.2f}'
                  f'      {r["ratio"]:6.2f}')
        _plot_separation(
            curves, label,
            os.path.join(out_dir, f'separation_{tag}.png'))
    _plot_stretch(tables, labels, styles,
                  os.path.join(out_dir, 'stretch_vs_current.png'))


# =============================================================================
if __name__ == '__main__':
    main()
