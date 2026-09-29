"""Stop-and-go DC drive postprocessing on the 700x500 SAF racetrack.

The stop-and-go run drives the track at a fixed current in three 1-ns
ON phases separated by 1-ns OFF phases over a 5-ns window
(ON [0,1], OFF [1,2], ON [2,3], OFF [3,4], ON [4,5]). This script
renders three views of that response:

- interlayer alignment <m_top . m_bot>(t) for the ens0 realization, at
  the lowest and highest current of each temperature;
- ensemble-mean drift speed <|v|>(t), one trace per current, per
  temperature panel;
- ens0 field snapshots at each phase boundary, to visualize the
  compact-to-elongated-and-back cycle.

Functions
---------
main
    Load the stop-and-go members / anim dumps and emit the figures.

Notes
-----
Alignment and snapshots use the ens0 anim dumps (only ens0 stores the
full two-layer field). The speed waveform averages over the surviving
members (final field classified S or E) of each cell. The on/off
boundary times are fixed by the driver schedule (1, 2, 3, 4 ns).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.stochastic_llgs.stability import classify_field
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _drive_alignment(anim_path):
    """Interlayer alignment <m_top . m_bot>(t) over the drive window.

    Mirrors `plot_rkky_breaking._alignment_of_field`: dot the two vector
    fields per cell and box-average, one scalar per frame. Restricted to
    the drive frames (phase_id == 1) and zeroed to the drive start.

    Parameters
    ----------
    anim_path : str
        ens0 anim npz with keys m_top, m_bot (nf, ny, nx, 3), time_s,
        phase_id.

    Returns
    -------
    t_ns : numpy.ndarray(1d)
        Time since drive start (ns).
    align : numpy.ndarray(1d)
        Box-averaged <m_top . m_bot> per drive frame.
    """
    d = np.load(anim_path)
    m_top = np.asarray(d['m_top'], dtype=float)
    m_bot = np.asarray(d['m_bot'], dtype=float)
    phase = np.asarray(d['phase_id'])
    time_s = np.asarray(d['time_s'], dtype=float)
    drive = np.flatnonzero(phase == 1)
    if drive.size == 0:
        raise RuntimeError(f'_drive_alignment: no drive frames in '
                           f'{anim_path!r}.')
    align = (m_top * m_bot).sum(axis=-1).mean(axis=(-2, -1))
    t_ns = (time_s[drive] - time_s[drive][0]) * 1e9
    return t_ns, align[drive]
# -----------------------------------------------------------------------------
def _speed_of_member(npz):
    """Centroid speed magnitude per frame from an unwrapped track.

    Mirrors `plot_pulse_inertia._speed`: centred difference of the
    unwrapped centroid.

    Parameters
    ----------
    npz : numpy.lib.npyio.NpzFile
        Member npz with t_sample, cx_unwrapped, cy_unwrapped.

    Returns
    -------
    t_ns : numpy.ndarray(1d)
        Sample times (ns).
    v : numpy.ndarray(1d)
        Speed magnitude (m/s).
    """
    t = np.asarray(npz['t_sample'], dtype=float)
    cx = np.asarray(npz['cx_unwrapped'], dtype=float)
    cy = np.asarray(npz['cy_unwrapped'], dtype=float)
    v = np.hypot(np.gradient(cx, t), np.gradient(cy, t))
    return t * 1e9, v
# -----------------------------------------------------------------------------
def _speed_waveform(root, temp, j_value):
    """Survivor-averaged <|v|>(t) for one (T, J) cell.

    Loads every ensemble member, keeps those whose final field
    classifies as a skyrmion (S or E), and averages the per-frame speed
    over members sharing the common sample grid.

    Parameters
    ----------
    root : str
        Directory of the stop-and-go members.
    temp : float
        Temperature (K).
    j_value : float
        Current density (A/m^2).

    Returns
    -------
    t_ns : {numpy.ndarray(1d), None}
        Sample times (ns), or None if no survivors.
    mean : {numpy.ndarray(1d), None}
        Ensemble-mean speed (m/s), or None.
    n_surv : int
        Number of surviving members averaged.
    """
    files = sorted(glob.glob(
        f'{root}/T{temp:05.1f}_j{j_value:.2e}_ens*.npz'))
    t_ref = None
    stack = []
    for path in files:
        d = np.load(path)
        q = abs(float(np.asarray(d['Q'])[-1]))
        d1 = d['D1_top']
        d2 = d['D2_top']
        half = max(1, d1.size // 2)
        code, _ = classify_field(
            np.asarray(d['mz_final_top'], dtype=float), q,
            float(np.nanmean(d1[half:])), float(np.nanmean(d2[half:])),
            float(d['L_x']))
        if code not in ('S', 'E'):
            continue
        t_ns, v = _speed_of_member(d)
        if t_ref is None:
            t_ref = t_ns
        if v.shape == t_ref.shape:
            stack.append(v)
    if not stack:
        return None, None, 0
    return t_ref, np.mean(np.vstack(stack), axis=0), len(stack)
# -----------------------------------------------------------------------------
def _draw_boundaries(ax, boundaries_ns):
    """Draw the dotted on/off phase boundaries on `ax`."""
    for tb in boundaries_ns:
        ax.axvline(tb, color='0.7', lw=0.8, ls=':')
# -----------------------------------------------------------------------------
def _plot_alignment(cells, boundaries_ns, out_path):
    """Alignment-vs-time traces for the ens0 drive.

    Parameters
    ----------
    cells : list[tuple]
        (label, t_ns, align) per (T, J) trace.
    boundaries_ns : list[float]
        Phase-boundary times (ns).
    out_path : str
        Output PNG path.
    """
    fig, ax = plt.subplots(figsize=(9.0, 5.0))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(cells)))
    for c, (label, t_ns, align) in zip(colors, cells):
        ax.plot(t_ns, align, color=c, lw=1.4, label=label)
    ax.axhline(-1.0, color='k', lw=0.8, ls='--')
    _draw_boundaries(ax, boundaries_ns)
    ax.set_xlabel(r'$t$ (ns)', fontsize=13)
    ax.set_ylabel(r'$\langle m_{\mathrm{top}}\cdot m_{\mathrm{bot}}'
                  r'\rangle$', fontsize=13)
    ax.legend(fontsize=8, loc='upper left', bbox_to_anchor=(1.02, 1.0))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
# -----------------------------------------------------------------------------
def _plot_speed(root, temps, js, boundaries_ns, out_path):
    """Per-temperature panels of survivor-averaged <|v|>(t).

    Parameters
    ----------
    root : str
        Stop-and-go member directory.
    temps : list[float]
        Temperatures (K), one panel each.
    js : list[float]
        Currents (A/m^2), one line each.
    boundaries_ns : list[float]
        Phase-boundary times (ns).
    out_path : str
        Output PNG path.
    """
    n = len(temps)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol,
                                                  3.8 * nrow),
                             squeeze=False)
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(js)))
    for idx, temp in enumerate(temps):
        ax = axes[idx // ncol][idx % ncol]
        for c, j_value in zip(colors, js):
            t_ns, mean, n_surv = _speed_waveform(root, temp, j_value)
            if t_ns is None:
                continue
            ax.plot(t_ns, mean, color=c, lw=1.3,
                    label=rf'$J={j_value*1e-11:.2g}$ ($n={n_surv}$)')
        _draw_boundaries(ax, boundaries_ns)
        ax.set_title(rf'$T={temp:.0f}$ K', fontsize=12)
        ax.set_xlabel(r'$t$ (ns)', fontsize=12)
        ax.set_ylabel(r'$\langle |v| \rangle$ (m s$^{-1}$)',
                      fontsize=12)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7, framealpha=0.9)
    for idx in range(n, nrow * ncol):
        axes[idx // ncol][idx % ncol].axis('off')
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
# -----------------------------------------------------------------------------
def _plot_snapshots(anim_path, times_ns, out_path, title):
    """ens0 top-layer m_z snapshots at the requested drive times.

    Parameters
    ----------
    anim_path : str
        ens0 anim npz (m_top, time_s, phase_id).
    times_ns : list[float]
        Times since drive start (ns) to snapshot.
    out_path : str
        Output PNG path.
    title : str
        Figure suptitle (empty string for none).
    """
    d = np.load(anim_path)
    m_top = np.asarray(d['m_top'], dtype=float)
    phase = np.asarray(d['phase_id'])
    time_s = np.asarray(d['time_s'], dtype=float)
    drive = np.flatnonzero(phase == 1)
    t_ns = (time_s[drive] - time_s[drive][0]) * 1e9
    fig, axes = plt.subplots(1, len(times_ns),
                             figsize=(3.0 * len(times_ns), 3.2))
    for ax, tt in zip(axes, times_ns):
        k = drive[int(np.clip(np.searchsorted(t_ns, tt, side='left'),
                              0, t_ns.size - 1))]
        ax.imshow(m_top[k, :, :, 2], origin='lower', aspect='auto',
                  cmap='RdBu_r', vmin=-1.0, vmax=1.0)
        ax.set_title(rf'$t={tt:.0f}$ ns', fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
# -----------------------------------------------------------------------------
def _representative_snapshots(root, agg, temps, js, snap_times_ns,
                              out_subdir):
    """Render ens0 snapshot rows where ens0 matches the dominant class.

    For each (T, J) the ensemble-dominant class is the argmax of the
    class counts (S, E, L, A). Cells whose dominant class is annihilated
    (A) are skipped (no meaningful field). If the ens0 member's own
    class equals the dominant class, its anim evolution is
    representative and a t=0..5 snapshot row is written; otherwise the
    cell is skipped and reported.

    Parameters
    ----------
    root : str
        Stop-and-go directory (members + ens0 anim dumps).
    agg : dict
        aggregate.npz payload (Ts, Js, n_S, n_E, n_L, n_A).
    temps : list[float]
        Temperatures (K).
    js : list[float]
        Currents (A/m^2).
    snap_times_ns : list[float]
        Times since drive start (ns).
    out_subdir : str
        Directory for the per-cell snapshot rows.
    """
    os.makedirs(out_subdir, exist_ok=True)
    ts = np.asarray(agg['Ts'], dtype=float)
    js_arr = np.asarray(agg['Js'], dtype=float)
    order = ('S', 'E', 'L', 'A')
    counts = {c: np.asarray(agg[f'n_{c}'], dtype=float) for c in order}
    rendered = []
    skipped = []
    for temp in temps:
        i = int(np.argmin(np.abs(ts - temp)))
        for j_value in js:
            k = int(np.argmin(np.abs(js_arr - j_value)))
            dom = order[int(np.argmax([counts[c][i, k]
                                       for c in order]))]
            if dom == 'A':
                skipped.append((temp, j_value, '-', 'A'))
                continue
            member = f'{root}/T{temp:05.1f}_j{j_value:.2e}_ens000.npz'
            if not os.path.isfile(member):
                raise RuntimeError(f'_representative_snapshots: no '
                                   f'ens0 member {member!r}.')
            d = np.load(member)
            q = abs(float(np.asarray(d['Q'])[-1]))
            d1 = d['D1_top']
            d2 = d['D2_top']
            half = max(1, d1.size // 2)
            code, _ = classify_field(
                np.asarray(d['mz_final_top'], dtype=float), q,
                float(np.nanmean(d1[half:])),
                float(np.nanmean(d2[half:])), float(d['L_x']))
            if code != dom:
                skipped.append((temp, j_value, code, dom))
                continue
            anim = f'{root}/anim_T{temp:05.1f}_j{j_value:.2e}.npz'
            if not os.path.isfile(anim):
                raise RuntimeError(f'_representative_snapshots: no '
                                   f'anim {anim!r}.')
            out = os.path.join(
                out_subdir, f'T{temp:05.1f}_j{j_value:.2e}_{code}.png')
            title = (rf'$T={temp:.0f}$ K, '
                     rf'$J={j_value*1e-11:.2g}\times10^{{11}}$ -- '
                     rf'{code} (dominant)')
            _plot_snapshots(anim, snap_times_ns, out, title)
            rendered.append((temp, j_value, code))
    print(f'representative snapshots: {len(rendered)} rendered, '
          f'{len(skipped)} skipped.')
    for temp, j_value, code, dom in skipped:
        print(f'  skip T={temp:.0f} J={j_value*1e-11:.2g}: '
              f'ens0={code} dominant={dom}')
# -----------------------------------------------------------------------------
def _plot_schedule(on_intervals, total_ns, out_path):
    """Schematic of the stop-and-go current loading cycle.

    Draws the current as a square wave: unit amplitude over the ON
    intervals and zero over the OFF intervals, with the phases labelled.

    Parameters
    ----------
    on_intervals : list[tuple]
        (start, end) in ns of each drive-ON phase.
    total_ns : float
        Total window (ns).
    out_path : str
        Output PNG path.
    """
    tt = np.linspace(0.0, total_ns, 5000)
    yy = np.zeros_like(tt)
    for a, b in on_intervals:
        yy[(tt >= a) & (tt <= b)] = 1.0
    fig, ax = plt.subplots(figsize=(8.0, 2.6))
    ax.fill_between(tt, 0.0, yy, color='#2c7bb6', alpha=0.2)
    ax.plot(tt, yy, color='#2c7bb6', lw=1.8)
    for a, b in on_intervals:
        ax.text(0.5 * (a + b), 1.12, 'ON', ha='center', va='center',
                color='#2c7bb6', fontsize=11)
    off = [(on_intervals[i][1], on_intervals[i + 1][0])
           for i in range(len(on_intervals) - 1)]
    for a, b in off:
        ax.text(0.5 * (a + b), 0.12, 'OFF', ha='center', va='center',
                color='0.4', fontsize=11)
    ax.set_xlim(0.0, total_ns)
    ax.set_ylim(-0.1, 1.35)
    ax.set_yticks([0.0, 1.0])
    ax.set_yticklabels(['0', r'$J_0$'])
    ax.set_xlabel(r'$t$ (ns)', fontsize=13)
    ax.set_ylabel(r'$J$', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
# -----------------------------------------------------------------------------
def main():
    """Emit the stop-and-go figures (alignment, speed, snapshots)."""
    # =========================== User Configuration =========================
    root = ('/Volumes/T7/skyrmion_simulator/output/stochastic_llgs/'
            'scan_track_width/stop_and_go/hk36_D0p72_700x500')
    out_dir = 'output/figures_sllg/track_width/hk36_D0p72_700x500'
    temps = [10.0, 50.0, 100.0, 130.0, 160.0, 200.0]
    speed_temps = [10.0, 50.0, 100.0]   # higher T leave no survivors
    js = [5.0e10, 1.0e11, 2.0e11, 3.0e11, 4.0e11, 5.0e11]
    boundaries_ns = [1.0, 2.0, 3.0, 4.0]
    align_js = [5.0e10, 5.0e11]     # lowest and highest current
    snap_temp = 10.0                # cold, strong-drive cycling cell
    snap_j = 5.0e11
    snap_times_ns = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]
    fig_dir = os.path.join(out_dir, 'stop_and_go')
    on_intervals = [(0.0, 1.0), (2.0, 3.0), (4.0, 5.0)]
    total_ns = 5.0
    do_schedule = False
    do_alignment = False
    do_speed = False
    do_single_snapshot = False
    do_representative = False
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Loading-cycle schematic (no data needed).
    if do_schedule:
        _plot_schedule(on_intervals, total_ns,
                       os.path.join(fig_dir, 'stop_and_go_schedule.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 1: ens0 alignment at lowest / highest current per T.
    if do_alignment:
        cells = []
        for temp in temps:
            for j_value in align_js:
                anim = f'{root}/anim_T{temp:05.1f}_j{j_value:.2e}.npz'
                if not os.path.isfile(anim):
                    raise RuntimeError(f'main: missing anim {anim!r}.')
                t_ns, align = _drive_alignment(anim)
                cells.append(
                    (rf'$T={temp:.0f}$ K, $J={j_value*1e-11:.2g}$',
                     t_ns, align))
        _plot_alignment(
            cells, boundaries_ns,
            os.path.join(fig_dir, 'stop_and_go_alignment.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 2: survivor-averaged speed waveform per temperature.
    if do_speed:
        _plot_speed(root, speed_temps, js, boundaries_ns,
                    os.path.join(fig_dir, 'stop_and_go_speed.png'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure 4: ens0 snapshots through the on/off cycle.
    if do_single_snapshot:
        snap_anim = f'{root}/anim_T{snap_temp:05.1f}_j{snap_j:.2e}.npz'
        if not os.path.isfile(snap_anim):
            raise RuntimeError(f'main: missing anim {snap_anim!r}.')
        _plot_snapshots(
            snap_anim, snap_times_ns,
            os.path.join(fig_dir, 'stop_and_go_snapshots.png'),
            rf'$T={snap_temp:.0f}$ K, $J={snap_j*1e-11:.2g}$')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-(T,J) representative rows (ens0 == ensemble-dominant class).
    if do_representative:
        agg = dict(np.load(os.path.join(root, 'aggregate.npz')))
        _representative_snapshots(
            root, agg, temps, js, snap_times_ns, fig_dir)
# =============================================================================
if __name__ == '__main__':
    main()
