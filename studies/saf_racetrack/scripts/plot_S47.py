"""Plotter for Pham et al. (2024) Figure S47.

Reads `output/sweeps_S41_S49/S47/cfg<i>_J_<J0>.npz` and emits:

S47_A_v.png        v_avg vs J, one line per config.
S47_B_psi.png      psi_bot (at max diameter) vs J, per config.
S47_C_D1.png       D1_top at the pulse maximum vs J, per config.
S47_D_D2.png       D2_top at the pulse maximum vs J, per config.
S47_E_snap.png     m_z snapshot at the pulse peak, config 0.
S47_F_snap.png     m_z snapshot at the pulse peak, config 1.
S47_G_snap.png     m_z snapshot at the pulse peak, config 2.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
from collections import defaultdict
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from studies.saf_racetrack.plots.plot_snapshot_mz import plot_snapshot_mz
from skyrmion_simulator.simulator.parameters import default_params
from skyrmion_simulator.stochastic_llgs.diagnostics import unwrap_trajectory
from studies.saf_racetrack.orchestrator.io import load_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults.
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


def _vavg(trace, metadata):
    """Net displacement over the +/- 3 sigma window divided by the
    FWHM (m/s)."""
    # Time stamps and box dimensions from metadata.
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    # PBC unwrap of the top-layer centroid stream.
    cx, cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
        periodic_y=True)
    # Identify the +/- tail_sigmas * sigma window around t_center.
    FWHM = float(metadata['FWHM'])
    tail = (float(metadata['tail_sigmas'])
            * float(metadata['sigma']))
    t_lo = float(metadata['t_center']) - tail
    t_hi = float(metadata['t_center']) + tail
    i_lo = int(np.argmin(np.abs(t - t_lo)))
    i_hi = int(np.argmin(np.abs(t - t_hi)))
    # delta(x,y) over the window, normalised by FWHM.
    dx = cx[i_hi] - cx[i_lo]
    dy = cy[i_hi] - cy[i_lo]
    return float(np.sqrt(dx * dx + dy * dy) / FWHM)


def _load_sweep(in_dir):
    """Load all NPZ traces bucketed by config index, J0-sorted."""
    # All NPZ traces in the sweep directory.
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
    # Bucket by configuration index for downstream grouping.
    bucket = defaultdict(list)
    for path in paths:
        trace, metadata = load_trace(path)
        bucket[int(metadata['cfg_idx'])].append(
            (metadata, trace))
    # Sort each bucket by J0.
    for cfg in bucket:
        bucket[cfg].sort(key=lambda mt: float(mt[0]['J0']))
    return bucket


def _config_label(metadata):
    """Legend label for one configuration."""
    # Short label encoding the config's FWHM and H_RKKY.
    return (f'cfg {int(metadata["cfg_idx"])}: '
            f'FWHM={float(metadata["FWHM"])*1e12:.0f} ps, '
            f'$H_{{\\mathrm{{RKKY}}}}$='
            f'{float(metadata["H_RKKY"])*1e3:.0f} mT')


def _plot_v(bucket, out_path):
    """Panel A: v_avg vs J, one curve per config."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    # One curve per configuration index.
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        # Stack J (A/m^2) and v_avg (m/s) for this cfg.
        J_arr = np.array([float(m['J0']) for m, _ in items])
        v_arr = np.array([_vavg(t, m) for m, t in items])
        # Colour by config's position in the sorted set.
        c = cmap(k / max(len(bucket) - 1, 1))
        ax.plot(J_arr / 1e11, v_arr, 'o-', color=c,
                label=_config_label(items[0][0]))
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)')
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _unwrap_to_zero_rest(psi_seq):
    """Convert a psi(J) series stored under the OLD dw_angle
    convention into the new signed-deviation convention via
    np.unwrap + small-J rest-value reduction mod pi."""
    psi = np.unwrap(np.asarray(psi_seq))
    rest = ((psi[0] + np.pi / 2.0) % np.pi) - np.pi / 2.0
    return psi - (psi[0] - rest)


# Panel B: psi_bot at max diameter vs J, one curve per config.
def _plot_psi(bucket, out_path):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    # One curve per configuration; psi sampled at d_top peak.
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        J_arr = np.array([float(m['J0']) for m, _ in items])
        # psi_bot at the time of peak top-layer diameter.
        psi_arr = []
        for _m, t in items:
            i_peak = int(np.argmax(t['d_top']))
            psi_arr.append(float(t['psi_bot'][i_peak]))
        # Sort by J so np.unwrap operates on a monotone series.
        order = np.argsort(J_arr)
        J_sorted = J_arr[order]
        psi_sorted = np.degrees(_unwrap_to_zero_rest(
            np.array(psi_arr)[order]))
        c = cmap(k / max(len(bucket) - 1, 1))
        ax.plot(J_sorted / 1e11, psi_sorted, 'o-', color=c,
                label=_config_label(items[0][0]))
    # Zero reference (natural Neel orientation).
    ax.axhline(0.0, color='0.7', linestyle=':')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$\psi_{\mathrm{bot}}$ (deg, signed deviation)')
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_axes(bucket, axis_key, ylabel, title, out_path,
               check_box_limit):
    """Generic plot for D1/D2 sampled at the pulse maximum
    (paper S47 caption convention). With `check_box_limit`,
    warn per trace when max D1 over the whole trace exceeds
    0.8x the box extent (post-pulse transient wrapping the
    periodic box; the value is box geometry, not a diameter)."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    # One curve per configuration.
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        J_arr = np.array([float(m['J0']) for m, _ in items])
        # Sample the axis at the pulse maximum (t_center).
        vals = []
        for m, t in items:
            i_pk = int(np.argmin(
                np.abs(t['t'] - float(m['t_center']))))
            vals.append(float(t[axis_key][i_pk]))
            if check_box_limit:
                box = (min(int(m['nx']), int(m['ny']))
                       * float(default_params().a))
                d_mx = float(np.max(t[axis_key]))
                if d_mx > 0.8 * box:
                    print(f'  WARN box-limited: cfg '
                          f'{int(m["cfg_idx"])}, '
                          f'J={float(m["J0"]):.2e}: max '
                          f'{axis_key} = {d_mx*1e9:.0f} nm '
                          f'> 0.8 x box ({box*1e9:.0f} nm)')
        vals = np.array(vals)
        c = cmap(k / max(len(bucket) - 1, 1))
        # Convert m -> nm for the axis-size plot.
        ax.plot(J_arr / 1e11, vals * 1e9, 'o-', color=c,
                label=_config_label(items[0][0]))
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_snapshots(bucket, out_dir):
    """Render snapshot panels E-G for each configuration."""
    for cfg_idx, suffix in zip([0, 1, 2], ['E', 'F', 'G']):
        if cfg_idx not in bucket:
            print(f'Skipping snapshot {suffix}: no '
                  f'config {cfg_idx} sweep.')
            continue
        # The snapshot was requested at J_max for each cfg.
        items = bucket[cfg_idx]
        # The trace with a snapshot is the largest-J one whose
        # snapshot_t is not None.
        snap_item = None
        for m, t in reversed(items):
            if t.get('snapshot_t') is not None:
                snap_item = (m, t)
                break
        if snap_item is None:
            print(f'Skipping snapshot {suffix}: no recorded '
                  f'spin field for cfg {cfg_idx}.')
            continue
        m, t = snap_item
        # The simulator's lattice spacing is not stored in the
        # trace: every sweep uses Set A's default `p.a`, so it is
        # recovered from default_params without rerunning the
        # simulation.
        a_default = float(default_params().a)
        m_top = t['snapshot_m_top']
        out_path = os.path.join(
            out_dir, f'S47_{suffix}_snap.png')
        fig, ax = plt.subplots()
        plot_snapshot_mz(
            m=m_top,
            a=a_default,
            ax=ax,
            title=_config_label(m),
            show_colorbar=True)
        ax.set_box_aspect(1)
        fig.tight_layout()
        fig.savefig(out_path)
        print(f'Saved {out_path}')


def main():
    """Load the S47 sweep and render all panels: A-D curves plus the
    E-G m_z snapshots at the pulse peak."""
    in_dir = 'output/sweeps_S41_S49/S47'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    bucket = _load_sweep(in_dir)
    print(f'S47: loaded {sum(len(v) for v in bucket.values())} '
          f'traces in {len(bucket)} configs')
    _plot_v(bucket,
            out_path=os.path.join(out_dir, 'S47_A_v.png'))
    _plot_psi(bucket,
              out_path=os.path.join(out_dir, 'S47_B_psi.png'))
    _plot_axes(
        bucket,
        axis_key='D1_top',
        ylabel=r'$D_1$ at pulse max (nm)',
        title='S47(c): major axis vs J',
        out_path=os.path.join(out_dir, 'S47_C_D1.png'),
        check_box_limit=True)
    _plot_axes(
        bucket,
        axis_key='D2_top',
        ylabel=r'$D_2$ at pulse max (nm)',
        title='S47(d): minor axis vs J',
        out_path=os.path.join(out_dir, 'S47_D_D2.png'),
        check_box_limit=False)
    _plot_snapshots(bucket, out_dir)


# =============================================================================
if __name__ == '__main__':
    main()
