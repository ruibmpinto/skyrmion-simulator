"""Plotter for Pham et al. (2024) Figure S47.

Reads `output/sweeps_S41_S49/S47/cfg<i>_J_<J0>.npz` and emits:

S47_A_v.png        v_avg vs J, one line per config.
S47_B_psi.png      psi_bot (at max diameter) vs J, per config.
S47_C_D1.png       max D1_top vs J, per config.
S47_D_D2.png       min D2_top vs J, per config.
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
from src.plots.plot_snapshot_mz import plot_snapshot_mz
from src.simulator.parameters import default_params
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.sweeps.io import load_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
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
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    cx, cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
    )
    FWHM = float(metadata['FWHM'])
    tail = (float(metadata['tail_sigmas'])
            * float(metadata['sigma']))
    t_lo = float(metadata['t_center']) - tail
    t_hi = float(metadata['t_center']) + tail
    i_lo = int(np.argmin(np.abs(t - t_lo)))
    i_hi = int(np.argmin(np.abs(t - t_hi)))
    dx = cx[i_hi] - cx[i_lo]
    dy = cy[i_hi] - cy[i_lo]
    return float(np.sqrt(dx * dx + dy * dy) / FWHM)


def _load_sweep(in_dir):
    paths = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not paths:
        raise RuntimeError(
            f'_load_sweep: no NPZ traces found in {in_dir!r}.')
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
    return (f'cfg {int(metadata["cfg_idx"])}: '
            f'FWHM={float(metadata["FWHM"])*1e12:.0f} ps, '
            f'$H_{{\\mathrm{{RKKY}}}}$='
            f'{float(metadata["H_RKKY"])*1e3:.0f} mT')


def _plot_v(bucket, out_path):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        J_arr = np.array([float(m['J0']) for m, _ in items])
        v_arr = np.array([_vavg(t, m) for m, t in items])
        c = cmap(k / max(len(bucket) - 1, 1))
        ax.plot(J_arr / 1e11, v_arr, 'o-', color=c,
                label=_config_label(items[0][0]))
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$v_{\mathrm{avg}}$ (m/s)')
    ax.set_title('S47(a): velocity vs J')
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_psi(bucket, out_path):
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        J_arr = np.array([float(m['J0']) for m, _ in items])
        psi_arr = []
        for _m, t in items:
            i_peak = int(np.argmax(t['d_top']))
            psi_arr.append(
                float(np.degrees(t['psi_bot'][i_peak])))
        c = cmap(k / max(len(bucket) - 1, 1))
        ax.plot(J_arr / 1e11, psi_arr, 'o-', color=c,
                label=_config_label(items[0][0]))
    ax.axhline(180.0, color='0.7', linestyle=':')
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(r'$\psi_{\mathrm{bot}}$ (deg)')
    ax.set_title('S47(b): DW angle vs J')
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path)
    print(f'Saved {out_path}')


def _plot_axes(bucket, axis_key, ylabel, title, out_path,
               reducer):
    """Generic plot for D1 (use reducer=np.max) and D2
    (reducer=np.min)."""
    fig, ax = plt.subplots()
    cmap = plt.get_cmap('viridis')
    for k, cfg_idx in enumerate(sorted(bucket.keys())):
        items = bucket[cfg_idx]
        J_arr = np.array([float(m['J0']) for m, _ in items])
        vals = np.array(
            [float(reducer(t[axis_key])) for _, t in items])
        c = cmap(k / max(len(bucket) - 1, 1))
        ax.plot(J_arr / 1e11, vals * 1e9, 'o-', color=c,
                label=_config_label(items[0][0]))
    ax.set_xlabel(r'$J$ ($10^{11}$ A/m$^2$)')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
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
        a = float(m['nx'])  # not used directly here; nx kept for sanity
        # The simulator's lattice spacing is in the trace
        # implicitly via metadata: every sweep uses Set A's
        # default `p.a`. Re-import default_params to recover it
        # without rerunning the simulation.
        from src.simulator.parameters import default_params
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
        ylabel=r'max $D_1$ (nm)',
        title='S47(c): major axis vs J',
        out_path=os.path.join(out_dir, 'S47_C_D1.png'),
        reducer=np.max)
    _plot_axes(
        bucket,
        axis_key='D2_top',
        ylabel=r'min $D_2$ (nm)',
        title='S47(d): minor axis vs J',
        out_path=os.path.join(out_dir, 'S47_D_D2.png'),
        reducer=np.min)
    _plot_snapshots(bucket, out_dir)


# =============================================================================
if __name__ == '__main__':
    main()
