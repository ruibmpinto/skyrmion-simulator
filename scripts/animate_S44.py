"""Reconstructed-ellipse animation from an S44 NPZ trace.

The S44 sweep stored time-resolved scalar observables only
(centroid, average diameter, ellipse axes D1/D2 + tilt theta,
DW magnetization angle psi, topological charge Q). The full
m(x, y, t) spin field is not in the NPZ. This script renders
an animation that re-creates the skyrmion as a moving,
deforming ellipse in the simulation box, with side panels
showing the time series of d, (D1, D2), psi, and Q.

Layout
------
- Main panel: 2D view of the simulation box (PBC-wrapped),
  top-layer skyrmion drawn as a filled tilted ellipse, with
  a short DW-magnetization arrow at angle psi. The centroid
  trail (post-unwrap, wrapped back into the box) is drawn
  with fading alpha.
- Side stack: d(t), (D1(t), D2(t)), psi(t), Q(t) traces with
  a vertical cursor at the current frame's time.

Output
------
Writes `output/figures_S41_S49/S44_anim.mp4` (or .gif if
ffmpeg is unavailable). Frames are sub-sampled from the
trace's 737 saved samples to keep the encoded video short.

Functions
---------
main
    Load the panelA NPZ, build the figure, drive FuncAnimation.
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
from matplotlib.animation import FuncAnimation, FFMpegWriter, \
    PillowWriter
from matplotlib.patches import Ellipse
# Local
from src.simulator.parameters import default_params
from src.orchestrator.io import load_trace

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
plt.rcParams['figure.dpi'] = 120
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['lines.linewidth'] = 1.5


def _wrap_into_box(coord, L):
    """Reduce a real-valued coordinate into [0, L) under PBC."""
    return coord - L * np.floor(coord / L)


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration (edit here).
    # Source trace.
    in_dir = 'output/sweeps_S41_S49/S44'
    in_glob = 'panelA*.npz'
    out_dir = 'output/figures_S41_S49'
    # Output format. Prefer MP4 (small, smooth); fall back to GIF
    # if ffmpeg is missing.
    out_basename = 'S44_anim'
    fps = 30
    # Sub-sample the trace to keep the encoded video short.
    n_frames_target = 240
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the trace.
    paths = sorted(glob.glob(os.path.join(in_dir, in_glob)))
    if not paths:
        raise RuntimeError(
            f'animate_S44: no NPZ matches {in_glob} in {in_dir}.')
    # Use the alphabetically-last match (D-tagged sorts after
    # the legacy `panelA.npz`).
    trace, metadata = load_trace(paths[-1])
    print(f'animate_S44: loaded {paths[-1]}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Time, geometry, and observables.
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    L_x = nx * a
    L_y = ny * a
    # Wrap centroids into the box (the NPZ stores PBC-wrapped
    # coordinates, but we re-wrap defensively for clarity).
    cx = _wrap_into_box(trace['cx_top'], L_x)
    cy = _wrap_into_box(trace['cy_top'], L_y)
    # Per-frame ellipse parameters (top layer).
    D1 = trace['D1_top']
    D2 = trace['D2_top']
    theta = trace['theta_top']
    psi = trace['psi_top']
    Q = trace['Q_top']
    d = trace['d_top']
    # Drive-pulse profile for the time axis (drawn as context).
    sigma = float(metadata['sigma'])
    t_center = float(metadata['t_center'])
    J0 = float(metadata['J0'])
    z = (t - t_center) / sigma
    J_t = J0 * np.exp(-0.5 * z * z)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Frame selection: even sub-sample down to ~n_frames_target.
    n = t.size
    step = max(1, int(np.ceil(n / n_frames_target)))
    frame_indices = np.arange(0, n, step)
    n_frames = frame_indices.size
    print(f'animate_S44: n_samples={n}, step={step}, '
          f'n_frames={n_frames}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure layout: 2D box on the left, 4 stacked time-series
    # subplots on the right.
    fig = plt.figure(figsize=(11, 6))
    gs = fig.add_gridspec(
        nrows=4, ncols=2,
        width_ratios=[1.4, 1.0],
        height_ratios=[1, 1, 1, 1],
        wspace=0.30, hspace=0.45)
    ax_box = fig.add_subplot(gs[:, 0])
    ax_d = fig.add_subplot(gs[0, 1])
    ax_D12 = fig.add_subplot(gs[1, 1], sharex=ax_d)
    ax_psi = fig.add_subplot(gs[2, 1], sharex=ax_d)
    ax_Q = fig.add_subplot(gs[3, 1], sharex=ax_d)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # 2D box: equal aspect, axes labelled in nm, fixed limits.
    ax_box.set_xlim(0.0, L_x * 1e9)
    ax_box.set_ylim(0.0, L_y * 1e9)
    ax_box.set_aspect('equal')
    ax_box.set_xlabel(r'$x$ (nm)')
    ax_box.set_ylabel(r'$y$ (nm)')
    ax_box.set_title(
        f'top-layer skyrmion, $J_0$ = {J0:.1e} A/m$^2$')
    # Tilted-ellipse patch for the skyrmion (initial frame).
    ellipse = Ellipse(
        xy=(cx[0] * 1e9, cy[0] * 1e9),
        width=D1[0] * 1e9, height=D2[0] * 1e9,
        angle=np.degrees(theta[0]),
        facecolor='C0', alpha=0.45, edgecolor='C0',
        linewidth=1.2)
    ax_box.add_patch(ellipse)
    # DW magnetization arrow (rooted at the centroid, length
    # = half-minor-axis for visibility).
    arrow_len_nm = 0.5 * D2[0] * 1e9
    dw_arrow = ax_box.annotate(
        '', xy=(cx[0] * 1e9 + arrow_len_nm * np.cos(psi[0]),
                cy[0] * 1e9 + arrow_len_nm * np.sin(psi[0])),
        xytext=(cx[0] * 1e9, cy[0] * 1e9),
        arrowprops=dict(arrowstyle='->', color='C3', lw=1.6))
    # Centroid trail (start empty; built up over frames).
    trail_line, = ax_box.plot(
        [], [], '-', color='0.55', alpha=0.7, linewidth=0.8)
    # Time tag.
    time_text = ax_box.text(
        0.02, 0.97, '', transform=ax_box.transAxes,
        ha='left', va='top', fontsize=11,
        bbox=dict(boxstyle='round', facecolor='white',
                  edgecolor='0.8', alpha=0.8))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Side-panel static traces. Cursor lines updated per frame.
    t_ps = t * 1e12
    ax_d.plot(t_ps, d * 1e9, '-', color='C0')
    ax_d.set_ylabel(r'$d$ (nm)')
    ax_d.tick_params(labelbottom=False)
    cursor_d = ax_d.axvline(t_ps[0], color='0.4', linestyle=':')
    # D1 / D2 panel.
    ax_D12.plot(t_ps, D1 * 1e9, '-', color='C3',
                label=r'$D_1$')
    ax_D12.plot(t_ps, D2 * 1e9, '-', color='C2',
                label=r'$D_2$')
    ax_D12.set_ylabel(r'axes (nm)')
    ax_D12.tick_params(labelbottom=False)
    ax_D12.legend(loc='upper left', frameon=False, ncol=2,
                  fontsize=8)
    cursor_D12 = ax_D12.axvline(
        t_ps[0], color='0.4', linestyle=':')
    # psi panel.
    ax_psi.plot(t_ps, np.degrees(psi), '-', color='C4')
    ax_psi.set_ylabel(r'$\psi$ (deg)')
    ax_psi.tick_params(labelbottom=False)
    cursor_psi = ax_psi.axvline(
        t_ps[0], color='0.4', linestyle=':')
    # Q panel.
    ax_Q.plot(t_ps, Q, '-', color='C5')
    ax_Q.set_ylabel(r'$Q_\mathrm{top}$')
    ax_Q.set_xlabel(r'$t$ (ps)')
    cursor_Q = ax_Q.axvline(
        t_ps[0], color='0.4', linestyle=':')
    # Light pulse shading on all four side panels (visual cue).
    for ax in (ax_d, ax_D12, ax_psi, ax_Q):
        ax.axvspan(
            (t_center - 3.0 * sigma) * 1e12,
            (t_center + 3.0 * sigma) * 1e12,
            color='violet', alpha=0.08)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-frame update.
    trail_x_nm = []
    trail_y_nm = []

    def update(frame_i):
        i = int(frame_indices[frame_i])
        # Update ellipse (top layer).
        ellipse.set_center((cx[i] * 1e9, cy[i] * 1e9))
        ellipse.set_width(D1[i] * 1e9)
        ellipse.set_height(D2[i] * 1e9)
        ellipse.set_angle(np.degrees(theta[i]))
        # Update DW arrow (re-rooted at centroid).
        L_arrow = 0.5 * D2[i] * 1e9
        dw_arrow.xy = (
            cx[i] * 1e9 + L_arrow * np.cos(psi[i]),
            cy[i] * 1e9 + L_arrow * np.sin(psi[i]))
        dw_arrow.xyann = (cx[i] * 1e9, cy[i] * 1e9)
        # Append to trail (wrap into box again to keep it visible).
        trail_x_nm.append(cx[i] * 1e9)
        trail_y_nm.append(cy[i] * 1e9)
        trail_line.set_data(trail_x_nm, trail_y_nm)
        # Update cursors and time tag.
        for cur in (cursor_d, cursor_D12, cursor_psi, cursor_Q):
            cur.set_xdata([t_ps[i], t_ps[i]])
        time_text.set_text(
            f't = {t_ps[i]:6.1f} ps\n'
            f'J = {J_t[i]/1e11:5.2f}'
            r' $\times 10^{11}$ A/m$^2$')
        return (ellipse, dw_arrow, trail_line, cursor_d,
                cursor_D12, cursor_psi, cursor_Q, time_text)

    anim = FuncAnimation(
        fig, update, frames=n_frames, interval=1000.0 / fps,
        blit=False, repeat=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode. MP4 preferred (ffmpeg writer); GIF fallback.
    os.makedirs(out_dir, exist_ok=True)
    out_mp4 = os.path.join(out_dir, out_basename + '.mp4')
    out_gif = os.path.join(out_dir, out_basename + '.gif')
    try:
        writer = FFMpegWriter(
            fps=fps, codec='libx264',
            extra_args=['-pix_fmt', 'yuv420p'])
        anim.save(out_mp4, writer=writer)
        print(f'Saved {out_mp4}')
    except Exception as e:
        print(f'FFmpeg unavailable ({e}); writing GIF instead.')
        anim.save(out_gif, writer=PillowWriter(fps=fps))
        print(f'Saved {out_gif}')


# =============================================================================
if __name__ == '__main__':
    main()
