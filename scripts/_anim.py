"""Shared m_z animation helpers for the snapshots-NPZ animators.

Both `animate_simulation` and `animate_all` render the
out-of-plane component m_z of a `snapshots.npz` (written by the
C++ simulator) as a 2D color map over all stored frames and
encode it to mp4. This module holds the frame-title formatter
and the render-to-file routine they share.

Functions
---------
frame_title
    Format the single-line title shown on each frame.
render_mz_animation
    Load one snapshots.npz, animate m_z, and write an mp4.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def frame_title(phase_id, step, time_s, layer):
    """Title string for one animation frame.

    Parameters
    ----------
    phase_id : int
        Simulation phase (0 = relax, 1 = drive).
    step : int
        Integrator step index at this frame.
    time_s : float
        Wall-clock simulation time in seconds.
    layer : str
        Layer label ('top' or 'bot').

    Returns
    -------
    title : str
        Formatted single-line frame title.
    """
    phase_name = 'relax' if int(phase_id) == 0 else 'drive'
    t_ps = float(time_s) * 1e12
    return (f'{layer} layer | phase={int(phase_id)} ({phase_name})  '
            f'step={int(step)}  t={t_ps:.2f} ps')


# -----------------------------------------------------------------------------
def render_mz_animation(npz_path, save_path, layer, fps, figsize,
                        cmap):
    """Load one snapshots.npz, animate m_z, and write an mp4.

    Parameters
    ----------
    npz_path : str
        Path to the snapshots.npz produced by the C++ simulator.
    save_path : str
        Output mp4 path.
    layer : str
        Which layer to render: 'top' or 'bot'.
    fps : int
        Frames per second of the encoded animation.
    figsize : tuple[float]
        Matplotlib figure size in inches.
    cmap : str
        Colormap name for the m_z map.

    Returns
    -------
    n_frames : int
        Number of frames written.
    """
    # Select the requested layer's stored spin field.
    z = np.load(npz_path)
    if layer == 'top':
        m = z['m_top']
    elif layer == 'bot':
        m = z['m_bot']
    else:
        raise RuntimeError(
            f'render_mz_animation: layer must be top/bot, got '
            f'{layer!r}.')
    step = z['step']
    phase = z['phase_id']
    time_s = z['time_s']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Static figure: m_z on a fixed [-1, 1] diverging map,
    # origin='lower' so image rows match the array row order.
    n_frames = m.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(m[0, ..., 2], vmin=-1.0, vmax=1.0, cmap=cmap,
                    origin='lower', interpolation='nearest')
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    ax.set_xlabel('x (sites)')
    ax.set_ylabel('y (sites)')
    title = ax.set_title(
        frame_title(phase[0], step[0], time_s[0], layer))

    # Per-frame update: swap the image data and retitle.
    def _update(frame_idx):
        img.set_data(m[frame_idx, ..., 2])
        title.set_text(
            frame_title(phase[frame_idx], step[frame_idx],
                        time_s[frame_idx], layer))
        return [img, title]

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000.0 / fps,
        blit=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Encode via ffmpeg; yuv420p keeps the output broadly playable.
    writer = animation.FFMpegWriter(
        fps=fps, codec='libx264',
        extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(save_path, writer=writer, dpi=120)
    plt.close(fig)
    return n_frames
