"""Animate a snapshots.npz produced by the C++ skyrmion simulator.

Renders the out-of-plane component m_z as a 2D color map across all
stored frames, exporting an mp4 alongside the input .npz by default.
Phases (0 = relax, 1 = drive) are shown in the frame title.

Functions
---------
main
    Entry point; configuration variables sit at the top.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pathlib
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


def main():
    """Render an mp4 animation of m_z over all frames in a snapshots .npz.

    Run configuration sits below; edit and re-run.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration
    npz_path = ('src/simulator_cpp/build/output/snapshots.npz')
    layer = 'top'              # 'top' or 'bot'
    fps = 20
    save_path = None           # None => derived from npz_path stem + '.mp4'
    figsize = (6, 6)
    cmap = 'RdBu_r'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load snapshots
    if not os.path.exists(npz_path):
        raise RuntimeError(f'animate_simulation: {npz_path} does not exist.')
    z = np.load(npz_path)
    if layer == 'top':
        m = z['m_top']
    elif layer == 'bot':
        m = z['m_bot']
    else:
        raise RuntimeError(f'animate_simulation: layer must be top/bot, '
                           f'got {layer!r}.')
    step = z['step']
    phase = z['phase_id']
    time_s = z['time_s']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Resolve output path
    if save_path is None:
        p = pathlib.Path(npz_path)
        save_path = str(p.with_suffix('.mp4'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build figure
    n_frames = m.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(m[0, ..., 2], vmin=-1.0, vmax=1.0, cmap=cmap,
                    origin='lower', interpolation='nearest')
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    ax.set_xlabel('x (sites)')
    ax.set_ylabel('y (sites)')
    title = ax.set_title(_frame_title(phase[0], step[0], time_s[0], layer))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Frame update

    def _update(frame_idx):
        img.set_data(m[frame_idx, ..., 2])
        title.set_text(
            _frame_title(phase[frame_idx], step[frame_idx],
                         time_s[frame_idx], layer))
        return [img, title]

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000.0 / fps, blit=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Export mp4
    writer = animation.FFMpegWriter(fps=fps, codec='libx264',
                                    extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(save_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f'animate_simulation: wrote {save_path} ({n_frames} frames @ '
          f'{fps} fps)')


# -----------------------------------------------------------------------------
def _frame_title(phase_id, step, time_s, layer):
    """Title string for one animation frame."""
    phase_name = 'relax' if int(phase_id) == 0 else 'drive'
    t_ps = float(time_s) * 1e12
    return (f'{layer} layer | phase={int(phase_id)} ({phase_name})  '
            f'step={int(step)}  t={t_ps:.2f} ps')


# =============================================================================
if __name__ == '__main__':
    main()
