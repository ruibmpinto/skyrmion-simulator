"""Batch animator: generates an mp4 next to every snapshots.npz found
under a root directory.

Use after a sweep to produce one animation per parameter point.

Functions
---------
main
    Entry point; configuration variables sit at the top.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
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
    """Walk a directory tree and emit an mp4 next to every snapshots.npz.

    Run configuration sits below; edit and re-run.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration
    root = 'src/simulator_cpp/build/output'
    glob_pattern = '**/snapshots.npz'
    overwrite = False
    layer = 'top'
    fps = 20
    figsize = (6, 6)
    cmap = 'RdBu_r'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Find inputs
    pattern = os.path.join(root, glob_pattern)
    inputs = sorted(glob.glob(pattern, recursive=True))
    if not inputs:
        print(f'animate_all: no files matched {pattern!r}.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Process each
    for npz_path in inputs:
        save_path = str(pathlib.Path(npz_path).with_suffix('.mp4'))
        if os.path.exists(save_path) and not overwrite:
            print(f'animate_all: skip {npz_path} (mp4 exists)')
            continue
        _render_one(npz_path, save_path, layer, fps, figsize, cmap)


# -----------------------------------------------------------------------------
def _render_one(npz_path, save_path, layer, fps, figsize, cmap):
    """Render the mp4 for one snapshots.npz file."""
    # Load
    z = np.load(npz_path)
    if layer == 'top':
        m = z['m_top']
    elif layer == 'bot':
        m = z['m_bot']
    else:
        raise RuntimeError(f'animate_all: layer must be top/bot, got {layer!r}.')
    step = z['step']
    phase = z['phase_id']
    time_s = z['time_s']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Figure
    n_frames = m.shape[0]
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(m[0, ..., 2], vmin=-1.0, vmax=1.0, cmap=cmap,
                    origin='lower', interpolation='nearest')
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    ax.set_xlabel('x (sites)')
    ax.set_ylabel('y (sites)')
    title = ax.set_title(_frame_title(phase[0], step[0], time_s[0], layer))

    def _update(frame_idx):
        img.set_data(m[frame_idx, ..., 2])
        title.set_text(
            _frame_title(phase[frame_idx], step[frame_idx],
                         time_s[frame_idx], layer))
        return [img, title]

    ani = animation.FuncAnimation(
        fig, _update, frames=n_frames, interval=1000.0 / fps, blit=False)
    writer = animation.FFMpegWriter(fps=fps, codec='libx264',
                                    extra_args=['-pix_fmt', 'yuv420p'])
    ani.save(save_path, writer=writer, dpi=120)
    plt.close(fig)
    print(f'animate_all: wrote {save_path} ({n_frames} frames)')


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
