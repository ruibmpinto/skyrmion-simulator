"""Static single-frame viewer for a snapshots.npz dump.

Renders the out-of-plane component m_z of one frame as a 2D color map.
By default plots the last frame; configure `frame_index` to pick
another frame, or `phase_filter` to pick the last frame of a phase.

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
import matplotlib.pyplot as plt
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def main():
    """Render a single frame of m_z to a PNG file.

    Run configuration sits below; edit and re-run.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration
    npz_path = 'src/simulator_cpp/build/output/snapshots.npz'
    layer = 'top'          # 'top' or 'bot'
    # Choose ONE of frame_index, phase_filter.
    frame_index = -1       # negative indices count from the end
    phase_filter = None    # 0 or 1 to pick the last frame of a phase
    save_path = None       # None => derived from npz_path stem + '_frame.png'
    figsize = (6, 6)
    cmap = 'RdBu_r'
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load snapshots
    if not os.path.exists(npz_path):
        raise RuntimeError(f'view_frame: {npz_path} does not exist.')
    z = np.load(npz_path)
    if layer == 'top':
        m = z['m_top']
    elif layer == 'bot':
        m = z['m_bot']
    else:
        raise RuntimeError(f'view_frame: layer must be top/bot, got {layer!r}.')
    step = z['step']
    phase = z['phase_id']
    time_s = z['time_s']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Resolve frame
    if phase_filter is not None:
        mask = phase == int(phase_filter)
        if not mask.any():
            raise RuntimeError(
                f'view_frame: no frame with phase_id == {phase_filter}.')
        # Last frame of the requested phase.
        idx = int(np.nonzero(mask)[0][-1])
    else:
        idx = int(frame_index) % m.shape[0]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Resolve output path
    if save_path is None:
        p = pathlib.Path(npz_path)
        suffix = f'_frame{idx:04d}_{layer}.png'
        save_path = str(p.with_name(p.stem + suffix))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Render
    mz = m[idx, ..., 2]
    fig, ax = plt.subplots(figsize=figsize)
    img = ax.imshow(mz, vmin=-1.0, vmax=1.0, cmap=cmap, origin='lower',
                    interpolation='nearest')
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    ax.set_xlabel('x (sites)')
    ax.set_ylabel('y (sites)')
    phase_name = 'relax' if int(phase[idx]) == 0 else 'drive'
    ax.set_title(
        f'{layer} layer | phase={int(phase[idx])} ({phase_name})  '
        f'step={int(step[idx])}  t={float(time_s[idx]) * 1e12:.2f} ps')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'view_frame: wrote {save_path} (frame {idx} of {m.shape[0]})')


# =============================================================================
if __name__ == '__main__':
    main()
