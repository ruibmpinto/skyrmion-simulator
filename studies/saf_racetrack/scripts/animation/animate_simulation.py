"""Animate a snapshots.npz produced by the C++ skyrmion simulator.

Renders the out-of-plane component m_z as a 2D color map across all
stored frames, exporting an mp4 alongside the input .npz by default.
Phases (0 = relax, 1 = drive) are shown in the frame title. The load,
figure, and encode logic lives in
`studies.saf_racetrack.scripts.animation._anim`.

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
# Local
from studies.saf_racetrack.scripts.animation._anim import render_mz_animation

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
    """Render an mp4 animation of m_z over all frames in a
    snapshots .npz.

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
    # Fail loudly if the input snapshots file is missing.
    if not os.path.exists(npz_path):
        raise RuntimeError(f'animate_simulation: {npz_path} does not exist.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Default the output next to the input, with an .mp4 suffix.
    if save_path is None:
        save_path = str(pathlib.Path(npz_path).with_suffix('.mp4'))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Render and report.
    n_frames = render_mz_animation(
        npz_path, save_path, layer, fps, figsize, cmap)
    print(f'animate_simulation: wrote {save_path} ({n_frames} frames @ '
          f'{fps} fps)')


# =============================================================================
if __name__ == '__main__':
    main()
