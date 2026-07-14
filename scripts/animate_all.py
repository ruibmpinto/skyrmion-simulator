"""Batch animator: generates an mp4 next to every snapshots.npz found
under a root directory.

Use after a sweep to produce one animation per parameter point. The
per-file load, figure, and encode logic lives in `scripts._anim`.

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
# Local
from scripts._anim import render_mz_animation

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
    """Walk a directory tree and emit an mp4 next to every
    snapshots.npz.

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
    # Find every snapshots.npz under the root.
    pattern = os.path.join(root, glob_pattern)
    inputs = sorted(glob.glob(pattern, recursive=True))
    if not inputs:
        print(f'animate_all: no files matched {pattern!r}.')
        return
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Render each, skipping those whose mp4 already exists.
    for npz_path in inputs:
        save_path = str(pathlib.Path(npz_path).with_suffix('.mp4'))
        if os.path.exists(save_path) and not overwrite:
            print(f'animate_all: skip {npz_path} (mp4 exists)')
            continue
        n_frames = render_mz_animation(
            npz_path, save_path, layer, fps, figsize, cmap)
        print(f'animate_all: wrote {save_path} ({n_frames} frames)')


# =============================================================================
if __name__ == '__main__':
    main()
