"""Animate the m_z field stored in a LAMMPS-style dump file.

Reads `output/skyrmion.dump`, written by `skyrmion_simulator.simulator.io_ovito`
in LAMMPS dump format with the spin components (mx, my, mz)
packed into the (fx, fy, fz) columns. Renders the top-layer
m_z(x, y) heatmap for every dumped frame and writes an MP4 (or
GIF) animation.

Functions
---------
read_dump_frames_mz_top
    Stream the dump file and return mz(t, y, x) for the top
    layer along with axis coordinates and timestep indices.
animate_mz
    Drive a matplotlib FuncAnimation over the parsed frames.

Notes
-----
Run from the repository root:
    python -m studies.saf_racetrack.scripts.animate_skyrmion
or equivalently:
    python studies/saf_racetrack/scripts/animate_skyrmion.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import sys
import pathlib
# Third-party
import numpy as np
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Add project root directory to sys.path
root_dir = str(pathlib.Path(__file__).resolve().parents[3])
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults (match plot_snapshot_mz).
plt.rcParams['figure.dpi'] = 180
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['lines.linewidth'] = 1.5


def read_dump_frames_mz_top(filepath):
    """Stream a LAMMPS dump file and pull top-layer m_z frames.

    Parses every frame in the file in a single pass. Only the
    top layer (type == 1) is retained. The atom block of each
    frame must be ordered row-major in (y, x), i.e. consecutive
    rows of constant y, matching how `skyrmion_simulator.simulator.io_ovito`
    flattens the spin arrays.

    Parameters
    ----------
    filepath : str
        Path to the LAMMPS dump file.

    Returns
    -------
    steps : numpy.ndarray(1d)
        Timestep integer of each frame, shape (n_frames,).
    x_nm : numpy.ndarray(1d)
        Unique top-layer x coordinates in nm, shape (nx,).
    y_nm : numpy.ndarray(1d)
        Unique top-layer y coordinates in nm, shape (ny,).
    mz : numpy.ndarray(3d)
        Top-layer m_z field, shape (n_frames, ny, nx).
    """
    if not os.path.isfile(filepath):
        raise RuntimeError(
            f'read_dump_frames_mz_top: file not found at '
            f'{filepath!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load the file once. `np.loadtxt` is the slow path; reading
    # all lines and slicing them per frame is faster on a 6 MB file.
    with open(filepath, 'r') as f:
        lines = f.readlines()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Locate frame boundaries.
    frame_starts = [i for i, ln in enumerate(lines)
                    if ln.startswith('ITEM: TIMESTEP')]
    if len(frame_starts) == 0:
        raise RuntimeError(
            f'read_dump_frames_mz_top: no frames found in '
            f'{filepath!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parse first frame to fix the (nx, ny) grid.
    start0 = frame_starts[0]
    n_atoms0 = int(lines[start0 + 3].strip())
    block0 = lines[start0 + 9:start0 + 9 + n_atoms0]
    arr0 = np.loadtxt(block0)
    types0 = arr0[:, 1].astype(int)
    mask0_top = (types0 == 1)
    if not np.any(mask0_top):
        raise RuntimeError(
            'read_dump_frames_mz_top: frame 0 has no type-1 '
            '(top-layer) atoms.')
    x0_top = arr0[mask0_top, 2]
    y0_top = arr0[mask0_top, 3]
    x_unique = np.unique(x0_top)
    y_unique = np.unique(y0_top)
    nx = int(x_unique.size)
    ny = int(y_unique.size)
    n_top = nx * ny
    if mask0_top.sum() != n_top:
        raise RuntimeError(
            f'read_dump_frames_mz_top: top-layer atom count '
            f'({int(mask0_top.sum())}) does not match nx*ny '
            f'({n_top}).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Allocate outputs.
    n_frames = len(frame_starts)
    steps = np.zeros(n_frames, dtype=int)
    mz = np.zeros((n_frames, ny, nx), dtype=float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Populate frame 0 (already parsed).
    steps[0] = int(lines[start0 + 1].strip())
    mz0_top = arr0[mask0_top, 7]
    mz[0] = mz0_top.reshape(ny, nx)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Parse remaining frames.
    for k in range(1, n_frames):
        s = frame_starts[k]
        n_atoms_k = int(lines[s + 3].strip())
        if n_atoms_k != 2 * n_top:
            raise RuntimeError(
                f'read_dump_frames_mz_top: frame {k} has '
                f'{n_atoms_k} atoms, expected {2 * n_top}.')
        block = lines[s + 9:s + 9 + n_atoms_k]
        arr = np.loadtxt(block)
        types = arr[:, 1].astype(int)
        mask_top = (types == 1)
        if mask_top.sum() != n_top:
            raise RuntimeError(
                f'read_dump_frames_mz_top: frame {k} top-layer '
                f'count mismatch ({int(mask_top.sum())} vs '
                f'{n_top}).')
        steps[k] = int(lines[s + 1].strip())
        mz[k] = arr[mask_top, 7].reshape(ny, nx)
    return steps, x_unique, y_unique, mz


def animate_mz(steps, x_nm, y_nm, mz, out_path, fps,
               dump_every, dt_s):
    """Animate the top-layer m_z field and write to disk.

    Parameters
    ----------
    steps : numpy.ndarray(1d)
        Timestep integer for each frame, shape (n_frames,).
    x_nm : numpy.ndarray(1d)
        Grid x coordinates in nm.
    y_nm : numpy.ndarray(1d)
        Grid y coordinates in nm.
    mz : numpy.ndarray(3d)
        m_z field, shape (n_frames, ny, nx).
    out_path : str
        Output file path. Extension drives the writer:
        '.mp4' uses ffmpeg, '.gif' uses pillow.
    fps : int
        Frames per second of the encoded animation.
    dump_every : int
        Simulator's dump_every; used together with dt_s to label
        the wall-clock time of each frame.
    dt_s : float
        Simulator integration step in seconds.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Static figure scaffold.
    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    X, Y = np.meshgrid(x_nm, y_nm, indexing='xy')
    norm = mcolors.Normalize(vmin=-1.0, vmax=1.0)
    im = ax.pcolormesh(
        X, Y, mz[0], cmap='RdBu_r', norm=norm,
        shading='nearest', rasterized=True)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r'$m_z$')
    ax.set_xlabel(r'$x$ (nm)')
    ax.set_ylabel(r'$y$ (nm)')
    ax.set_aspect('equal', adjustable='box')
    title = ax.set_title('')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-frame update: pcolormesh.set_array wants the flattened
    # field for 'nearest' shading.
    def update(k):
        im.set_array(mz[k].ravel())
        t_ps = float(steps[k]) * dt_s * 1e12
        title.set_text(
            f'frame {k} / step {int(steps[k])} / t = '
            f'{t_ps:.2f} ps')
        return (im, title)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build the animation. blit=False to keep the title redraw.
    anim = FuncAnimation(
        fig, update, frames=mz.shape[0],
        interval=1000.0 / fps, blit=False)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pick writer from extension.
    ext = os.path.splitext(out_path)[1].lower()
    if ext == '.mp4':
        anim.save(out_path, fps=fps, dpi=180,
                  writer='ffmpeg')
    elif ext == '.gif':
        anim.save(out_path, fps=fps, dpi=120,
                  writer='pillow')
    else:
        raise RuntimeError(
            f'animate_mz: unsupported extension {ext!r}; use '
            f'.mp4 or .gif.')
    plt.close(fig)
    print(f'Saved {out_path}')
    # Mention dump cadence so the title's time axis is auditable.
    print(f'  fps={fps}, dt={dt_s:.3e} s, dump_every='
          f'{dump_every} steps')


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Run configuration (no argparse; tweak here).
    dump_path = 'output/skyrmion.dump'
    out_path = 'output/figures/skyrmion_animation.mp4'
    fps = 5
    # The dump cadence and integration step govern the wall-clock
    # time label printed on each frame. The defaults below match
    # `default_params()` at the time of writing; importing them
    # here ties the label to the simulator authoritatively.
    from skyrmion_simulator.simulator.parameters import default_params
    p = default_params()
    dt_s = float(p.dt)
    dump_every = int(p.dump_every)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Read the dump file.
    print(f'Reading {dump_path}...')
    steps, x_nm, y_nm, mz = read_dump_frames_mz_top(dump_path)
    print(f'  {mz.shape[0]} frames, grid '
          f'{mz.shape[2]} x {mz.shape[1]}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Ensure output directory exists.
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Render.
    animate_mz(steps=steps, x_nm=x_nm, y_nm=y_nm, mz=mz,
               out_path=out_path, fps=fps,
               dump_every=dump_every, dt_s=dt_s)


# =============================================================================
if __name__ == '__main__':
    main()
