"""LAMMPS dump file writer for Ovito visualization.

Writes spin configurations in LAMMPS dump format with spin
components mapped to force vectors (fx, fy, fz) since Ovito
does not natively support spin degrees of freedom.
Positions are written in nanometers for proper Ovito display.

Functions
---------
write_dump
    Write one timestep to a LAMMPS dump file.
"""
#
#                                                                Modules
# =====================================================================
# Third-party
import numpy as np

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================

# Conversion factor: meters to nanometers
_M_TO_NM = 1e9


def write_dump(filepath, m_top, m_bot, pos_top, pos_bot,
               step, mode='a'):
    """Write SAF spin configuration in LAMMPS dump format.

    Both layers are written as separate atom types.
    Spin components are stored as force columns (fx, fy, fz)
    so Ovito can render them as arrows. Positions are
    converted from meters to nanometers.

    Parameters
    ----------
    filepath : str
        Output file path.
    m_top : numpy.ndarray(3d)
        Top layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom layer spins, shape (ny, nx, 3).
    pos_top : numpy.ndarray(3d)
        Top layer positions in meters, shape (ny, nx, 3).
    pos_bot : numpy.ndarray(3d)
        Bottom layer positions in meters, shape (ny, nx, 3).
    step : int
        Timestep number.
    mode : str, default='a'
        File open mode. Use 'w' for first frame, 'a' to
        append subsequent frames.

    Notes
    -----
    Type 1 = top layer, type 2 = bottom layer.
    Positions output in nanometers.
    """
    ny, nx = m_top.shape[:2]
    n_top = nx * ny
    n_atoms = 2 * n_top
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Flatten and convert to nm
    pt = pos_top.reshape(-1, 3) * _M_TO_NM
    pb = pos_bot.reshape(-1, 3) * _M_TO_NM
    st = m_top.reshape(-1, 3)
    sb = m_bot.reshape(-1, 3)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Box bounds in nm
    all_x = np.concatenate([pt[:, 0], pb[:, 0]])
    all_y = np.concatenate([pt[:, 1], pb[:, 1]])
    all_z = np.concatenate([pt[:, 2], pb[:, 2]])
    margin = 0.5  # nm
    x_lo, x_hi = all_x.min() - margin, all_x.max() + margin
    y_lo, y_hi = all_y.min() - margin, all_y.max() + margin
    z_lo, z_hi = all_z.min() - 1.0, all_z.max() + 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Build data block (vectorized)
    ids_top = np.arange(1, n_top + 1)
    ids_bot = np.arange(n_top + 1, n_atoms + 1)
    types_top = np.ones(n_top, dtype=int)
    types_bot = np.full(n_top, 2, dtype=int)
    # Stack top + bottom
    ids = np.concatenate([ids_top, ids_bot])
    types = np.concatenate([types_top, types_bot])
    pos = np.vstack([pt, pb])
    spins = np.vstack([st, sb])
    # Combine into single array for writing
    data = np.column_stack([
        ids, types,
        pos[:, 0], pos[:, 1], pos[:, 2],
        spins[:, 0], spins[:, 1], spins[:, 2],
    ])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write
    with open(filepath, mode) as f:
        f.write('ITEM: TIMESTEP\n')
        f.write(f'{step}\n')
        f.write('ITEM: NUMBER OF ATOMS\n')
        f.write(f'{n_atoms}\n')
        f.write('ITEM: BOX BOUNDS pp pp pp\n')
        f.write(f'{x_lo:.4f} {x_hi:.4f}\n')
        f.write(f'{y_lo:.4f} {y_hi:.4f}\n')
        f.write(f'{z_lo:.4f} {z_hi:.4f}\n')
        f.write(
            'ITEM: ATOMS id type x y z '
            'fx fy fz\n'
        )
        np.savetxt(
            f, data,
            fmt='%d %d %.4f %.4f %.4f %.6f %.6f %.6f',
        )
