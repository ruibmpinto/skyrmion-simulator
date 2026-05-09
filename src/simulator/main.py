"""Main simulation loop for SAF skyrmion dynamics.

Runs the LLGS time integration for a synthetic
antiferromagnet with two coupled Co layers, writing
spin configurations to LAMMPS dump format for Ovito.

Functions
---------
topological_charge
    Compute the topological charge of a spin texture.
run
    Execute the simulation loop.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import os
import time
# Third-party
import numpy as np
# Local
from src.simulator.parameters import default_params
from src.simulator.lattice import lattice_positions
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rk4_step
from src.simulator.io_ovito import write_dump

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def topological_charge(m, a):
    """Compute the topological charge of a 2D spin texture.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.

    Returns
    -------
    Q : float
        Topological charge. Should be close to +/-1 for
        a single skyrmion.

    Notes
    -----
    Q = (1/4*pi) * integral m . (dm/dx x dm/dy) dA

    Uses central finite differences for the derivatives.
    """
    # Central differences
    dmdx = (
        np.roll(m, -1, axis=1) - np.roll(m, +1, axis=1)
    ) / (2.0 * a)
    dmdy = (
        np.roll(m, -1, axis=0) - np.roll(m, +1, axis=0)
    ) / (2.0 * a)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Cross product dm/dx x dm/dy
    cross = np.cross(dmdx, dmdy)
    # Dot with m
    density = np.sum(m * cross, axis=-1)
    # Integrate
    Q = np.sum(density) * a * a / (4.0 * np.pi)
    return Q


# ---------------------------------------------------------------------
def run(p=None):
    """Execute the SAF skyrmion simulation.

    Parameters
    ----------
    p : SimpleNamespace, default=None
        Simulation parameters. If None, uses defaults.

    Notes
    -----
    Workflow:
        1. Build lattice positions for both layers.
        2. Initialize SAF skyrmion pair.
        3. Relax to equilibrium (J=0) for n_relax steps.
        4. Current-driven dynamics with RK4.
        5. Dump snapshots and print diagnostics.
    """
    if p is None:
        p = default_params()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output directory
    os.makedirs(p.output_dir, exist_ok=True)
    dump_path = os.path.join(
        p.output_dir, p.output_file
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Lattice positions
    pos_top = lattice_positions(p.nx, p.ny, p.a)
    pos_bot = lattice_positions(p.nx, p.ny, p.a)
    pos_bot[..., 2] = -p.t_Co
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Initial conditions: SAF skyrmion pair
    m_top, m_bot = saf_skyrmion(
        p.nx, p.ny, p.a,
        p.skyrmion_R, p.skyrmion_dw,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Header
    print('SAF Skyrmion Simulator')
    print(f'  Lattice: {p.nx} x {p.ny}')
    print(f'  Relax: {p.n_relax} steps (J=0)')
    print(f'  Drive: {p.n_steps} steps, '
          f'dt = {p.dt:.2e} s')
    print(f'  Dump every {p.dump_every} steps')
    print(f'  Output: {dump_path}')
    if p.J_current != 0.0:
        print(f'  Current: J = {p.J_current:.2e} A/m^2')
        print(f'  H_DL = {p.H_DL:.4e} T')
        print(f'  H_FL = {p.H_FL:.4e} T')
    print('-' * 50)
    t_start = time.time()
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 1: Relaxation (J=0)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if p.n_relax > 0:
        print('Relaxation phase (J=0)...')
        # Temporarily disable SOT
        H_DL_save = p.H_DL
        H_FL_save = p.H_FL
        p.H_DL = 0.0
        p.H_FL = 0.0
        for step in range(1, p.n_relax + 1):
            m_top, m_bot = rk4_step(
                m_top, m_bot, p.dt, p
            )
            if step % (p.n_relax // 5) == 0:
                Q = topological_charge(m_top, p.a)
                t_ps = step * p.dt * 1e12
                print(f'  t={t_ps:.0f} ps  Q={Q:+.4f}')
        # Restore SOT
        p.H_DL = H_DL_save
        p.H_FL = H_FL_save
        print('Relaxation done.')
        print('-' * 50)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Phase 2: Current-driven dynamics
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('Current-driven phase...')
    for step in range(p.n_steps + 1):
        # Dump and diagnostics
        if step % p.dump_every == 0:
            Q_top = topological_charge(m_top, p.a)
            Q_bot = topological_charge(m_bot, p.a)
            print(
                f'Step {step:>6d}  '
                f'Q_top = {Q_top:+.4f}  '
                f'Q_bot = {Q_bot:+.4f}'
            )
            file_mode = 'w' if step == 0 else 'a'
            write_dump(
                dump_path, m_top, m_bot,
                pos_top, pos_bot,
                step, mode=file_mode,
            )
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        # RK4 step
        if step < p.n_steps:
            m_top, m_bot = rk4_step(
                m_top, m_bot, p.dt, p
            )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Summary
    elapsed = time.time() - t_start
    print('-' * 50)
    print(f'Done. Wall time: {elapsed:.1f} s')
    print(f'Output written to: {dump_path}')


# =====================================================================
if __name__ == '__main__':
    run()
