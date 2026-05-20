"""Post-simulation analysis for SAF skyrmion dynamics.

Relaxes a skyrmion to equilibrium, then drives it with
current and measures velocity, Hall angle, and diameter.
Compares results against the reference paper.

Functions
---------
skyrmion_center
    Compute skyrmion center of mass.
skyrmion_diameter
    Compute skyrmion diameter from mz < 0 area.
run_analysis
    Full relaxation + current-driven analysis.
"""
#
#                                                                Modules
# =====================================================================
# Standard
import sys
# Third-party
import numpy as np
# Local
from src.simulator.parameters import default_params
from src.simulator.initial_conditions import saf_skyrmion
from src.simulator.integrator import rk4_step
from src.simulator.main import topological_charge

#
#                                                   Authorship & Credits
# =====================================================================
__author__ = 'Rui Barreira'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =====================================================================
#
# =====================================================================


def skyrmion_center(m, a):
    """Compute skyrmion center of mass.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.

    Returns
    -------
    cx : float
        Center x-coordinate in meters.
    cy : float
        Center y-coordinate in meters.
    """
    # Extract number of grid points per direction
    ny, nx = m.shape[:2]
    # Create grid of lattic sites in units of a. jj, ii have shape (ny, nx).
    jj, ii = np.meshgrid(
        np.arange(nx, dtype=float),
        np.arange(ny, dtype=float),)
    # Weight w = (1 - m_z)/2 peaks at the skyrmion core (m_z = -1).
    # m_z stored in m[..., 2].
    # w has shape (ny, nx)
    w = (1.0 - m[..., 2]) / 2.0
    ws = w.sum()
    # Weighted centroid in physical (meters) coordinates.
    cx = np.sum(w * jj * a) / ws
    cy = np.sum(w * ii * a) / ws
    # Return
    return cx, cy


# ---------------------------------------------------------------------
def skyrmion_diameter(m, a):
    """Compute skyrmion diameter from mz < 0 area.

    Parameters
    ----------
    m : numpy.ndarray(3d)
        Spin configuration, shape (ny, nx, 3).
    a : float
        Lattice constant in meters.

    Returns
    -------
    d : float
        Skyrmion diameter in meters.
    """
    # Count sites inside the m_z = 0 contour (the skyrmion interior).
    # m_z stored in m[..., 2].
    n_inside = np.sum(m[..., 2] < 0)
    area = n_inside * a * a
    # Equivalent disk diameter d = 2 sqrt(A / pi).
    return 2.0 * np.sqrt(area / np.pi)


# ---------------------------------------------------------------------
def run_analysis():
    """Full relaxation + current-driven analysis.

    Phase 1: Relax skyrmion with J=0 for 500 ps.
    Phase 2: Drive with J=4e11 A/m^2 for 1 ns.
    Compare velocity, Hall angle, diameter to paper.
    """
    # =========================================================
    # Phase 1: Relaxation (no current)
    # =========================================================
    print('=== Phase 1: Relaxation (J=0, 500 ps) ===')
    p = default_params()
    # Zero current and SOT fields to find the true zero-drive equilibrium.
    p.J_current = 0.0
    p.H_DL = 0.0
    p.H_FL = 0.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    m_top, m_bot = saf_skyrmion(p.nx, p.ny, p.a, p.skyrmion_R)
    d0 = skyrmion_diameter(m_top, p.a)
    print(f'Initial diameter: {d0 * 1e9:.1f} nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_relax = 10000  # 10000 * 50 fs = 500 ps
    for step in range(1, n_relax + 1):
        m_top, m_bot = rk4_step(m_top, m_bot, p.dt, p)
        if step % 2000 == 0:
            d = skyrmion_diameter(m_top, p.a)
            t_ps = step * p.dt * 1e12
            print(f'  t={t_ps:.0f} ps: d={d * 1e9:.1f} nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    d_eq = skyrmion_diameter(m_top, p.a)
    Q_eq = topological_charge(m_top, p.a)
    print(f'Equilibrium: d={d_eq * 1e9:.1f} nm, Q={Q_eq:.4f}')
    print()
    # =========================================================
    # Phase 2: Current-driven dynamics
    # =========================================================
    print('=== Phase 2: Current drive (1 ns) ===')
    p2 = default_params()
    print(f'J = {p2.J_current:.2e} A/m^2')
    print(f'H_DL = {p2.H_DL:.4e} T')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Copy so the relaxed state is preserved as the initial drive frame.
    m_top_d = m_top.copy()
    m_bot_d = m_bot.copy()
    c0 = skyrmion_center(m_top_d, p2.a)
    print(f'Start: ({c0[0] * 1e9:.1f}, {c0[1] * 1e9:.1f}) nm')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_drive = 20000  # 20000 * 50 fs = 1 ns
    track = []
    for step in range(1, n_drive + 1):
        m_top_d, m_bot_d = rk4_step(m_top_d, m_bot_d, p2.dt, p2)
        if step % 4000 == 0:
            c = skyrmion_center(m_top_d, p2.a)
            d = skyrmion_diameter(m_top_d, p2.a)
            Q = topological_charge(m_top_d, p2.a)
            t_ps = step * p2.dt * 1e12
            track.append((t_ps, c[0], c[1]))
            print(
                f'  t={t_ps:.0f} ps: '
                f'({c[0] * 1e9:.1f}, '
                f'{c[1] * 1e9:.1f}) nm, '
                f'd={d * 1e9:.1f} nm, '
                f'Q={Q:.4f}')
    # =========================================================
    # Velocity from steady-state (last 3 points)
    # =========================================================
    if len(track) >= 3:
        # Finite-difference velocity from last two stored centers.
        t1, x1, y1 = track[-3]
        t2, x2, y2 = track[-1]
        dt_s = (t2 - t1) * 1e-12
        vx = (x2 - x1) / dt_s
        vy = (y2 - y1) / dt_s
        v = np.sqrt(vx ** 2 + vy ** 2)
        # Hall angle: deviation of motion from the drive direction (~0 SAF).
        hall = np.degrees(np.arctan2(abs(vy), abs(vx)))
        # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
        print()
        print('=== Comparison with paper ===')
        print(f'Equilibrium diameter:')
        print(f'  Sim:   {d_eq * 1e9:.0f} nm')
        print(f'  Paper: 197 nm')
        print(f'Velocity @ J={p2.J_current:.0e}:')
        print(f'  Sim:   {v:.0f} m/s (vx={vx:.0f}, vy={vy:.0f})')
        print(f'  Paper: ~400 m/s (mumag)')
        print(f'Hall angle:')
        print(f'  Sim:   {hall:.1f} deg')
        print(f'  Paper: ~0 deg (SAF)')
        print(f'Topological charge:')
        print(f'  Sim:   {Q:.4f}')
        print(f'  Paper: +/-1')

# =====================================================================
if __name__ == '__main__':
    run_analysis()
