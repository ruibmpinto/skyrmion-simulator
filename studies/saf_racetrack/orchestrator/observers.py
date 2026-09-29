"""Per-frame observation dictionary for sweep traces.

`observe_state(m_top, m_bot, p)` packs every scalar that any of
the S41-S49 figures might need into a single dict, so the sweep
driver records one consistent payload regardless of which
figure is being reproduced. Plot scripts pick the keys they
need and ignore the rest.

Functions
---------
observe_state
    Return a dict of scalar observables for one simulation
    frame.
"""
#
#                                                                       Modules
# =============================================================================
# Local
from skyrmion_simulator.simulator.analysis import (
    dw_angle,
    skyrmion_diameter,
    skyrmion_ellipse,
)
from skyrmion_simulator.simulator.main import topological_charge
from skyrmion_simulator.stochastic_llgs.diagnostics import skyrmion_center_pbc

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def observe_state(m_top, m_bot, p):
    """Compute the scalar observables for one simulation frame.

    Parameters
    ----------
    m_top : numpy.ndarray(3d)
        Top-layer spins, shape (ny, nx, 3).
    m_bot : numpy.ndarray(3d)
        Bottom-layer spins, shape (ny, nx, 3).
    p : SimpleNamespace
        Simulation parameters; only `p.a` is used here.

    Returns
    -------
    obs : dict
        Maps observable name to float. Keys:
            cx_top, cy_top, cx_bot, cy_bot
                Skyrmion centroids in metres.
            d_top, d_bot
                Equivalent-disk diameters (m).
            D1_top, D2_top, theta_top
                Top-layer ellipse major/minor diameters (m)
                and orientation (rad).
            D1_bot, D2_bot, theta_bot
                Same for the bottom layer.
            psi_top, psi_bot
                Right-DW in-plane angle relative to -x (rad).
            Q_top, Q_bot
                Topological charges (dimensionless).
    """
    # Layer polarities for the SAF: top layer core at m_z = -1
    # (saf_skyrmion polarity=+1), bottom layer core at m_z = +1
    # (polarity=-1). These are the conventions used by
    # `initial_conditions.saf_skyrmion`.
    pol_top = +1
    pol_bot = -1
    # Centroid of each layer's skyrmion. Use the PBC-safe
    # variant from `stochastic_llgs.diagnostics` so a skyrmion
    # that wraps around the box still gives a finite, single-
    # valued centroid; downstream plot scripts must unwrap the
    # stream before differencing (see `unwrap_trajectory`).
    cx_top, cy_top = skyrmion_center_pbc(m_top, p.a, pol_top)
    cx_bot, cy_bot = skyrmion_center_pbc(m_bot, p.a, pol_bot)
    # Equivalent-disk diameter from the core mask.
    d_top = skyrmion_diameter(m_top, p.a, pol_top)
    d_bot = skyrmion_diameter(m_bot, p.a, pol_bot)
    # Ellipse fit (major axis, minor axis, orientation).
    D1_top, D2_top, theta_top = skyrmion_ellipse(m_top, p.a, pol_top)
    D1_bot, D2_bot, theta_bot = skyrmion_ellipse(m_bot, p.a, pol_bot)
    # Domain-wall magnetization angle at the right DW.
    psi_top = dw_angle(m_top, p.a, pol_top)
    psi_bot = dw_angle(m_bot, p.a, pol_bot)
    # Topological charges (sanity check / qualitative).
    Q_top = topological_charge(m_top, p.a)
    Q_bot = topological_charge(m_bot, p.a)
    # Pack everything into one dict so the driver appends a
    # consistent payload at every sampled frame.
    return {
        'cx_top': float(cx_top),
        'cy_top': float(cy_top),
        'cx_bot': float(cx_bot),
        'cy_bot': float(cy_bot),
        'd_top': float(d_top),
        'd_bot': float(d_bot),
        'D1_top': float(D1_top),
        'D2_top': float(D2_top),
        'theta_top': float(theta_top),
        'D1_bot': float(D1_bot),
        'D2_bot': float(D2_bot),
        'theta_bot': float(theta_bot),
        'psi_top': float(psi_top),
        'psi_bot': float(psi_bot),
        'Q_top': float(Q_top),
        'Q_bot': float(Q_bot),
    }
