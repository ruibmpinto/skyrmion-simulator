"""Sweep orchestration utilities for SAF skyrmion simulations.

Shared helpers used by the per-figure sweep scripts that
reproduce Pham et al. (2024) supplementary figures S41-S49.

Modules
-------
observers
    `observe_state(m_top, m_bot, p)` - return a dict of scalar
    observables (centres, diameters, ellipse axes, DW angles,
    topological charges) for one simulation frame.
integrators
    Factories returning a callable matching the integrator
    protocol `step(m_top, m_bot, t, dt, p) -> (m_top, m_bot)`.
    Variants: deterministic RK4, demag-aware RK4, stochastic
    Heun. Lets `driver.run_one` stay integrator-agnostic.
driver
    `run_one(...)` - run a single trajectory: relax with J = 0,
    then drive with the supplied pulse, sampling observables
    at intervals and optionally recording a spin snapshot.
io
    `save_trace`, `load_trace` - NPZ persistence with explicit
    metadata.
"""
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'
