"""Stochastic LLGS solver for thermal SAF skyrmion dynamics.

Brown's stochastic Landau-Lifshitz-Gilbert-Slonczewski equation
with multiplicative Gaussian white-noise thermal field, integrated
in the Stratonovich sense by a single-step stochastic Heun
predictor-corrector. Existing deterministic machinery
(`src.simulator.fields.effective_field`,
`src.simulator.integrator.llgs_rhs`, demag pair) is imported
verbatim and never modified.

Submodules
----------
parameters_thermal
    Attach T, R_th, seed, V_cell, sigma_noise to an existing
    parameters namespace.
thermal_field
    Sample Gaussian noise field with std sigma / sqrt(dt).
integrator_sllg
    Stochastic Heun step for the SAF pair.
joule_heating
    Uniform global T(j) = T_sub + R_th * j**2.
diagnostics
    PBC-unwrapped skyrmion tracking, annihilation guard,
    Hall-angle fit.
ensemble
    ProcessPoolExecutor-based ensemble runner.
io
    NPZ writers for trajectories and grid results.
validation/
    Macrospin Langevin, Brown reversal, lattice
    equipartition, T=0 deterministic-limit gates.
production/
    Single-trajectory runner and (T, j), (T, j=0)
    Arrhenius, (D, H_z) radius, and pair-potential scans.
"""
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'
