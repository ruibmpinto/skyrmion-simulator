"""Validation gates for the stochastic LLGS solver.

Modules
-------
macrospin
    Shared macrospin driver used by the Langevin and Brown
    reversal tests.
test_langevin
    Macrospin in Zeeman field; verifies <m_z>(T, B) matches
    the Langevin function.
test_brown_reversal
    Uniaxial macrospin; verifies the mean reversal time
    matches Brown's high-barrier formula.
test_equipartition
    Small ferromagnetic patch; verifies low-k magnon-mode
    energies match the Rayleigh-Jeans equipartition kT/2.
test_t0_limit_nodemag
    Phase 3a: T=0 SAF skyrmion drive matches the
    deterministic analysis.run_analysis to 2%.
test_t0_limit_demag
    Phase 3b: T=0 SAF skyrmion drive with demag matches a
    generated deterministic-with-demag baseline to 2%.
"""
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'
