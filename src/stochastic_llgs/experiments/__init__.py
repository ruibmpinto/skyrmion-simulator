"""Production runs for stochastic SAF skyrmion dynamics.

Modules
-------
run_single
    Single (T_sub, j) trajectory.
scan_tj
    Grid scan over (T_sub, j); demag off.
scan_arrhenius
    Zero-drive thermal-stability scan tau(T); demag off.
scan_radius
    Sweep (D, H_z) to vary skyrmion radius; demag on.
pair_potential
    Forced-pair runs to reconstruct V(r); demag on.
"""
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'
