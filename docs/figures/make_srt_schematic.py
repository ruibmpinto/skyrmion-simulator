"""Generate the spin-reorientation-transition schematic
plot used in `docs/theory.tex` (Section 2.3.1).

Plots three curves of effective anisotropy K_eff (J/m^3) vs
Co thickness t (nm):
  - Interface PMA contribution +2*K_s/t (decreasing with t).
  - Shape-anisotropy contribution -1/2 mu_0 M_s^2 (constant).
  - Their sum K_eff = +2*K_s/t - 1/2 mu_0 M_s^2.
A vertical dashed line marks the SRT thickness t_c where
K_eff changes sign. A vertical solid line marks the Co
layer thickness t_Co = 1.3 nm used by the simulator.

Outputs
-------
docs/figures/srt_schematic.png
docs/figures/srt_schematic.pdf
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import pathlib
# Third-party
import matplotlib.pyplot as plt
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Material parameters (Co/Pt). K_s is set so that the bare
# K = 2*K_s/t_Co matches the K_top = 1.294 MJ/m^3 used in the
# simulator parameters; this places t_c right at t_Co = 1.3 nm,
# illustrating the "right at the SRT, deliberately" design.
mu0 = 4.0 * np.pi * 1e-7  # T*m/A
Ms = 1.43e6               # A/m
shape_term = 0.5 * mu0 * Ms ** 2   # J/m^3 (~1.286e6)
t_Co = 1.30e-9            # m, simulator Co thickness
K_top = 1.294e6           # J/m^3, simulator bare anisotropy
K_s = K_top * t_Co / 2.0  # J/m^2 (~8.4e-4)
# Critical thickness from setting K_eff = 0 (K_vol = 0):
# 2*K_s/t_c = shape_term  -> t_c = 2 K_s / shape_term.
t_c = 2.0 * K_s / shape_term
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Sample t over [0.5, 3.0] nm.
t_nm = np.linspace(0.5, 3.0, 400)
t_m = t_nm * 1.0e-9
pma_term = 2.0 * K_s / t_m   # J/m^3
shape_arr = np.full_like(t_m, -shape_term)
K_eff = pma_term + shape_arr
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
fig, ax = plt.subplots(figsize=(6.5, 4.2))
# In-plane shaded region (K_eff < 0) and out-of-plane (K_eff > 0)
y_min, y_max = -2.0e6, 2.0e6
ax.axhspan(0.0, y_max, color='tab:blue', alpha=0.08,
           zorder=0)
ax.axhspan(y_min, 0.0, color='tab:orange', alpha=0.08,
           zorder=0)
# Three curves
ax.plot(t_nm, pma_term / 1.0e6, '-', color='tab:green',
        lw=1.6, label=r'$+2K_s/t$  (interface PMA)')
ax.plot(t_nm, shape_arr / 1.0e6, '--', color='tab:red',
        lw=1.6, label=r'$-\frac{1}{2}\mu_0 M_s^2$  '
                       r'(shape anisotropy)')
ax.plot(t_nm, K_eff / 1.0e6, '-', color='black', lw=2.0,
        label=r'$K_{\rm eff} = +2K_s/t '
              r'- \frac{1}{2}\mu_0 M_s^2$')
# Zero line
ax.axhline(0.0, color='gray', lw=0.8, ls='-', alpha=0.6)
# t_c marker
ax.axvline(t_c * 1.0e9, color='gray', lw=1.0, ls=':',
           alpha=0.8)
ax.annotate(f'$t_c \\approx {t_c*1e9:.2f}$ nm',
            xy=(t_c * 1.0e9, -1.6),
            xytext=(t_c * 1.0e9 + 0.18, -1.6),
            fontsize=10, color='gray')
# Simulator marker
ax.axvline(t_Co * 1.0e9, color='black', lw=1.0, ls='-',
           alpha=0.8)
ax.annotate(f'$t_{{\\rm Co}} = {t_Co*1e9:.2f}$ nm '
            r'(simulator)',
            xy=(t_Co * 1.0e9, 1.6),
            xytext=(t_Co * 1.0e9 + 0.1, 1.6),
            fontsize=10)
# Phase labels (placed in regions clear of any curve).
ax.text(2.5, 1.4, 'PMA wins\n(out-of-plane easy axis)',
        fontsize=10, color='tab:blue',
        ha='center', va='center')
ax.text(0.85, -0.55, 'in-plane easy axis',
        fontsize=10, color='tab:orange',
        ha='center', va='center')
# Cosmetics: generic y-axis label so the legend disambiguates
# the three curves (any of the three can be read off this axis).
ax.set_xlabel(r'Co layer thickness $t$ (nm)')
ax.set_ylabel(r'Anisotropy energy density  '
              r'($10^6$ J/m$^3$)')
ax.set_xlim(0.5, 3.0)
ax.set_ylim(y_min / 1e6, y_max / 1e6)
# Legend below the plot so it does not overlap any curve or
# annotation in the data area.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.18),
          ncol=3, framealpha=0.95, fontsize=9,
          handlelength=2.5)
ax.set_title('Spin reorientation transition in '
             r'Co/Pt: $K_{\rm eff}(t)$', fontsize=11)
ax.grid(True, alpha=0.25)
fig.tight_layout()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
out_dir = pathlib.Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'srt_schematic.png', dpi=180)
fig.savefig(out_dir / 'srt_schematic.pdf')
print(f'wrote {out_dir / "srt_schematic.png"}')
print(f'wrote {out_dir / "srt_schematic.pdf"}')
print(f't_c = {t_c*1e9:.3f} nm (with K_s={K_s*1e4:.2f} '
      f'x 10^-4 J/m^2)')
