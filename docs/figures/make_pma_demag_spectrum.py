"""Generate the PMA-vs-demag k-space-spectrum schematic used
in `docs/theory.tex` (Section 5.4, the conceptual subsection
``PMA vs demagnetization: same form, different physics'').

Plots three curves of the out-of-plane field per unit m_z
versus wavelength lambda = 2*pi/|k|:

  - PMA contribution H_PMA = +2 K / Ms (green, dashed).
    Constant in k because PMA is local in real space.
  - Slab demag contribution H_demag = -mu_0 Ms * f(|k| t_Co)
    with f(x) = (1 - exp(-x)) / x (red, dashed). Approaches
    -mu_0 Ms at lambda -> infinity and decays to zero at
    short wavelength.
  - Their sum (black, solid).
  - Local-K_eff approximation (grey, dotted) sitting at the
    long-wavelength sum value 2 K_eff / Ms for all
    wavelengths.

Vertical shaded bands mark the cell scale a = 2 nm, the
domain-wall width Delta_DW ~ 27 nm, and the skyrmion radius
R ~ 100 nm.

Outputs
-------
docs/figures/pma_demag_spectrum.png
docs/figures/pma_demag_spectrum.pdf
"""
#
#                                                                       Modules
# =============================================================================
# Standard
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
# Material parameters (top Co layer).
mu0 = 4.0 * np.pi * 1e-7   # T*m/A
Ms = 1.43e6                # A/m
K = 1.294e6                # J/m^3, bare PMA constant (Pham 2024)
t_Co = 1.30e-9             # m
shape = 0.5 * mu0 * Ms ** 2          # J/m^3
K_eff = K - shape                    # J/m^3, small difference
# Field magnitudes (Tesla per unit m_z).
H_PMA = 2.0 * K / Ms                 # +1.81 T
mu0_Ms = mu0 * Ms                    # +1.80 T
H_eff = 2.0 * K_eff / Ms             # ~0.011 T (small)

# Wavelength sweep (in nm), log-spaced over the relevant range.
lam_nm = np.logspace(0.0, 3.5, 400)  # 1 nm to ~3000 nm
lam_m = lam_nm * 1.0e-9
k = 2.0 * np.pi / lam_m              # 1/m
kt = k * t_Co                        # dimensionless
# Slab shape factor f(x) = (1 - exp(-x))/x; safe at x -> 0.
small = kt < 1e-6
f = np.where(small, 1.0 - kt / 2.0,
             (1.0 - np.exp(-kt)) / np.where(small, 1.0, kt))
# Per-mode demag contribution (slab, k=0 limit -> -mu0 Ms).
H_demag = -mu0_Ms * f                # Tesla per m_z
H_total = H_PMA + H_demag            # Tesla per m_z

# =============================================================================
fig, ax = plt.subplots(figsize=(7.5, 4.6))
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Length-scale bands (drawn first so curves overlay them).
ax.axvspan(1.0, 3.0, color='gray', alpha=0.10, zorder=0)
ax.axvspan(20.0, 35.0, color='tab:purple', alpha=0.10,
           zorder=0)
ax.axvspan(80.0, 130.0, color='tab:blue', alpha=0.10,
           zorder=0)
ax.text(1.8, 1.55, 'cell\n$a = 2$ nm',
        ha='center', va='center', fontsize=8,
        color='gray')
ax.text(26.0, 1.55, 'DW\n$\\Delta \\approx 27$ nm',
        ha='center', va='center', fontsize=8,
        color='tab:purple')
ax.text(102.0, 1.55, 'skyrmion\n$R \\approx 100$ nm',
        ha='center', va='center', fontsize=8,
        color='tab:blue')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Curves.
ax.axhline(0.0, color='black', lw=0.6, alpha=0.5)
ax.plot(lam_nm, np.full_like(lam_nm, H_PMA), '--',
        color='tab:green', lw=1.6,
        label=r'PMA  $+2K/M_s$  (local; $k$-independent)')
ax.plot(lam_nm, H_demag, '--', color='tab:red', lw=1.6,
        label=r'slab demag  $-\mu_0 M_s\,f(|k|\,t_{\rm Co})$')
ax.plot(lam_nm, H_total, '-', color='black', lw=2.0,
        label=r'PMA $+$ demag (full)')
ax.axhline(H_eff, color='gray', lw=1.2, ls=':',
           label=r'local $K_{\rm eff}$ approximation '
           r'($k=0$ value)')
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Bottom axis: wavelength in nm.
ax.set_xscale('log')
ax.set_xlim(1.0, 3000.0)
ax.set_xlabel(r'In-plane wavelength $\lambda = 2\pi/|k|$  '
              r'(nm)')
ax.set_ylabel(r'$H_z$ contribution per unit $m_z$  (T)')
ax.set_ylim(-2.0, 2.2)
ax.grid(True, which='both', axis='x', alpha=0.2)
ax.grid(True, which='major', axis='y', alpha=0.2)

# Top axis: |k| * t_Co. We map lambda -> kt = 2*pi*t_Co/lambda.
def lam_to_kt(lam_nm_arr):
    return 2.0 * np.pi * (t_Co * 1e9) / lam_nm_arr


def kt_to_lam(kt_arr):
    return 2.0 * np.pi * (t_Co * 1e9) / kt_arr


secax = ax.secondary_xaxis(
    'top', functions=(lam_to_kt, kt_to_lam))
secax.set_xlabel(r'$|k|\,t_{\rm Co}$')

# Legend below.
ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.22),
          ncol=2, framealpha=0.95, fontsize=9,
          handlelength=2.5)
ax.set_title('PMA vs slab demag: per-mode spectrum  '
             '($K_{\\rm top}\\!=\\!1.294$ MJ/m$^3$, '
             '$\\mu_0 M_s\\!=\\!1.80$ T)', fontsize=10)

fig.tight_layout()
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
out_dir = pathlib.Path(__file__).parent
out_dir.mkdir(parents=True, exist_ok=True)
fig.savefig(out_dir / 'pma_demag_spectrum.png', dpi=180)
fig.savefig(out_dir / 'pma_demag_spectrum.pdf')
print(f'wrote {out_dir / "pma_demag_spectrum.png"}')
print(f'wrote {out_dir / "pma_demag_spectrum.pdf"}')
print(f'  K_eff = {K_eff:.3e} J/m^3, 2 K_eff/M_s = {H_eff:.4f} T')
print(f'  H_PMA = {H_PMA:.4f} T, mu0 M_s = {mu0_Ms:.4f} T')
