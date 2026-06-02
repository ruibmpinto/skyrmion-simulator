"""Bogdanov-Hubert + soft-mode diagnostics from S41_D_sweep.

Reads every NPZ in `output/sweeps_S41_S49/S41_D_sweep/` and
emits one figure with three side-by-side panels:

A. Equilibrium skyrmion diameter d_eq(D), the Bogdanov-Hubert
   curve. Curves are grouped by `field_kind` (keff vs newell
   vs slab) when both are present.
B. Relax residual torque tau_max_final(D), log-scale on y.
   Drop with D = stiff regime; flat plateau or rise as D
   approaches D_c = soft-mode signature.
C. Analytical D_c overlay computed from default_params:
   D_c = (4 / pi) sqrt(A_ex * K_eff), K_eff = K_bar - mu0*Ms^2/2.

Reads
-----
output/sweeps_S41_S49/S41_D_sweep/D_*.npz

Writes
------
output/figures_S41_S49/bogdanov_diag.png
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import json
import math
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.simulator.parameters import default_params

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _load_records(in_dir):
    """Load every NPZ in in_dir, grouped by field_kind.

    Newer runs carry `field_kind` in metadata; legacy Newell
    runs only had `demag_kind`. Files with neither are tagged
    `'unknown'` so they cannot silently merge with a real
    group."""
    paths = sorted(glob.glob(os.path.join(in_dir, 'D_*.npz')))
    if not paths:
        raise RuntimeError(
            f'plot_bogdanov_diag: no D_*.npz traces in {in_dir}.')
    groups = {}
    for p in paths:
        z = np.load(p, allow_pickle=True)
        md = json.loads(z['_metadata'].item())
        kind = (md.get('field_kind')
                or md.get('demag_kind')
                or 'unknown')
        d_eq_nm = float(z['d_top'][0]) * 1e9
        D_mJ = float(md['D']) * 1e3
        tau = float(md.get('relax_tau_max_final', float('nan')))
        conv = bool(md.get('relax_converged', False))
        groups.setdefault(kind, []).append(
            {'D': D_mJ, 'd_eq': d_eq_nm,
             'tau': tau, 'conv': conv})
    for kind in groups:
        groups[kind].sort(key=lambda r: r['D'])
    return groups


# Fixed (colour, marker) per field model so the two physics
# paths stay visually distinct.
style = {
    'keff':    ('tab:blue', 'o'),
    'slab':    ('tab:cyan', 's'),
    'newell':  ('tab:red',  '^'),
    'unknown': ('0.4',      'x'),
}


def _analytical_Dc(p=None):
    """D_c = (4/pi) sqrt(A * K_eff) for the layer-averaged K.

    K_eff = (K_top + K_bot)/2 - mu0*Ms^2/2
    """
    if p is None:
        p = default_params()
    K_bar = 0.5 * (float(p.K_top) + float(p.K_bot))
    K_eff = K_bar - 0.5 * float(p.mu0) * float(p.Ms) ** 2
    A = float(p.A_ex)
    if K_eff <= 0.0:
        return float('nan'), K_bar, K_eff, A
    Dc = (4.0 / math.pi) * math.sqrt(A * K_eff)
    return Dc, K_bar, K_eff, A


# Build the 2-panel d_eq(D) and residual-torque(D) diagnostic.
def main():
    in_dir = 'output/sweeps_S41_S49/S41_D_sweep'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'bogdanov_diag.png')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Load and analytical Dc.
    groups = _load_records(in_dir)
    Dc, K_bar, K_eff, A = _analytical_Dc()
    Dc_mJ = Dc * 1e3 if not math.isnan(Dc) else float('nan')
    print(f'Analytical D_c = {Dc_mJ:.3f} mJ/m^2 '
          f'(A = {A*1e12:.1f} pJ/m, K_bar = {K_bar*1e-6:.3f} '
          f'MJ/m^3, K_eff = {K_eff*1e-3:.1f} kJ/m^3)',
          flush=True)
    for kind, rs in groups.items():
        print(f'  group {kind}: {len(rs)} points '
              f'(D in {rs[0]["D"]:.2f}..{rs[-1]["D"]:.2f} '
              f'mJ/m^2)', flush=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel A: d_eq vs D (Bogdanov-Hubert).
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.5))
    for kind, rs in sorted(groups.items()):
        D = np.array([r['D'] for r in rs])
        d = np.array([r['d_eq'] for r in rs])
        color, marker = style.get(kind, style['unknown'])
        a.plot(D, d, marker=marker, linestyle='-',
               color=color, label=kind)
    if not math.isnan(Dc_mJ):
        a.axvline(Dc_mJ, ls='--', color='0.4',
                  label=f'analytic $D_c$ = {Dc_mJ:.3f} mJ/m$^2$')
    a.set_xlabel('$D$ (mJ/m$^2$)')
    a.set_ylabel('$d_\\mathrm{eq}$ (nm)')
    a.set_title('(a) Bogdanov-Hubert: $d_\\mathrm{eq}$ vs $D$')
    a.legend(loc='best', frameon=False)
    a.set_box_aspect(1)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Panel B: tau_max_final vs D, log y.
    for kind, rs in sorted(groups.items()):
        D = np.array([r['D'] for r in rs])
        tau = np.array([r['tau'] for r in rs])
        color, marker = style.get(kind, style['unknown'])
        b.semilogy(D, tau, marker=marker, linestyle='-',
                   color=color, label=kind)
    if not math.isnan(Dc_mJ):
        b.axvline(Dc_mJ, ls='--', color='0.4')
    b.set_xlabel('$D$ (mJ/m$^2$)')
    b.set_ylabel(r'$\tau_\mathrm{max,\,final}$ (T)')
    b.set_title('(b) Relax residual torque vs $D$')
    b.legend(loc='best', frameon=False)
    b.set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    print(f'Saved {out_path}', flush=True)


# =============================================================================
if __name__ == '__main__':
    main()
