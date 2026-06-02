"""Plotter for Pham et al. (2024) Figure S49.

Overlays the analytic SOT and TSH speeds (orange + blue/violet)
on top of the numerical LLGS-measured speeds.

Output: `output/figures_S41_S49/S49_v_R.png`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
from collections import defaultdict
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from src.simulator.parameters import default_params
from src.stochastic_llgs.diagnostics import unwrap_trajectory
from src.orchestrator.io import load_trace

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults.
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (6, 6)
plt.rcParams['lines.linewidth'] = 1.5


def _v_steady(trace, metadata):
    """Steady-state speed in the middle of the pulse, m/s,
    with PBC unwrap applied to the centroid stream."""
    t = trace['t']
    nx = int(metadata['nx'])
    ny = int(metadata['ny'])
    a = float(default_params().a)
    cx, _cy = unwrap_trajectory(
        trace['cx_top'], trace['cy_top'],
        L_x=nx * a, L_y=ny * a,
    )
    window = (t > 0.5e-9) & (t < 1.5e-9)
    if not np.any(window):
        return float('nan')
    return (float(cx[window][-1] - cx[window][0])
            / float(t[window][-1] - t[window][0]))


# Overlay analytic SOT/TSH speeds on the LLGS-measured speeds.
def main():
    in_dir = 'output/sweeps_S41_S49/S49'
    out_dir = 'output/figures_S41_S49'
    os.makedirs(out_dir, exist_ok=True)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Analytic curves. Accepts both legacy `analytic.npz` and
    # the new D-tagged `analytic_D{X}e-3.npz`; picks the last
    # match alphabetically (D-tagged sorts after the bare name).
    analytic_paths = sorted(glob.glob(
        os.path.join(in_dir, 'analytic*.npz')))
    if not analytic_paths:
        raise RuntimeError(
            f'plot_S49: no analytic*.npz in {in_dir}.')
    analytic, ana_meta = load_trace(analytic_paths[-1])
    R_over_Delta = analytic['R_over_Delta']
    v_sot_ana = analytic['v_SOT']
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # LLGS results: bucket by lambda_sq_nm2.
    llgs_paths = sorted(glob.glob(
        os.path.join(in_dir, 'llgs_R*.npz')))
    llgs_by_lam = defaultdict(list)
    for path in llgs_paths:
        trace, metadata = load_trace(path)
        llgs_by_lam[float(metadata['lambda_sq_nm2'])].append(
            (metadata, trace))
    # Sort each bucket by R/Delta.
    for lam in llgs_by_lam:
        llgs_by_lam[lam].sort(
            key=lambda mt: float(mt[0]['R_over_Delta']))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Plot.
    fig, ax = plt.subplots()
    # Analytic SOT (orange line).
    ax.plot(R_over_Delta, v_sot_ana,
            '-', color='C1',
            label='analytic $v_{\\mathrm{SOT}}$')
    # Analytic TSH curves (blue + violet).
    lam_colors = {3.0: 'C0', 50.0: 'mediumorchid'}
    for lam_nm2 in (3.0, 50.0):
        key = f'v_TSH_lam_{lam_nm2:.0f}nm2'
        if key not in analytic:
            continue
        ax.plot(
            R_over_Delta, np.abs(analytic[key]),
            '--', color=lam_colors[lam_nm2],
            label=(r'analytic $|v_{\mathrm{TSH}}|$, '
                   rf'$\lambda^2 = {lam_nm2:.0f}$ nm$^2$'))
    # LLGS points (open markers).
    marker_for_lam = {0.0: 'o', 3.0: 's', 50.0: '^'}
    for lam_nm2, items in sorted(llgs_by_lam.items()):
        rd_arr = np.array(
            [float(m['R_over_Delta']) for m, _ in items])
        v_arr = np.array(
            [_v_steady(t, m) for m, t in items])
        if lam_nm2 == 0.0:
            color = 'C1'
            label = 'LLGS $v$ (SOT only)'
        else:
            color = lam_colors.get(lam_nm2, 'k')
            label = (r'LLGS $v$, '
                     rf'$\lambda^2 = {lam_nm2:.0f}$ nm$^2$')
        ax.plot(
            rd_arr, np.abs(v_arr),
            marker_for_lam.get(lam_nm2, 'x'),
            linestyle='none',
            color=color, mfc='none',
            label=label)
    ax.set_xlabel(r'$R / \Delta$')
    ax.set_ylabel(r'$|v|$ (m/s)')
    ax.set_yscale('log')
    ax.legend(loc='best', frameon=False, fontsize=10)
    ax.set_box_aspect(1)
    fig.tight_layout()
    out_path = os.path.join(out_dir, 'S49_v_R.png')
    fig.savefig(out_path)
    print(f'Saved {out_path}')
    # Console summary.
    for i, rd in enumerate(R_over_Delta):
        line = (f'  R/Delta = {rd:.1f}: '
                f'v_SOT_ana = {v_sot_ana[i]:6.1f} m/s')
        for lam_nm2 in (3.0, 50.0):
            key = f'v_TSH_lam_{lam_nm2:.0f}nm2'
            if key in analytic:
                line += (f', v_TSH_ana(lam={lam_nm2:.0f}) = '
                         f'{analytic[key][i]:+7.2f} m/s')
        print(line)
    for lam_nm2, items in sorted(llgs_by_lam.items()):
        print(f'  LLGS, lambda_sq = {lam_nm2:.0f} nm^2:')
        for m, t in items:
            print(f'    R/Delta = {float(m["R_over_Delta"]):.1f}: '
                  f'v_steady = {_v_steady(t, m):6.1f} m/s')


# =============================================================================
if __name__ == '__main__':
    main()
