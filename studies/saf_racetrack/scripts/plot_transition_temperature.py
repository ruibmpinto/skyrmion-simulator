"""Transition-temperature postprocessing for the DC scan.

Drives the $700\\times500$ SAF racetrack at a fixed current
$J=3\\times10^{11}$~A/m$^2$ over a narrow temperature band
$T\\in\\{105,110,115,120,125\\}$~K to locate the temperature at which the
skyrmion collapses. This script renders the three views of that scan:

- a config row: the ens0 final top-layer field at each temperature,
  class-coded, showing the compact-to-labyrinth melt;
- the class composition versus temperature (the 1D stability map);
- the survival probability and Hall angle versus temperature, with the
  interpolated transition temperature.

Functions
---------
main
    Load the ttrans aggregate + ens0 members and emit the figures.

Notes
-----
Reads the per-cell members for the config row and the aggregate for the
survival / composition curves. Classification uses the same
`classify_field` as the aggregation (S/E/L/A).
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt
# Local
from skyrmion_simulator.stochastic_llgs.stability import classify_field
#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira', ]
__status__ = 'Development'
# =============================================================================
#
# =============================================================================
def _class_of_member(path):
    """Classify one ens member from its stored final field.

    Parameters
    ----------
    path : str
        Member npz with keys mz_final_top, Q, D1_top, D2_top, L_x.

    Returns
    -------
    code : str
        Class letter from `classify_field` ('S', 'E', 'L', 'A').
    mz : numpy.ndarray(2d)
        Final top-layer m_z field.
    """
    d = np.load(path)
    q = abs(float(np.asarray(d['Q'])[-1]))
    d1 = d['D1_top']
    d2 = d['D2_top']
    half = max(1, d1.size // 2)
    d1_m = float(np.nanmean(d1[half:]))
    d2_m = float(np.nanmean(d2[half:]))
    mz = np.asarray(d['mz_final_top'], dtype=float)
    code, _ = classify_field(mz, q, d1_m, d2_m, float(d['L_x']))
    return code, mz
# -----------------------------------------------------------------------------
def _plot_config_row(ttrans_dir, temps, j_value, out_path, class_color):
    """Render the ens0 final-field config row over temperature.

    Parameters
    ----------
    ttrans_dir : str
        Directory holding the T{T}_j{J}_ens000.npz members.
    temps : list[float]
        Temperatures (K), left to right.
    j_value : float
        Fixed current density (A/m^2).
    out_path : str
        Output PNG path.
    class_color : dict
        Class-letter -> hex color for the title annotation.
    """
    fig, axes = plt.subplots(1, len(temps),
                             figsize=(3.0 * len(temps), 3.2))
    for ax, t in zip(axes, temps):
        g = sorted(glob.glob(
            f'{ttrans_dir}/T{t:05.1f}_j{j_value:.2e}_ens000.npz'))
        if not g:
            raise RuntimeError(f'_plot_config_row: no member for T={t}.')
        code, mz = _class_of_member(g[0])
        ax.imshow(mz, origin='lower', aspect='auto', cmap='RdBu_r',
                  vmin=-1.0, vmax=1.0)
        ax.set_title(f'$T={t:.0f}$ K -- {code}',
                     color=class_color[code], fontsize=13)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        r'ens0 final top-layer $m_z$ at $J=3\times10^{11}$'
        r' A m$^{-2}$', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}')
# -----------------------------------------------------------------------------
def _transition_temp(ts, p_surv):
    """Linear-interpolate the temperature where P_surv crosses 0.5.

    Parameters
    ----------
    ts : numpy.ndarray(1d)
        Temperatures (K), increasing.
    p_surv : numpy.ndarray(1d)
        Survival probability at each temperature.

    Returns
    -------
    t_c : float
        Interpolated crossing temperature, or NaN if 0.5 is not
        bracketed by the data.
    """
    for i in range(len(ts) - 1):
        a = p_surv[i]
        b = p_surv[i + 1]
        if (a - 0.5) * (b - 0.5) <= 0.0 and a != b:
            frac = (a - 0.5) / (a - b)
            return float(ts[i] + frac * (ts[i + 1] - ts[i]))
    return float('nan')
# -----------------------------------------------------------------------------
def _plot_transition(agg, out_path, class_color):
    """Class composition, survival, and Hall angle versus temperature.

    Parameters
    ----------
    agg : dict
        ttrans aggregate (Ts, P_surv, theta_mean, n_ens, n_S/E/L/A).
    out_path : str
        Output PNG path.
    class_color : dict
        Class-letter -> hex color for the stacked bars.
    """
    ts = np.asarray(agg['Ts'], float)
    p_surv = np.asarray(agg['P_surv'], float)[:, 0]
    theta = np.asarray(agg['theta_mean'], float)[:, 0]
    n = np.asarray(agg['n_ens'], float)[:, 0]
    fracs = {c: np.asarray(agg[f'n_{c}'], float)[:, 0] / n
             for c in ('S', 'E', 'L', 'A')}
    t_c = _transition_temp(ts, p_surv)
    fig, axes = plt.subplots(1, 3, figsize=(16.0, 4.6))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Stacked class composition (the 1D stability map).
    bottom = np.zeros_like(ts)
    labels = {'S': 'compact', 'E': 'elongated', 'L': 'labyrinth',
              'A': 'annihilated'}
    for c in ('S', 'E', 'L', 'A'):
        axes[0].bar(ts, fracs[c], width=3.5, bottom=bottom,
                    color=class_color[c], label=labels[c])
        bottom = bottom + fracs[c]
    axes[0].set_xlabel(r'$T$ (K)', fontsize=13)
    axes[0].set_ylabel('class fraction', fontsize=13)
    axes[0].set_title('stability composition', fontsize=13)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].legend(fontsize=10, framealpha=0.9)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Survival probability with the interpolated transition.
    axes[1].plot(ts, p_surv, '-o', color='#2c7bb6', lw=2)
    axes[1].axhline(0.5, color='k', ls=':', lw=0.8)
    if np.isfinite(t_c):
        axes[1].axvline(t_c, color='#d7191c', ls='--', lw=1.2)
        axes[1].text(t_c, 0.55, f'$T_c={t_c:.1f}$ K',
                     color='#d7191c', fontsize=12, ha='left')
    axes[1].set_xlabel(r'$T$ (K)', fontsize=13)
    axes[1].set_ylabel(r'$P_{\mathrm{surv}}$', fontsize=13)
    axes[1].set_title('survival probability', fontsize=13)
    axes[1].set_ylim(-0.03, 1.03)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Hall angle over the surviving members.
    axes[2].plot(ts, theta, '-s', color='#fdae61', lw=2)
    axes[2].axhline(0.0, color='k', ls=':', lw=0.8)
    axes[2].set_xlabel(r'$T$ (K)', fontsize=13)
    axes[2].set_ylabel(r'$\theta_{\mathrm{H}}$ (deg)', fontsize=13)
    axes[2].set_title('Hall angle', fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f'wrote {out_path}  (T_c={t_c:.1f} K)')
# -----------------------------------------------------------------------------
def main():
    """Emit the transition figures (config row + transition panels)."""
    # =========================== User Configuration =========================
    ttrans_dir = ('output/'
                  'stochastic_llgs/scan_track_width/campaign/'
                  'hk36_D0p72_700x500/ttrans')
    out_dir = 'output/figures_sllg/track_width/hk36_D0p72_700x500'
    temps = [105.0, 110.0, 115.0, 120.0, 125.0]
    j_value = 3.0e11
    class_color = {'S': '#2c7bb6', 'E': '#fdae61', 'L': '#d7191c',
                   'A': '#999999'}
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    agg = dict(np.load(os.path.join(ttrans_dir, 'aggregate.npz')))
    _plot_config_row(
        ttrans_dir, temps, j_value,
        os.path.join(out_dir, 'transition_temperature_config_row.png'),
        class_color)
    _plot_transition(
        agg, os.path.join(out_dir, 'transition_temperature.png'),
        class_color)
# =============================================================================
if __name__ == '__main__':
    main()
