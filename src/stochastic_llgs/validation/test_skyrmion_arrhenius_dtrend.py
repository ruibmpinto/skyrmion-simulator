"""Skyrmion collapse barrier vs DMI trend (benchmark #8b).

Post-processing on the DMI-sweep scan
(`scan_skyrmion_arrhenius_dsweep`). For each DMI value D it
builds the censored mean collapse time tau(T), fits the
Arrhenius law to extract the activation barrier Delta_E(D),
then checks that the barrier RISES with D -- the qualitative
trend of

    Rohart, Miltat, Thiaville, PRB 93, 214412 (2016), Fig. 5(b)

(collapse barrier increases strongly with DMI strength at fixed
field). The absolute Delta_E is not gated (grid-dependent,
model-class mismatch -- see rohart_skyrmion); the gated quantity
is the monotone increase of Delta_E with D.

Pass criteria:
  - at least `n_D_min` DMI values yield a usable per-D Arrhenius
    fit (>= `n_T_min` temperatures with >= `n_events_min`
    collapses, R^2 >= `r2_min`),
  - Delta_E(D) is strictly increasing across those D
    (Spearman-style: every step up in D raises Delta_E).

Run with:
    python -m src.stochastic_llgs.validation.test_skyrmion_arrhenius_dtrend
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import math
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.stochastic_llgs.validation.test_skyrmion_arrhenius import (
    censored_tau,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
_KB_J = 1.380649e-23


def _fit_dE(by_T, t_max, n_events_min, n_T_min, r2_min):
    """Fit Delta_E (meV), tau_0 (s), R^2 for one DMI value.

    Returns (dE_meV, tau0, r2, n_used) or (nan, nan, nan, n)
    if fewer than `n_T_min` temperatures are usable.
    """
    T_used, tau_used = [], []
    for T in sorted(by_T):
        tau_hat, n_ev, _n_tot = censored_tau(
            np.array(by_T[T], dtype=float), t_max)
        if (n_ev >= n_events_min and np.isfinite(tau_hat)
                and tau_hat > 0.0):
            T_used.append(T)
            tau_used.append(tau_hat)
    if len(T_used) < n_T_min:
        return float('nan'), float('nan'), float('nan'), \
            len(T_used)
    x = 1.0 / np.array(T_used, dtype=float)
    y = np.log(np.array(tau_used, dtype=float))
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    dE_meV = slope * _KB_J / 1.602176634e-22
    tau0 = math.exp(intercept)
    return dE_meV, tau0, r2, len(T_used)


def main():
    # Load the DMI-sweep output, fit Delta_E(D) per DMI value, and
    # gate on Delta_E rising monotonically with D. Returns True on
    # PASS.
    # =========================== User Configuration =========================
    in_dir = ('output/stochastic_llgs/validation/'
              'skyrmion_arrhenius_dsweep')
    plot_path = ('docs/figures/validation/'
                 'skyrmion_arrhenius_dtrend.png')
    n_events_min    = 10
    n_T_min         = 3
    n_D_min         = 3
    r2_min          = 0.85
    # ======================= End User Configuration =========================
    print('test_skyrmion_arrhenius_dtrend (Delta_E vs DMI, '
          'Rohart 2016 Fig.5b trend):')
    # Prefer the single aggregate (D, T_sub, t_collapse) NPZ.
    agg_path = in_dir + '_agg.npz'
    by_DT = {}
    t_max = None
    if os.path.exists(agg_path):
        z = np.load(agg_path, allow_pickle=True)
        D_arr = np.asarray(z['D'], dtype=float)
        T_arr = np.asarray(z['T_sub'], dtype=float)
        tc_arr = np.asarray(z['t_collapse'], dtype=float)
        t_max = float(z['n_drive']) * float(z['dt'])
        for D, T, tc in zip(D_arr, T_arr, tc_arr):
            by_DT.setdefault(float(D), {}).setdefault(
                float(T), []).append(float(tc))
        print(f'  using aggregate {agg_path!r} '
              f'({D_arr.size} trajectories)')
    else:
        files = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
        if not files:
            print(f'  NO scan output in {in_dir!r} (or '
                  f'{agg_path!r}): run the DMI sweep first.')
            print('  status: INCONCLUSIVE (no scan data)')
            return False
        for path in files:
            z = np.load(path, allow_pickle=True)
            D = float(z['D'])
            T = float(z['T_sub'])
            tc = float(z['t_collapse'])
            by_DT.setdefault(D, {}).setdefault(T, []).append(tc)
            this_tmax = float(z['n_drive']) * float(z['dt'])
            if t_max is None:
                t_max = this_tmax
            elif abs(this_tmax - t_max) > 1e-18:
                raise RuntimeError(
                    'test_..._dtrend: inconsistent t_max '
                    f'({this_tmax} vs {t_max}).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Fit Delta_E per DMI value.
    D_vals, dE_vals, tau0_vals = [], [], []
    for D in sorted(by_DT):
        dE, tau0, r2, n_used = _fit_dE(
            by_DT[D], t_max, n_events_min, n_T_min, r2_min)
        ok_fit = np.isfinite(dE) and r2 >= r2_min
        flag = 'use' if ok_fit else 'skip(poor fit / few T)'
        dE_str = 'n/a' if not np.isfinite(dE) \
            else f'{dE:6.1f} meV'
        r2_str = 'n/a' if not np.isfinite(r2) else f'{r2:.3f}'
        print(f'  D={D*1e3:.2f} mJ/m^2: n_T={n_used:2d}, '
              f'R^2={r2_str}, Delta_E={dE_str}, '
              f'tau_0={tau0*1e9:.2f} ns [{flag}]'
              if np.isfinite(tau0) else
              f'  D={D*1e3:.2f} mJ/m^2: n_T={n_used:2d}, '
              f'R^2={r2_str}, Delta_E={dE_str} [{flag}]')
        if ok_fit:
            D_vals.append(D)
            dE_vals.append(dE)
            tau0_vals.append(tau0)
    D_vals = np.array(D_vals, dtype=float)
    dE_vals = np.array(dE_vals, dtype=float)
    if D_vals.size < n_D_min:
        print(f'  only {D_vals.size} usable D (need {n_D_min}).')
        print('  status: INCONCLUSIVE (insufficient D coverage)')
        return False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Trend gate: Delta_E strictly increasing with D.
    order = np.argsort(D_vals)
    dE_sorted = dE_vals[order]
    increasing = bool(np.all(np.diff(dE_sorted) > 0.0))
    print('-' * 60)
    print(f'  Delta_E(D): '
          + ', '.join(f'{d*1e3:.2f}->{e:.1f}meV'
                      for d, e in zip(D_vals[order], dE_sorted)))
    print(f'  monotone increasing with D (Rohart Fig.5b trend): '
          f'[{"PASS" if increasing else "FAIL"}]')
    print('  (absolute Delta_E reported only -- grid-dependent, '
          'not gated.)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=160)
    ax.plot(D_vals[order] * 1e3, dE_sorted, 'o-',
            color='tab:green')
    ax.set_xlabel(r'$D$ (mJ/m$^2$)')
    ax.set_ylabel(r'$\Delta E$ (meV)')
    ax.set_title('Skyrmion collapse barrier vs DMI '
                 '(Rohart Fig. 5b trend)')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  Delta_E(D) plot: {plot_path}')
    print(f'  status: {"PASS" if increasing else "FAIL"} '
          f'(Delta_E rises with D)')
    return increasing


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
