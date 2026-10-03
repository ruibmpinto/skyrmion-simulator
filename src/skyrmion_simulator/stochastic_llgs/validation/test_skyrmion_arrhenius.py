"""Skyrmion Neel-Arrhenius gate (benchmark #8).

Post-processing on the `scan_skyrmion_arrhenius` output. For
each temperature it forms the censored maximum-likelihood mean
collapse time tau(T), fits the Arrhenius law

    tau(T) = tau_0 * exp(Delta_E / (k_B T)),

i.e. ln(tau) linear in 1/T, and gates on the Arrhenius FORM,
the attempt-time order of magnitude, and the T-trend. The
extracted barrier Delta_E is REPORTED for context against
Rohart 2016 (Delta_E = 26 +/- 4 meV, tau_0 = 0.22 +/- 0.1 ns at
250 mT) but is NOT gated: Rohart's model is atomistic and the
micromagnetic collapse barrier is grid-dependent, so the
absolute value is not transferable across model classes (see
`rohart_skyrmion`).

Pass criteria (all required):
  - Arrhenius linearity: R^2 of ln(tau) vs 1/T >= `r2_min`.
  - Positive barrier / falling lifetime: fitted Delta_E > 0
    (equivalently tau decreasing with T).
  - Attempt time tau_0 within `tau0_lo`..`tau0_hi` (sub-ns,
    bracketing Rohart's 0.22 ns).
At least `n_T_min` temperatures must yield enough collapse
events (`n_events_min`) for the censored estimator; otherwise
the result is INCONCLUSIVE.

Run with:
    python -m \
    skyrmion_simulator.stochastic_llgs.validation.test_skyrmion_arrhenius
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

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Boltzmann constant in meV/K (for reporting Delta_E in meV).
_KB_MEV = 8.617333262e-2
# Boltzmann constant in J/K.
_KB_J = 1.380649e-23
# Rohart 2016 Fig. 2(c) Langevin anchor (250 mT, 68-101 K).
_ROHART_DE_MEV = 26.0
_ROHART_DE_ERR = 4.0
_ROHART_TAU0_NS = 0.22


def censored_tau(t_collapse, t_max):
    """Censored-exponential MLE of the mean collapse time.

    Trajectories that never collapse are right-censored at
    `t_max`. For a single-rate (exponential) first-passage,
    the MLE is the total observed time divided by the number
    of collapse events.

    Parameters
    ----------
    t_collapse : numpy.ndarray(1d)
        Per-trajectory collapse time in seconds; NaN for
        survivors.
    t_max : float
        Observation-window length in seconds (censoring time).

    Returns
    -------
    tau_hat : float
        MLE mean collapse time in seconds (NaN if no events).
    n_events : int
        Number of trajectories that collapsed.
    n_total : int
        Total trajectories at this temperature.
    """
    t_collapse = np.asarray(t_collapse, dtype=float)
    n_total = int(t_collapse.size)
    collapsed = np.isfinite(t_collapse)
    n_events = int(collapsed.sum())
    if n_events == 0:
        return float('nan'), 0, n_total
    total_time = float(t_collapse[collapsed].sum()) \
        + float((~collapsed).sum()) * float(t_max)
    return total_time / n_events, n_events, n_total


# -----------------------------------------------------------------------------
def main():
    """Load the scan output and gate the Arrhenius fit.

    Builds censored tau(T), fits the Arrhenius law, and gates on
    linearity, barrier sign, and attempt time. Returns True on
    PASS.
    """
    # =========================== User Configuration =========================
    in_dir = ('output/stochastic_llgs/validation/'
              'skyrmion_arrhenius')
    plot_path = ('docs/figures/validation/'
                 'skyrmion_arrhenius.png')
    n_events_min    = 10        # min collapses to trust tau(T)
    n_T_min         = 3         # min usable T for the fit
    r2_min          = 0.90      # Arrhenius linearity gate
    tau0_lo         = 1.0e-12   # sane attempt-time window (s)
    tau0_hi         = 1.0e-8
    # ======================= End User Configuration =========================
    print('test_skyrmion_arrhenius (Neel-Arrhenius collapse '
          'vs Rohart 2016 form):')
    # Prefer the single aggregate NPZ (produced by
    # aggregate_skyrmion_arrhenius, so only one file is pulled);
    # fall back to the per-trajectory glob.
    agg_path = in_dir + '_agg.npz'
    by_T = {}
    t_max = None
    field_mT = None
    if os.path.exists(agg_path):
        z = np.load(agg_path, allow_pickle=True)
        T_arr = np.asarray(z['T_sub'], dtype=float)
        tc_arr = np.asarray(z['t_collapse'], dtype=float)
        t_max = float(z['n_drive']) * float(z['dt'])
        field_mT = float(np.asarray(z['H_ext'])[2]) * 1e3
        for T, tc in zip(T_arr, tc_arr):
            by_T.setdefault(float(T), []).append(float(tc))
        print(f'  using aggregate {agg_path!r} '
              f'({T_arr.size} trajectories)')
    else:
        files = sorted(
            glob.glob(os.path.join(in_dir, 'T*.npz')))
        if not files:
            print(f'  NO scan output in {in_dir!r} (or '
                  f'{agg_path!r}): run scan_skyrmion_arrhenius '
                  f'(cluster) first.')
            print('  status: INCONCLUSIVE (no scan data)')
            return False
        # Group per-trajectory records by temperature.
        for path in files:
            z = np.load(path, allow_pickle=True)
            T = float(z['T_sub'])
            tc = float(z['t_collapse'])
            if field_mT is None:
                field_mT = float(np.asarray(z['H_ext'])[2]) * 1e3
            by_T.setdefault(T, []).append(tc)
            this_tmax = float(z['n_drive']) * float(z['dt'])
            if t_max is None:
                t_max = this_tmax
            elif abs(this_tmax - t_max) > 1e-18:
                raise RuntimeError(
                    f'test_skyrmion_arrhenius: inconsistent '
                    f't_max across records ({this_tmax} vs '
                    f'{t_max}); mixed scan configs in '
                    f'{in_dir!r}.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Censored tau(T) per temperature.
    T_used, tau_used = [], []
    for T in sorted(by_T):
        tau_hat, n_ev, n_tot = censored_tau(
            np.array(by_T[T], dtype=float), t_max)
        usable = n_ev >= n_events_min
        flag = 'use' if usable else 'skip(too few events)'
        tau_str = 'n/a' if not np.isfinite(tau_hat) \
            else f'{tau_hat*1e9:.2f} ns'
        print(f'  T={T:6.1f} K  events={n_ev:4d}/{n_tot:<4d}  '
              f'tau={tau_str:>10s}  [{flag}]')
        if usable and np.isfinite(tau_hat) and tau_hat > 0.0:
            T_used.append(T)
            tau_used.append(tau_hat)
    T_used = np.array(T_used, dtype=float)
    tau_used = np.array(tau_used, dtype=float)
    if T_used.size < n_T_min:
        print(f'  only {T_used.size} usable T (need '
              f'{n_T_min}); widen T range or lengthen the '
              f'observation window.')
        print('  status: INCONCLUSIVE (insufficient Arrhenius '
              'regime)')
        return False
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Arrhenius fit: ln(tau) = ln(tau_0) + (Delta_E / k_B) * (1/T).
    x = 1.0 / T_used
    y = np.log(tau_used)
    slope, intercept = np.polyfit(x, y, 1)
    y_fit = slope * x + intercept
    ss_res = float(np.sum((y - y_fit) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0.0 else 0.0
    dE_J = slope * _KB_J
    dE_meV = dE_J / 1.602176634e-22
    tau0 = math.exp(intercept)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    form_ok = r2 >= r2_min
    barrier_ok = dE_meV > 0.0
    tau0_ok = (tau0_lo <= tau0 <= tau0_hi)
    all_ok = form_ok and barrier_ok and tau0_ok
    print('-' * 60)
    print(f'  Arrhenius fit ({T_used.size} T): R^2={r2:.4f} '
          f'[{"PASS" if form_ok else "FAIL"} >= {r2_min}]')
    print(f'  Delta_E = {dE_meV:.1f} meV  '
          f'[{"PASS" if barrier_ok else "FAIL"} > 0; '
          f'REPORTED ONLY -- not gated, grid-dependent]')
    print(f'  tau_0   = {tau0*1e9:.3f} ns  '
          f'[{"PASS" if tau0_ok else "FAIL"} in '
          f'{tau0_lo*1e9:.3f}-{tau0_hi*1e9:.1f} ns]')
    print(f'  Rohart 2016 (atomistic, 250 mT): '
          f'Delta_E={_ROHART_DE_MEV:.0f}+/-{_ROHART_DE_ERR:.0f} '
          f'meV, tau_0={_ROHART_TAU0_NS:.2f} ns (context).')
    print('  D-trend (tau up with D, Rohart Fig.5b) needs a '
          'D-sweep: not run here (single-D scan).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Arrhenius plot: data + fit, x = 1/(k_B T) in meV^-1.
    fig, ax = plt.subplots(figsize=(5.5, 4.0), dpi=160)
    x_mev = 1.0 / (_KB_MEV * T_used)
    ax.plot(x_mev, np.log(tau_used / 1e-9), 'o',
            label='sim (censored MLE)')
    order = np.argsort(x_mev)
    ax.plot(x_mev[order],
            (y_fit[order] - math.log(1e-9)),
            '-', label=(f'fit: dE={dE_meV:.1f} meV, '
                        f'tau0={tau0*1e9:.2f} ns'))
    ax.set_xlabel(r'$1/(k_B T)$  (meV$^{-1}$)')
    ax.set_ylabel(r'$\ln(\tau / 1\,\mathrm{ns})$')
    field_str = ('?' if field_mT is None
                 else f'{field_mT:.0f}')
    ax.set_title('Skyrmion collapse Arrhenius (single FM layer, '
                 f'{field_str} mT)')
    ax.legend(loc='best', fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  Arrhenius plot: {plot_path}')
    print(f'  status: {"PASS" if all_ok else "FAIL"} '
          f'(form + tau_0 + barrier-sign)')
    return all_ok


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
