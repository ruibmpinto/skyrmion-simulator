"""Finite-T skyrmion-size gate (Tomasello 2018 Fig. 1b, #9b).

Post-processing on `scan_skyrmion_size_thermal`. Pools the
per-trajectory R_sk samples to a mean +/- total std per
(H, T), gates the H = 0 expansion ratio against the paper
(<R_sk>(300 K)/<R_sk>(0) ~ 3.3), and overlays the digitized
Fig. 1(b). This is the stochastic counterpart of #9a; the
gated scalar is the same H = 0 expansion ratio.

The total std per (H, T) combines the within-trajectory
thermal fluctuation (mean of the per-seed variances) and the
between-seed spread of the means, so the error bars reflect
both sources -- matching the paper's mean +/- std symbols.

Run with:
    python -m src.phase_diagram.validation.test_skyrmion_size_thermal
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import matplotlib.pyplot as plt
import numpy as np
# Local
from src.phase_diagram.validation.test_skyrmion_size_vs_T_tomasello2018 import (
    _DSK_H0,
    _DSK_H25,
    _DSK_H50,
    _T_REF,
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


def _pool(by_seed):
    """Pool per-seed (mean, std) into a grand mean and a total
    std (within-trajectory + between-seed)."""
    means = np.array([m for m, _s in by_seed], dtype=float)
    stds = np.array([s for _m, s in by_seed], dtype=float)
    grand_mean = float(np.mean(means))
    within = float(np.mean(stds ** 2))
    between = float(np.var(means)) if means.size > 1 else 0.0
    return grand_mean, float(np.sqrt(within + between))


def main():
    # Pool the thermal scan, gate the H=0 expansion ratio, and
    # overlay the digitized Fig. 1(b) symbols.
    # =========================== User Configuration =========================
    in_dir = ('output/phase_diagram/validation/'
              'skyrmion_size_thermal')
    plot_path = ('docs/figures/validation/'
                 'tomasello_size_vs_T_thermal.png')
    ratio_ref = _DSK_H0[-1] / _DSK_H0[0]    # ~3.3 (Fig 1b H=0)
    ratio_rtol = 0.30
    # ======================= End User Configuration =========================
    print('test_skyrmion_size_thermal (Tomasello 2018 Fig. 1b, '
          'finite-T symbols):')
    agg_path = in_dir + '_agg.npz'
    # records[(H, T)] = list of (R_sk_mean, R_sk_std) per seed.
    records = {}
    # Prefer the single aggregate NPZ; else fall back to the
    # per-trajectory files; else report INCONCLUSIVE.
    if os.path.exists(agg_path):
        z = np.load(agg_path, allow_pickle=True)
        Hs = np.asarray(z['H'], dtype=float)
        Ts = np.asarray(z['T'], dtype=float)
        mu = np.asarray(z['R_sk_mean'], dtype=float)
        sd = np.asarray(z['R_sk_std'], dtype=float)
        for H, T, m, s in zip(Hs, Ts, mu, sd):
            records.setdefault((float(H), float(T)), []).append(
                (float(m), float(s)))
        print(f'  using aggregate {agg_path!r} '
              f'({Hs.size} trajectories)')
    else:
        files = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
        if not files:
            print(f'  NO scan output in {in_dir!r} (or '
                  f'{agg_path!r}): run the thermal scan first.')
            print('  status: INCONCLUSIVE (no scan data)')
            return False
        for path in files:
            z = np.load(path, allow_pickle=True)
            key = (float(z['H']), float(z['T']))
            records.setdefault(key, []).append(
                (float(z['R_sk_mean']), float(z['R_sk_std'])))
        print(f'  using {len(files)} per-trajectory files')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # <R_sk> +/- total std per (H, T).
    Hs = sorted({H for (H, _T) in records})
    curves = {}
    for H in Hs:
        Ts = sorted(T for (HH, T) in records if HH == H)
        mean_T, std_T = [], []
        for T in Ts:
            gm, gs = _pool(records[(H, T)])
            mean_T.append(gm)
            std_T.append(gs)
        curves[H] = (np.array(Ts), np.array(mean_T),
                     np.array(std_T))
        print(f'  --- H = {H*1e3:.0f} mT ---')
        for T, gm, gs in zip(Ts, mean_T, std_T):
            print(f'    T={T:6.1f} K  <R_sk>={gm*1e9:6.1f} '
                  f'+/- {gs*1e9:4.1f} nm  '
                  f'(D_sk={2*gm*1e9:5.1f} nm)')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Gate: H = 0 expansion ratio <R_sk>(300)/<R_sk>(0).
    if 0.0 not in curves:
        print('  status: INCONCLUSIVE (no H=0 data)')
        return False
    T0, mean0, _s0 = curves[0.0]
    if not (mean0[0] > 0.0 and mean0[-1] > 0.0):
        print('  status: FAIL (no H=0 endpoint radii)')
        return False
    ratio = mean0[-1] / mean0[0]
    rel = abs(ratio - ratio_ref) / ratio_ref
    passed = rel < ratio_rtol
    print('-' * 60)
    print(f'  H=0 expansion ratio <R_sk>({T0[-1]:.0f})/'
          f'<R_sk>({T0[0]:.0f}) = {ratio:.2f} '
          f'(paper {ratio_ref:.2f}, rel.err {rel*100:.0f}%) '
          f'[{"PASS" if passed else "FAIL"} < '
          f'{ratio_rtol*100:.0f}%]')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    refs = {0.0: _DSK_H0, 25.0e-3: _DSK_H25, 50.0e-3: _DSK_H50}
    cols = {0.0: 'k', 25.0e-3: 'tab:red', 50.0e-3: 'tab:green'}
    fig, ax = plt.subplots(figsize=(6.0, 4.4), dpi=160)
    for H in Hs:
        Ts, mean_T, std_T = curves[H]
        c = cols.get(H, 'tab:blue')
        ax.errorbar(Ts, 2.0 * mean_T * 1e9, yerr=2.0 * std_T * 1e9,
                    fmt='o', color=c, capsize=3,
                    label=f'sim H={H*1e3:.0f} mT')
        if H in refs:
            ax.plot(_T_REF, refs[H], '--', color=c, alpha=0.6,
                    label=f'paper H={H*1e3:.0f} mT')
    ax.set_xlabel('T (K)')
    ax.set_ylabel(r'$D_{sk}$ (nm)')
    ax.set_title('Skyrmion size vs T (Tomasello 2018 Fig. 1b, '
                 'thermal)')
    ax.legend(fontsize=7, frameon=False, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(plot_path), exist_ok=True)
    fig.savefig(plot_path)
    plt.close(fig)
    print(f'  figure: {plot_path}')
    print(f'  status: {"PASS" if passed else "FAIL"}')
    return passed


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
