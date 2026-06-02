"""Aggregate the per-trajectory NPZ dumps of one
stochastic-LLGS scan into a single `aggregate.npz`.

Walks `output/stochastic_llgs/<scan>/` for the per-task files
written by the production scripts, groups them by cell, and
computes the ensemble statistics that the downstream analysis
scripts read. Each per-scan plot script (`analyze_sllg_*.py`)
consumes the aggregate file written by this script; no plot
script re-aggregates from raw trajectories.

Usage
-----
The analysis type is passed as the first positional argument
(read directly from `sys.argv[1]`; no argparse). One of
`scan_tj`, `scan_arrhenius`, `scan_radius`, `pair_potential`.
Any other value (or no value) raises `RuntimeError`.

    python -m scripts.aggregate_sllg scan_tj
    python -m scripts.aggregate_sllg scan_arrhenius
    python -m scripts.aggregate_sllg scan_radius
    python -m scripts.aggregate_sllg pair_potential

Output
------
`output/stochastic_llgs/<scan>/aggregate.npz`. The schema is
scan-specific; see `_aggregate_<scan>` below.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import math
import os
import sys
# Third-party
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
_KNOWN_TYPES = (
    'scan_tj',
    'scan_arrhenius',
    'scan_radius',
    'pair_potential',
)


def _safe_mean_se(xs):
    """Mean and standard error of a 1D float sequence."""
    if xs.size == 0:
        return float('nan'), float('nan')
    m = float(np.nanmean(xs))
    if xs.size <= 1:
        return m, float('nan')
    se = float(np.nanstd(xs, ddof=1) / math.sqrt(xs.size))
    return m, se


# -----------------------------------------------------------------------------
def _aggregate_scan_tj(in_dir):
    """Group `T{T_sub}_j{j}_ens{idx}.npz` files into a
    `(T_sub, j)` grid of ensemble statistics.

    Returns
    -------
    payload : dict
        Keys: Ts, Js, n_ens, P_surv, v_mean, v_se,
        theta_mean, theta_se, sigma_y_mean.
    """
    files = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not files:
        raise RuntimeError(
            f'No T*.npz files in {in_dir!r}.'
        )
    by_cell = {}
    for path in files:
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        T_sub = float(d['T_sub'])
        j = float(d['j_current'])
        cell = (T_sub, j)
        by_cell.setdefault(cell, {
            'alive': 0, 'n': 0,
            'v': [], 'theta': [], 'sigma_y': [],
        })
        rec = by_cell[cell]
        rec['n'] += 1
        if bool(d['alive_at_end']):
            rec['alive'] += 1
            v = float(d['velocity'])
            th = float(d['hall_deg'])
            sy = float(d['sigma_y'])
            if np.isfinite(v):
                rec['v'].append(v)
            if np.isfinite(th):
                rec['theta'].append(th)
            if np.isfinite(sy):
                rec['sigma_y'].append(sy)
    Ts = np.array(
        sorted({c[0] for c in by_cell}), dtype=float)
    Js = np.array(
        sorted({c[1] for c in by_cell}), dtype=float)
    nT, nJ = Ts.size, Js.size
    P_surv = np.full((nT, nJ), np.nan)
    v_mean = np.full((nT, nJ), np.nan)
    v_se = np.full((nT, nJ), np.nan)
    th_mean = np.full((nT, nJ), np.nan)
    th_se = np.full((nT, nJ), np.nan)
    sy_mean = np.full((nT, nJ), np.nan)
    n_ens = np.zeros((nT, nJ), dtype=np.int64)
    for i, T_sub in enumerate(Ts):
        for k, j in enumerate(Js):
            rec = by_cell.get((float(T_sub), float(j)))
            if rec is None:
                continue
            n_ens[i, k] = rec['n']
            P_surv[i, k] = rec['alive'] / max(rec['n'], 1)
            v_mean[i, k], v_se[i, k] = _safe_mean_se(
                np.array(rec['v']))
            th_mean[i, k], th_se[i, k] = _safe_mean_se(
                np.array(rec['theta']))
            sy_mean[i, k], _ = _safe_mean_se(
                np.array(rec['sigma_y']))
    return {
        'Ts': Ts, 'Js': Js, 'n_ens': n_ens,
        'P_surv': P_surv,
        'v_mean': v_mean, 'v_se': v_se,
        'theta_mean': th_mean, 'theta_se': th_se,
        'sigma_y_mean': sy_mean,
    }


# -----------------------------------------------------------------------------
def _aggregate_scan_arrhenius(in_dir):
    """Group `T{T_sub}_ens{idx}.npz` files into per-T lifetime
    estimates via the censored MLE.

    Returns
    -------
    payload : dict
        Keys: T_arr, tau_mle, n_flipped, n_ens, t_obs_by_T
        (object array of variable-length flip-time arrays),
        t_max.
    """
    files = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not files:
        raise RuntimeError(
            f'No T*.npz files in {in_dir!r}.'
        )
    by_T = {}
    t_max_seen = None
    for path in files:
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        T_sub = float(d['T_sub'])
        flip_idx = int(d['flip_index'])
        t_sample = d['t_sample']
        if t_sample.size < 1:
            raise RuntimeError(
                f'{path}: t_sample is empty.'
            )
        t_max = float(t_sample[-1])
        if t_max_seen is None:
            t_max_seen = t_max
        elif abs(t_max - t_max_seen) > 1e-3 * t_max_seen:
            raise RuntimeError(
                f'{path}: inconsistent t_max '
                f'({t_max:.3e} vs {t_max_seen:.3e}).'
            )
        by_T.setdefault(T_sub, {
            'flipped': [], 'n_censored': 0,
        })
        if flip_idx == -1:
            by_T[T_sub]['n_censored'] += 1
        else:
            t_flip = float(t_sample[flip_idx])
            by_T[T_sub]['flipped'].append(t_flip)
    T_arr = np.array(sorted(by_T.keys()), dtype=float)
    n_T = T_arr.size
    tau_mle = np.full(n_T, np.nan)
    n_flipped = np.zeros(n_T, dtype=np.int64)
    n_ens = np.zeros(n_T, dtype=np.int64)
    t_obs_by_T = np.empty(n_T, dtype=object)
    for i, T in enumerate(T_arr):
        rec = by_T[T]
        t_obs = np.array(rec['flipped'], dtype=float)
        n_c = int(rec['n_censored'])
        n_flipped[i] = t_obs.size
        n_ens[i] = t_obs.size + n_c
        t_obs_by_T[i] = t_obs
        # Censored-exponential MLE: total observed time (flips
        # + right-censored runs capped at t_max) over flip count.
        if t_obs.size > 0:
            tau_mle[i] = float(
                (t_obs.sum() + n_c * t_max_seen) / t_obs.size
            )
    return {
        'T_arr': T_arr, 'tau_mle': tau_mle,
        'n_flipped': n_flipped, 'n_ens': n_ens,
        't_obs_by_T': t_obs_by_T,
        't_max': float(t_max_seen),
    }


# -----------------------------------------------------------------------------
def _aggregate_scan_radius(in_dir):
    """Group `D{D}_Hz{Hz}_ens{idx}.npz` files into per-cell
    ensemble statistics with inter-layer diagnostics.

    Beyond the top-layer summary, also collects per-cell
    bot-layer diameter, the top-bot center offset (steady
    state), and Q_top + Q_bot (compensation residual).
    """
    files = sorted(glob.glob(os.path.join(in_dir, 'D*.npz')))
    if not files:
        raise RuntimeError(f'No D*.npz files in {in_dir!r}.')
    by_cell = {}
    for path in files:
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        D = float(d['D'])
        H_z = float(d['H_z'])
        cell = (D, H_z)
        by_cell.setdefault(cell, {
            'alive': 0, 'n': 0,
            'v': [], 'theta': [], 'd': [],
            # Inter-layer per-trajectory statistics
            'd_bot': [], 'offset': [], 'q_total': [],
        })
        rec = by_cell[cell]
        rec['n'] += 1
        if not bool(d['alive_at_end']):
            continue
        rec['alive'] += 1
        # Top: drift, Hall angle, steady-state diameter.
        v = float(d['velocity'])
        th = float(d['hall_deg'])
        # Prefer LCC diameter (single-skyrmion-area, no noise
        # blob contamination); fall back to all-mask diameter
        # for NPZs written before the LCC enrichment.
        diam = d['diameter_lcc'] if 'diameter_lcc' in d.files \
            else d['diameter']
        # Average over the second half of the trace as the
        # steady-state window (skip the initial transient).
        half = max(1, diam.size // 2)
        d_t_mean = float(np.nanmean(diam[half:]))
        if np.isfinite(v):
            rec['v'].append(v)
        if np.isfinite(th):
            rec['theta'].append(th)
        if np.isfinite(d_t_mean):
            rec['d'].append(d_t_mean)
        # Bot: same LCC-with-fallback choice.
        diam_bot = d['diameter_lcc_bot'] \
            if 'diameter_lcc_bot' in d.files \
            else d['diameter_bot']
        d_b_mean = float(np.nanmean(diam_bot[half:]))
        if np.isfinite(d_b_mean):
            rec['d_bot'].append(d_b_mean)
        # Top-bot center offset (steady state, unwrapped).
        # Use LCC centers when present.
        if 'cx_lcc' in d.files and 'cx_lcc_bot' in d.files:
            cx_t_w = d['cx_lcc']
            cy_t_w = d['cy_lcc']
            cx_b_w = d['cx_lcc_bot']
            cy_b_w = d['cy_lcc_bot']
        else:
            cx_t_w = d['cx_unwrapped']
            cy_t_w = d['cy_unwrapped']
            cx_b_w = d['cx_unwrapped_bot']
            cy_b_w = d['cy_unwrapped_bot']
        with np.errstate(invalid='ignore'):
            dx = cx_t_w[half:] - cx_b_w[half:]
            dy = cy_t_w[half:] - cy_b_w[half:]
            offset = np.sqrt(dx ** 2 + dy ** 2)
            off_mean = float(np.nanmean(offset))
        if np.isfinite(off_mean):
            rec['offset'].append(off_mean)
        # Compensation residual Q_top + Q_bot (~0 if locked).
        Q_t = d['Q']
        Q_b = d['Q_bot']
        with np.errstate(invalid='ignore'):
            q_tot = float(np.nanmean(
                Q_t[half:] + Q_b[half:]))
        if np.isfinite(q_tot):
            rec['q_total'].append(q_tot)
    cells = sorted(by_cell.keys())
    n_C = len(cells)
    D_arr = np.array([c[0] for c in cells], dtype=float)
    Hz_arr = np.array([c[1] for c in cells], dtype=float)
    p_surv = np.zeros(n_C)
    n_ens = np.zeros(n_C, dtype=np.int64)
    d_mean = np.full(n_C, np.nan)
    d_se = np.full(n_C, np.nan)
    v_mean = np.full(n_C, np.nan)
    v_se = np.full(n_C, np.nan)
    th_mean = np.full(n_C, np.nan)
    th_se = np.full(n_C, np.nan)
    d_bot_mean = np.full(n_C, np.nan)
    d_bot_se = np.full(n_C, np.nan)
    offset_mean = np.full(n_C, np.nan)
    offset_se = np.full(n_C, np.nan)
    q_total_mean = np.full(n_C, np.nan)
    q_total_se = np.full(n_C, np.nan)
    for i, cell in enumerate(cells):
        rec = by_cell[cell]
        n_ens[i] = rec['n']
        p_surv[i] = rec['alive'] / max(rec['n'], 1)
        d_mean[i], d_se[i] = _safe_mean_se(np.array(rec['d']))
        v_mean[i], v_se[i] = _safe_mean_se(np.array(rec['v']))
        th_mean[i], th_se[i] = _safe_mean_se(
            np.array(rec['theta']))
        d_bot_mean[i], d_bot_se[i] = _safe_mean_se(
            np.array(rec['d_bot']))
        offset_mean[i], offset_se[i] = _safe_mean_se(
            np.array(rec['offset']))
        q_total_mean[i], q_total_se[i] = _safe_mean_se(
            np.array(rec['q_total']))
    return {
        'D_arr': D_arr, 'Hz_arr': Hz_arr,
        'n_ens': n_ens, 'p_surv': p_surv,
        # Top-layer
        'd_mean': d_mean, 'd_se': d_se,
        'v_mean': v_mean, 'v_se': v_se,
        'theta_mean': th_mean, 'theta_se': th_se,
        # Bot-layer + inter-layer diagnostics
        'd_bot_mean': d_bot_mean, 'd_bot_se': d_bot_se,
        'offset_mean': offset_mean, 'offset_se': offset_se,
        'q_total_mean': q_total_mean,
        'q_total_se': q_total_se,
    }


# -----------------------------------------------------------------------------
def _aggregate_pair_potential(in_dir):
    """Group `r{r_init}_ens{idx}.npz` files into per-r_init
    ensemble-averaged r(t) traces.
    """
    files = sorted(glob.glob(os.path.join(in_dir, 'r*.npz')))
    if not files:
        raise RuntimeError(
            f'No r*.npz files in {in_dir!r}.'
        )
    by_r0 = {}
    t_sample_ref = None
    for path in files:
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        r_init = float(d['r_init'])
        t_sample = d['t_sample']
        if t_sample_ref is None:
            t_sample_ref = np.asarray(t_sample, dtype=float)
        elif (t_sample.shape != t_sample_ref.shape
              or not np.allclose(t_sample, t_sample_ref)):
            raise RuntimeError(
                f'{path}: t_sample mismatch across '
                f'trajectories.'
            )
        by_r0.setdefault(r_init, []).append({
            'r_pair': np.asarray(d['r_pair'], dtype=float),
            'alive_both': bool(d['alive_both']),
        })
    r0_arr = np.array(sorted(by_r0.keys()), dtype=float)
    n_R = r0_arr.size
    n_T = t_sample_ref.size
    r_mean_grid = np.full((n_R, n_T), np.nan)
    n_alive = np.zeros(n_R, dtype=np.int64)
    n_ens = np.zeros(n_R, dtype=np.int64)
    for i, r_init in enumerate(r0_arr):
        recs = by_r0[float(r_init)]
        n_ens[i] = len(recs)
        n_alive[i] = int(sum(int(r['alive_both']) for r in recs))
        if n_alive[i] == 0:
            continue
        stack = np.array(
            [r['r_pair'] for r in recs], dtype=float)
        with np.errstate(invalid='ignore'):
            r_mean_grid[i, :] = np.nanmean(stack, axis=0)
    return {
        'r0_arr': r0_arr,
        't_sample': t_sample_ref,
        'r_mean_grid': r_mean_grid,
        'n_alive': n_alive, 'n_ens': n_ens,
    }


# -----------------------------------------------------------------------------
def main():
    if len(sys.argv) != 2:
        raise RuntimeError(
            'aggregate_sllg: pass the analysis type as the '
            f'single positional argument. One of '
            f'{_KNOWN_TYPES}. Got: {sys.argv[1:]!r}.'
        )
    analysis = sys.argv[1].strip()
    if analysis not in _KNOWN_TYPES:
        raise RuntimeError(
            f'aggregate_sllg: unknown analysis type '
            f'{analysis!r}. Must be one of {_KNOWN_TYPES}.'
        )
    in_dir = os.path.join(
        'output', 'stochastic_llgs', analysis)
    if not os.path.isdir(in_dir):
        raise RuntimeError(
            f'aggregate_sllg: input directory {in_dir!r} '
            f'does not exist; run the production script '
            f'and pull its outputs first.'
        )
    out_path = os.path.join(in_dir, 'aggregate.npz')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Dispatch to the scan-specific aggregator and echo a
    # per-cell summary as it builds the payload.
    print(f'aggregate_sllg: {analysis}')
    print('-' * 56)
    if analysis == 'scan_tj':
        payload = _aggregate_scan_tj(in_dir)
        print(
            f'  cells: {payload["Ts"].size} T x '
            f'{payload["Js"].size} j  '
            f'(total n_ens entries: '
            f'{int(payload["n_ens"].sum())})'
        )
    elif analysis == 'scan_arrhenius':
        payload = _aggregate_scan_arrhenius(in_dir)
        for T, tau, n_f, n_e in zip(
                payload['T_arr'], payload['tau_mle'],
                payload['n_flipped'], payload['n_ens']):
            print(
                f'  T={T:5.1f} K  n_ens={int(n_e):3d}  '
                f'flipped={int(n_f):3d}  tau={tau:.3e} s'
            )
    elif analysis == 'scan_radius':
        payload = _aggregate_scan_radius(in_dir)
        for D, Hz, ne, ps, dm, dbm, om, qtm in zip(
                payload['D_arr'], payload['Hz_arr'],
                payload['n_ens'], payload['p_surv'],
                payload['d_mean'], payload['d_bot_mean'],
                payload['offset_mean'],
                payload['q_total_mean']):
            print(
                f'  D={D:.2e}  H_z={Hz:+.3f}  '
                f'n={int(ne):3d}  P_surv={ps:.2f}  '
                f'<d_top>={dm*1e9:5.1f} nm  '
                f'<d_bot>={dbm*1e9:5.1f} nm  '
                f'<|r_t-r_b|>={om*1e9:5.2f} nm  '
                f'<Q_t+Q_b>={qtm:+.4f}'
            )
    else:
        payload = _aggregate_pair_potential(in_dir)
        for r0, na, ne in zip(
                payload['r0_arr'], payload['n_alive'],
                payload['n_ens']):
            print(
                f'  r_init={r0*1e9:5.1f} nm  '
                f'n_alive={int(na):3d}/{int(ne):3d}'
            )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write the single aggregate the analysis script reads.
    np.savez_compressed(out_path, **payload)
    print(f'Saved {out_path}')


# =============================================================================
if __name__ == '__main__':
    main()
