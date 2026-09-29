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
# Local
from src.stochastic_llgs.stability import classify_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
_KNOWN_TYPES = (
    'scan_tj',
    'scan_track_width',
    'scan_arrhenius',
    'scan_radius',
    'pair_potential',
)


def _safe_mean_se(xs):
    """Mean and standard error of a 1D float sequence."""
    # No samples: mean and error are both undefined.
    if xs.size == 0:
        return float('nan'), float('nan')
    m = float(np.nanmean(xs))
    # A single sample has a defined mean but no standard error.
    if xs.size <= 1:
        return m, float('nan')
    # Sample standard error: Bessel-corrected std over sqrt(n).
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
    # One dump per ensemble member; T-prefixed by the worker.
    files = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not files:
        raise RuntimeError(
            f'No T*.npz files in {in_dir!r}.'
        )
    # Accumulate raw per-member samples keyed by (T_sub, j) cell.
    by_cell = {}
    for path in files:
        # Never re-ingest a previously written aggregate.
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        T_sub = float(d['T_sub'])
        j = float(d['j_current'])
        cell = (T_sub, j)
        # First member of a cell seeds its accumulator.
        by_cell.setdefault(cell, {
            'alive': 0, 'n': 0,
            'v': [], 'theta': [], 'sigma_y': [],
        })
        rec = by_cell[cell]
        # Every member counts toward the survival denominator.
        rec['n'] += 1
        # Drive statistics are collected from survivors only; the
        # finite guards drop NaN samples.
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
    # Sorted unique axis values define the output grid.
    Ts = np.array(
        sorted({c[0] for c in by_cell}), dtype=float)
    Js = np.array(
        sorted({c[1] for c in by_cell}), dtype=float)
    nT, nJ = Ts.size, Js.size
    # Pre-allocate the (T, j) grids as NaN; empty cells stay NaN.
    P_surv = np.full((nT, nJ), np.nan)
    v_mean = np.full((nT, nJ), np.nan)
    v_se = np.full((nT, nJ), np.nan)
    th_mean = np.full((nT, nJ), np.nan)
    th_se = np.full((nT, nJ), np.nan)
    sy_mean = np.full((nT, nJ), np.nan)
    n_ens = np.zeros((nT, nJ), dtype=np.int64)
    # Reduce each cell's samples to one grid entry.
    for i, T_sub in enumerate(Ts):
        for k, j in enumerate(Js):
            rec = by_cell.get((float(T_sub), float(j)))
            # A grid cell with no member data stays NaN.
            if rec is None:
                continue
            n_ens[i, k] = rec['n']
            # Survival fraction plus reduced drive statistics.
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
def _aggregate_track_width(in_dir):
    """Group `T{T_sub}_j{j}_ens{idx}.npz` files into a
    `(T_sub, j)` grid of ensemble statistics, carrying the
    elliptical axes D_1, D_2 alongside the scan_tj fields.

    Survival is the per-realization field-classifier verdict
    (`classify_field` on the final top-layer m_z snapshot): a
    member survived iff its final configuration is a skyrmion
    ('S' or 'E'). The |Q|-threshold flag `alive_at_end` is NOT
    used -- a melted labyrinth keeps its winding and would count
    as alive under |Q|. All drive statistics (velocity, Hall
    vector, sigma_y, D_1, D_2) are restricted to surviving
    members. Per-cell class counts are recorded.

    The `anim_*.npz` full-field dumps are skipped here; only the
    per-trajectory files are aggregated. The drive-phase second
    half is used as the steady-state window for D_1, D_2.

    Returns
    -------
    payload : dict
        Keys: Ts, Js, n_ens, P_surv, n_S, n_E, n_P, n_L, n_A, n_R,
        v_mean,
        v_se, theta_mean, theta_se, sigma_y_mean, D1_mean, D1_se,
        D2_mean, D2_se, D1r_mean, D1r_se, D2r_mean, D2r_se,
        L_x, L_y.
    """
    # One dump per ensemble member; T-prefixed by the worker.
    files = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not files:
        raise RuntimeError(
            f'No T*.npz files in {in_dir!r}.'
        )
    # Accumulate samples and class counts per (T_sub, j) cell.
    by_cell = {}
    L_x = float('nan')
    L_y = float('nan')
    for path in files:
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        T_sub = float(d['T_sub'])
        j = float(d['j_current'])
        cell = (T_sub, j)
        by_cell.setdefault(cell, {
            'n': 0,
            'codes': {'S': 0, 'E': 0, 'P': 0, 'L': 0, 'A': 0,
                      'R': 0},
            'v': [], 'vx': [], 'vy': [], 'sigma_y': [],
            'D1': [], 'D2': [], 'D1r': [], 'D2r': [],
        })
        rec = by_cell[cell]
        rec['n'] += 1
        # Physical box extent for the downstream length-scale
        # ratios; recorded authoritatively by the worker.
        L_x = float(d['L_x'])
        L_y = float(d['L_y'])
        # Pre-drive (J=0) finite-T equilibrium size, recorded for
        # every trajectory regardless of drive survival.
        if 'D1_relaxed_top' in d.files:
            d1r = float(d['D1_relaxed_top'])
            d2r = float(d['D2_relaxed_top'])
            if np.isfinite(d1r):
                rec['D1r'].append(d1r)
            if np.isfinite(d2r):
                rec['D2r'].append(d2r)
        # Steady-state ellipse axes: mean over the drive-phase
        # second half (skip the initial deformation transient).
        d1 = d['D1_top']
        d2 = d['D2_top']
        half = max(1, d1.size // 2)
        d1_m = float(np.nanmean(d1[half:]))
        d2_m = float(np.nanmean(d2[half:]))
        # Field-classifier survival verdict on the final snapshot.
        if 'mz_final_top' not in d.files:
            raise RuntimeError(
                f'_aggregate_track_width: {path!r} lacks '
                f'mz_final_top; the classifier-based survival '
                f'criterion needs the final field snapshot '
                f'(re-run the campaign with the updated driver).')
        q_abs = abs(float(np.asarray(d['Q'])[-1]))
        code, _m = classify_field(
            np.asarray(d['mz_final_top'], dtype=float),
            q_abs, d1_m, d2_m, L_x)
        rec['codes'][code] += 1
        if code not in ('S', 'E'):
            continue
        v = float(d['velocity'])
        vx = float(d['v_x'])
        vy = float(d['v_y'])
        sy = float(d['sigma_y'])
        if np.isfinite(v):
            rec['v'].append(v)
        # Hall statistics are built from the velocity vector, not
        # from per-member angles (circular data averages wrongly).
        if np.isfinite(vx) and np.isfinite(vy):
            rec['vx'].append(vx)
            rec['vy'].append(vy)
        if np.isfinite(sy):
            rec['sigma_y'].append(sy)
        if np.isfinite(d1_m):
            rec['D1'].append(d1_m)
        if np.isfinite(d2_m):
            rec['D2'].append(d2_m)
    # Sorted unique axis values define the output grid.
    Ts = np.array(
        sorted({c[0] for c in by_cell}), dtype=float)
    Js = np.array(
        sorted({c[1] for c in by_cell}), dtype=float)
    nT, nJ = Ts.size, Js.size
    # Pre-allocate the (T, j) grids as NaN; empty cells stay NaN.
    P_surv = np.full((nT, nJ), np.nan)
    v_mean = np.full((nT, nJ), np.nan)
    v_se = np.full((nT, nJ), np.nan)
    th_mean = np.full((nT, nJ), np.nan)
    th_se = np.full((nT, nJ), np.nan)
    sy_mean = np.full((nT, nJ), np.nan)
    D1_mean = np.full((nT, nJ), np.nan)
    D1_se = np.full((nT, nJ), np.nan)
    D2_mean = np.full((nT, nJ), np.nan)
    D2_se = np.full((nT, nJ), np.nan)
    D1r_mean = np.full((nT, nJ), np.nan)
    D1r_se = np.full((nT, nJ), np.nan)
    D2r_mean = np.full((nT, nJ), np.nan)
    D2r_se = np.full((nT, nJ), np.nan)
    n_ens = np.zeros((nT, nJ), dtype=np.int64)
    # Per-cell counts of each stability class (S/E/P/L/A/R).
    n_cls = {c: np.zeros((nT, nJ), dtype=np.int64)
             for c in ('S', 'E', 'P', 'L', 'A', 'R')}
    # Reduce each cell's samples to one grid entry.
    for i, T_sub in enumerate(Ts):
        for k, j in enumerate(Js):
            rec = by_cell.get((float(T_sub), float(j)))
            # A grid cell with no member data stays NaN.
            if rec is None:
                continue
            n_ens[i, k] = rec['n']
            for c in ('S', 'E', 'P', 'L', 'A', 'R'):
                n_cls[c][i, k] = rec['codes'][c]
            # Every member must land in exactly one class; a code the
            # aggregator does not know would otherwise vanish from the
            # counts while still inflating n_ens.
            n_counted = sum(rec['codes'][c]
                            for c in ('S', 'E', 'P', 'L', 'A', 'R'))
            if n_counted != rec['n']:
                raise RuntimeError(
                    f'aggregate: cell (T={T_sub}, j={j}) has '
                    f'{rec["n"]} members but {n_counted} classified; '
                    f'an unknown class code was returned.')
            # Survival = fraction whose final configuration is a
            # skyrmion (field classifier 'S' or 'E').
            P_surv[i, k] = (rec['codes']['S'] + rec['codes']['E']) \
                / max(rec['n'], 1)
            v_mean[i, k], v_se[i, k] = _safe_mean_se(
                np.array(rec['v']))
            # Cell Hall angle: deflection of the ensemble-mean
            # velocity vector off the drive (x) axis; SE from the
            # per-member deflection angles (range (-90, 90], no
            # branch cut).
            vx_arr = np.array(rec['vx'])
            vy_arr = np.array(rec['vy'])
            if vx_arr.size > 0:
                th_mean[i, k] = float(np.degrees(np.arctan2(
                    np.mean(vy_arr), abs(np.mean(vx_arr)))))
                _, th_se[i, k] = _safe_mean_se(np.degrees(
                    np.arctan2(vy_arr, np.abs(vx_arr))))
            sy_mean[i, k], _ = _safe_mean_se(
                np.array(rec['sigma_y']))
            D1_mean[i, k], D1_se[i, k] = _safe_mean_se(
                np.array(rec['D1']))
            D2_mean[i, k], D2_se[i, k] = _safe_mean_se(
                np.array(rec['D2']))
            D1r_mean[i, k], D1r_se[i, k] = _safe_mean_se(
                np.array(rec['D1r']))
            D2r_mean[i, k], D2r_se[i, k] = _safe_mean_se(
                np.array(rec['D2r']))
    return {
        'Ts': Ts, 'Js': Js, 'n_ens': n_ens,
        'P_surv': P_surv,
        'n_S': n_cls['S'], 'n_E': n_cls['E'],
        'n_P': n_cls['P'],
        'n_L': n_cls['L'], 'n_A': n_cls['A'],
        'n_R': n_cls['R'],
        'v_mean': v_mean, 'v_se': v_se,
        'theta_mean': th_mean, 'theta_se': th_se,
        'sigma_y_mean': sy_mean,
        'D1_mean': D1_mean, 'D1_se': D1_se,
        'D2_mean': D2_mean, 'D2_se': D2_se,
        'D1r_mean': D1r_mean, 'D1r_se': D1r_se,
        'D2r_mean': D2r_mean, 'D2r_se': D2r_se,
        'L_x': float(L_x), 'L_y': float(L_y),
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
    # One dump per ensemble member; T-prefixed by the worker.
    files = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not files:
        raise RuntimeError(
            f'No T*.npz files in {in_dir!r}.'
        )
    # Collect flip times (and censored counts) per temperature.
    by_T = {}
    t_max_seen = None
    for path in files:
        # Never re-ingest a previously written aggregate.
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
        # All members must share one observation horizon t_max,
        # else the censoring correction below is inconsistent.
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
        # flip_index == -1 marks a right-censored (never-flipped)
        # run; otherwise record its sampled flip time.
        if flip_idx == -1:
            by_T[T_sub]['n_censored'] += 1
        else:
            t_flip = float(t_sample[flip_idx])
            by_T[T_sub]['flipped'].append(t_flip)
    # Sorted temperatures index the per-T output arrays.
    T_arr = np.array(sorted(by_T.keys()), dtype=float)
    n_T = T_arr.size
    tau_mle = np.full(n_T, np.nan)
    n_flipped = np.zeros(n_T, dtype=np.int64)
    n_ens = np.zeros(n_T, dtype=np.int64)
    t_obs_by_T = np.empty(n_T, dtype=object)
    # One lifetime estimate per temperature.
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
    # One dump per ensemble member; D-prefixed by the worker.
    files = sorted(glob.glob(os.path.join(in_dir, 'D*.npz')))
    if not files:
        raise RuntimeError(f'No D*.npz files in {in_dir!r}.')
    # Accumulate per-member samples keyed by (D, H_z) cell.
    by_cell = {}
    for path in files:
        # Never re-ingest a previously written aggregate.
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
    # Flat list of populated (D, H_z) cells indexes the outputs.
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
    # Reduce each cell's samples to one entry per output array.
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
    # One dump per member; r-prefixed by the initial separation.
    files = sorted(glob.glob(os.path.join(in_dir, 'r*.npz')))
    if not files:
        raise RuntimeError(
            f'No r*.npz files in {in_dir!r}.'
        )
    # Group the r(t) traces by initial separation r_init.
    by_r0 = {}
    t_sample_ref = None
    for path in files:
        # Never re-ingest a previously written aggregate.
        if os.path.basename(path) == 'aggregate.npz':
            continue
        d = np.load(path, allow_pickle=True)
        r_init = float(d['r_init'])
        t_sample = d['t_sample']
        # Traces must share a common time base to be averaged;
        # the first file sets the reference grid.
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
    # Sorted initial separations index the (r_init, t) grid.
    r0_arr = np.array(sorted(by_r0.keys()), dtype=float)
    n_R = r0_arr.size
    n_T = t_sample_ref.size
    r_mean_grid = np.full((n_R, n_T), np.nan)
    n_alive = np.zeros(n_R, dtype=np.int64)
    n_ens = np.zeros(n_R, dtype=np.int64)
    # Ensemble-average r(t) at each initial separation.
    for i, r_init in enumerate(r0_arr):
        recs = by_r0[float(r_init)]
        n_ens[i] = len(recs)
        n_alive[i] = int(sum(int(r['alive_both']) for r in recs))
        # No surviving pair at this r_init: leave the trace NaN.
        if n_alive[i] == 0:
            continue
        # Average over surviving pairs only.
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
    """Dispatch to the scan-specific aggregator named on the
    command line, print a per-cell summary, and write the single
    `aggregate.npz` its analysis script reads back.
    """
    # First positional argument: the analysis type. Optional second
    # positional argument: an explicit input directory (used for the
    # per-case track-width campaign dirs under
    # .../scan_track_width/campaign/<tag>). Its aggregate.npz is
    # written into that same directory.
    if len(sys.argv) not in (2, 3):
        raise RuntimeError(
            'aggregate_sllg: pass the analysis type as the first '
            f'positional argument (one of {_KNOWN_TYPES}) and, '
            f'optionally, an explicit input directory as the '
            f'second. Got: {sys.argv[1:]!r}.'
        )
    analysis = sys.argv[1].strip()
    # Reject unknown scan names loudly rather than guessing.
    if analysis not in _KNOWN_TYPES:
        raise RuntimeError(
            f'aggregate_sllg: unknown analysis type '
            f'{analysis!r}. Must be one of {_KNOWN_TYPES}.'
        )
    # Inputs live under output/stochastic_llgs/<scan>/, unless an
    # explicit directory is supplied.
    if len(sys.argv) == 3:
        in_dir = sys.argv[2]
    else:
        in_dir = os.path.join(
            'output', 'stochastic_llgs', analysis)
    if not os.path.isdir(in_dir):
        raise RuntimeError(
            f'aggregate_sllg: input directory {in_dir!r} '
            f'does not exist; run the production script '
            f'and pull its outputs first.'
        )
    # Single file the matching analysis script will read back.
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
    elif analysis == 'scan_track_width':
        payload = _aggregate_track_width(in_dir)
        print(
            f'  cells: {payload["Ts"].size} T x '
            f'{payload["Js"].size} j  '
            f'(L_x={payload["L_x"]*1e9:.0f} nm, '
            f'L_y={payload["L_y"]*1e9:.0f} nm, '
            f'total n_ens entries: '
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
