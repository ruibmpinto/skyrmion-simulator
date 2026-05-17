"""Combine SLURM-array partial NPZs into the final NPZ.

Reads every `part_*.npz` written by `sweep_array_partial`,
fills the aggregated `(n_D, n_H, n_IC)` arrays expected by
the plotting code, selects the lowest-energy converged IC
at each `(D, H)` point as the ground state, and writes the
result as a single NPZ matching the schema produced by
`src.phase_diagram.sweep.sweep`.

Configuration lives at the top of `main()`. Edit the
variables in that block to set the grid, lattice size,
partial directory, and output path, then run:

    python -m src.phase_diagram.aggregate

Functions
---------
aggregate
    Combine partials in `partial_dir` into the final NPZ at
    `out_path`. Pure-Python entry point.
main
    Read the User Configuration block and call `aggregate`.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
import sys
# Third-party
import numpy as np
# Local
from src.phase_diagram.classifier import PHASE_LABELS
from src.phase_diagram.params_helper import make_params
from src.phase_diagram.sweep import (
    _allocate_arrays,
    _grid,
    _ic_specs,
)
from src.simulator.energy import (
    critical_dmi,
    effective_anisotropy,
    pma_anisotropy_field,
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
_PHASE_LABELS = PHASE_LABELS


def aggregate(grid_name, nx, ny, partial_dir, out_path,
              require_all_records,
              K_top=None, K_bot=None):
    """Combine partial NPZs in `partial_dir` into `out_path`.

    Parameters
    ----------
    grid_name : str
        Grid name (must match the array job).
    nx : {int, None}
        Lattice size along x; None for namespace default.
    ny : {int, None}
        Lattice size along y; None for namespace default.
    partial_dir : str
        Directory holding `part_*.npz` files.
    out_path : str
        Output NPZ path.
    require_all_records : bool
        If True, raise when any `(i, j, k)` record is
        missing across the partials. If False, warn and
        leave missing entries at their allocation defaults.
    K_top, K_bot : {float, None}, default=None
        Per-layer raw anisotropy overrides in J/m^3 (must
        match the array job). `None` keeps the namespace
        defaults.
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Replicate the grid axes and allocate aggregation
    # arrays exactly as the local sweep would.
    D_arr, H_arr = _grid(grid_name)
    n_D, n_H = len(D_arr), len(H_arr)
    n_IC = len(_ic_specs())
    probe_kwargs = {}
    if nx is not None:
        probe_kwargs['nx'] = int(nx)
    if ny is not None:
        probe_kwargs['ny'] = int(ny)
    if K_top is not None:
        probe_kwargs['K_top'] = float(K_top)
    if K_bot is not None:
        probe_kwargs['K_bot'] = float(K_bot)
    p_probe = make_params(**probe_kwargs)
    arr = _allocate_arrays(n_D, n_H, n_IC, p_probe.ny, p_probe.nx)
    best_E = np.full((n_D, n_H), np.inf)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    parts = sorted(
        glob.glob(os.path.join(partial_dir, 'part_*.npz'))
    )
    if not parts:
        raise RuntimeError(
            f'No partial NPZs found in {partial_dir}.'
        )
    # Track which (i, j, k) records have been seen so we can
    # detect duplicates and missing entries.
    seen = np.zeros((n_D, n_H, n_IC), dtype=bool)
    n_records = 0
    for part_path in parts:
        with np.load(part_path, allow_pickle=True) as d:
            n_rec = int(d['i'].size)
            # Sanity-check that the grid axes match
            if d['D_grid'].shape != D_arr.shape \
                    or not np.allclose(d['D_grid'], D_arr):
                raise RuntimeError(
                    f'D-axis mismatch in {part_path}: '
                    f'expected {D_arr}, got {d["D_grid"]}.'
                )
            if d['H_z_grid'].shape != H_arr.shape \
                    or not np.allclose(d['H_z_grid'], H_arr):
                raise RuntimeError(
                    f'H_z-axis mismatch in {part_path}: '
                    f'expected {H_arr}, got '
                    f'{d["H_z_grid"]}.'
                )
            for r in range(n_rec):
                i = int(d['i'][r])
                j = int(d['j'][r])
                k = int(d['k'][r])
                if seen[i, j, k]:
                    raise RuntimeError(
                        f'Duplicate record for (i={i}, j={j}, '
                        f'k={k}) in {part_path}.'
                    )
                seen[i, j, k] = True
                arr['E'][i, j, k] = float(d['E'][r])
                arr['Q'][i, j, k] = float(d['Q'][r])
                arr['mz_top'][i, j, k] = float(d['mz_top'][r])
                arr['mz_bot'][i, j, k] = float(d['mz_bot'][r])
                arr['m_dot'][i, j, k] = float(d['m_dot'][r])
                arr['k_star'][i, j, k] = float(d['k_star'][r])
                arr['P_2'][i, j, k] = float(d['P_2'][r])
                arr['P_6'][i, j, k] = float(d['P_6'][r])
                arr['P_iso'][i, j, k] = float(d['P_iso'][r])
                arr['peak_over_bg'][i, j, k] = float(
                    d['peak_over_bg'][r]
                )
                arr['n_periods'][i, j, k] = float(
                    d['n_periods'][r]
                )
                arr['q_per_period'][i, j, k] = float(
                    d['q_per_period'][r]
                )
                arr['tau_max'][i, j, k] = float(d['tau_max'][r])
                arr['n_steps'][i, j, k] = int(d['n_steps'][r])
                conv = bool(d['converged'][r])
                arr['converged'][i, j, k] = conv
                lab = str(d['label'][r])
                arr['label_idx_per_ic'][i, j, k] = (
                    _PHASE_LABELS.index(lab)
                )
                # Ground-state update: lowest-energy converged
                # IC wins at each (i, j).
                if conv and float(d['E'][r]) < best_E[i, j]:
                    best_E[i, j] = float(d['E'][r])
                    arr['gs_idx'][i, j] = k
                    arr['gs_label_idx'][i, j] = (
                        _PHASE_LABELS.index(lab)
                    )
                    arr['gs_m_top'][i, j] = d['m_top'][r]
                    arr['gs_m_bot'][i, j] = d['m_bot'][r]
                n_records += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_expected = n_D * n_H * n_IC
    n_missing = int((~seen).sum())
    if n_missing > 0:
        miss_ijk = np.argwhere(~seen)
        msg = (
            f'{n_missing} of {n_expected} (i, j, k) records '
            f'missing across {len(parts)} partials. First '
            f'missing: {miss_ijk[:5].tolist()}.'
        )
        if require_all_records:
            raise RuntimeError(msg)
        else:
            print(f'WARNING: {msg}', flush=True)
    if n_records != n_expected - n_missing:
        raise RuntimeError(
            f'Internal accounting error: '
            f'n_records={n_records}, expected '
            f'{n_expected - n_missing}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reduced-unit normalizations (material-only).
    K_eff_dict = effective_anisotropy(p_probe)
    D_c = critical_dmi(p_probe)
    H_K = pma_anisotropy_field(p_probe)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        D=D_arr, H_z=H_arr,
        ic_names=np.array(
            [s[0] for s in _ic_specs()], dtype=object,
        ),
        labels=np.array(_PHASE_LABELS, dtype=object),
        D_c=np.float64(D_c),
        H_K=np.float64(H_K),
        K_top_raw=np.float64(p_probe.K_top),
        K_bot_raw=np.float64(p_probe.K_bot),
        K_eff_top=np.float64(K_eff_dict['K_eff_top']),
        K_eff_bot=np.float64(K_eff_dict['K_eff_bot']),
        K_eff_avg=np.float64(K_eff_dict['K_eff_avg']),
        **{k: v for k, v in arr.items()},
    )
    print(
        f'Wrote {out_path} '
        f'({n_records} records from {len(parts)} partials, '
        f'{n_missing} missing).'
    )


# -----------------------------------------------------------------------------
def main():
    """Read the User Configuration block and call `aggregate`.

    Edit the variables in the block below to match the
    array-job settings (`grid_name`, `nx`, `ny`).
    """
    # ================ User Configuration ================
    # Must match the grid used by the array job.
    grid_name = 'medium'
    # Must match the lattice size used by the array job.
    # None means "use the namespace default".
    nx = 128
    ny = 128
    # Directory containing the per-element partial NPZs.
    # None falls back to output/phase_diagram/<grid>_partials.
    partial_dir = None
    # Output NPZ. None falls back to
    # output/phase_diagram/<grid>.npz.
    out_path = None
    # If True, raise on any missing (i, j, k) record; if
    # False, leave the missing entries at their allocation
    # defaults (NaN for floats, -1 for labels) and warn.
    require_all_records = True
    # Material overrides; must match the array-job
    # configuration. Use `Q_PMA` to compute K from a
    # quality factor (mutually exclusive with explicit K).
    K_top = None
    K_bot = None
    Q_PMA = None
    # ============ End User Configuration =================
    if Q_PMA is not None:
        if K_top is not None or K_bot is not None:
            raise RuntimeError(
                'Set either Q_PMA or explicit K_top/K_bot, '
                'not both.'
            )
        p_probe = make_params()
        half_mu0_Ms2 = effective_anisotropy(p_probe)[
            'mu0_Ms2_over_2'
        ]
        K_resolved = float(Q_PMA) * half_mu0_Ms2 + half_mu0_Ms2
        K_top = K_resolved
        K_bot = K_resolved
    if partial_dir is None:
        partial_dir = os.path.join(
            'output', 'phase_diagram',
            f'{grid_name}_partials',
        )
    if out_path is None:
        out_path = os.path.join(
            'output', 'phase_diagram', f'{grid_name}.npz',
        )
    aggregate(
        grid_name=grid_name, nx=nx, ny=ny,
        partial_dir=partial_dir, out_path=out_path,
        require_all_records=require_all_records,
        K_top=K_top, K_bot=K_bot,
    )


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
