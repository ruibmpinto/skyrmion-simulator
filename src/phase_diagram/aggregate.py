"""Combine SLURM-array partial NPZs into the final NPZ.

Reads every `part_*.npz` written by
`sweep.sweep_array_partial`, fills the aggregated
`(n_x, n_y, n_ic)` arrays expected by the plotting code,
selects the lowest-energy converged IC at each
`(axis_x, axis_y)` point as the ground state, and writes
the result as a single NPZ matching the schema produced by
`src.phase_diagram.sweep.sweep`.

Strict mode: any partial missing a required key, any
inconsistency between partials on axes / IC list /
material scalars, and any duplicate `(i, j, k)` records
all raise `RuntimeError`. There are no legacy schema
fallbacks.

Configuration lives at the top of `main()`. Edit the
variables in that block to set the partial directory and
output path, then run:

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

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
_phase_labels = PHASE_LABELS

# Required keys in each partial NPZ. Missing key -> RuntimeError.
_partial_required_keys = (
    # axis schema
    'axis_x_name', 'axis_x_grid',
    'axis_y_name', 'axis_y_grid',
    # complete IC ensemble (every partial carries the full
    # list so n_ic is fixed before scanning all records)
    'ic_names_full', 'ic_seeds_full',
    # material scalars (axis-invariant)
    'Ms', 'A_ex', 'mu0', 'half_mu0_Ms2', 'H_RKKY',
    'K_top_probe', 'K_bot_probe',
    't_Co', 'd_Ru', 'a',
    # per-task records
    'i', 'j', 'k',
    'axis_x_value', 'axis_y_value',
    'ic_name', 'ic_seed',
    'converged', 'n_steps', 'E', 'tau_max', 'label',
    'mz_top', 'mz_bot', 'm_dot', 'Q',
    'k_star', 'P_2', 'P_6', 'P_iso',
    'peak_over_bg', 'n_periods', 'q_per_period',
    'm_top', 'm_bot',
)

# Material-scalar keys that must agree across partials.
_material_scalar_keys = (
    'Ms', 'A_ex', 'mu0', 'half_mu0_Ms2', 'H_RKKY',
    'K_top_probe', 'K_bot_probe', 't_Co', 'd_Ru', 'a',
)


def _check_required(npz, path):
    """Raise if any required key is missing from `npz`."""
    files = set(npz.files)
    missing = set(_partial_required_keys) - files
    if missing:
        raise RuntimeError(
            f'Partial {path!r} is missing required keys: '
            f'{sorted(missing)}.'
        )


def _ic_list_from(npz):
    """Return the canonical IC list saved by the sweep.

    Each partial NPZ stores the full IC ensemble (names +
    seeds) under `ic_names_full` and `ic_seeds_full`. The
    aggregator uses this to size the output arrays before
    iterating records.
    """
    names = list(npz['ic_names_full'])
    seeds = np.asarray(npz['ic_seeds_full'])
    return [
        (str(name), int(seed))
        for name, seed in zip(names, seeds)
    ]


def aggregate(partial_dir, out_path,
              require_all_records):
    """Combine partial NPZs in `partial_dir` into `out_path`.

    Parameters
    ----------
    partial_dir : str
        Directory holding `part_*.npz` files. Must contain
        at least one partial.
    out_path : str
        Output NPZ path.
    require_all_records : bool
        If True, raise when any `(i, j, k)` record is
        missing across the partials. If False, leave
        missing entries at their allocation defaults (NaN
        floats, -1 labels) and report the count.
    """
    parts = sorted(
        glob.glob(os.path.join(partial_dir, 'part_*.npz'))
    )
    if not parts:
        raise RuntimeError(
            f'No partial NPZs found in {partial_dir!r}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # First partial sets the canonical axes / IC list /
    # material scalars. Every subsequent partial must
    # agree exactly.
    first_path = parts[0]
    with np.load(first_path, allow_pickle=True) as first:
        _check_required(first, first_path)
        axis_x_name = str(first['axis_x_name'])
        axis_y_name = str(first['axis_y_name'])
        axis_x_grid = np.asarray(first['axis_x_grid'])
        axis_y_grid = np.asarray(first['axis_y_grid'])
        ic_list_first = _ic_list_from(first)
        material = {
            key: np.float64(first[key])
            for key in _material_scalar_keys
        }
        # Pick lattice size from any texture in the first
        # partial; all texture shapes must match exactly.
        m_top_first = np.asarray(first['m_top'])
        if m_top_first.ndim != 4:
            raise RuntimeError(
                f'{first_path!r}: m_top has rank '
                f'{m_top_first.ndim}, expected 4 '
                f'(records, ny, nx, 3).'
            )
        ny = int(m_top_first.shape[1])
        nx = int(m_top_first.shape[2])
    n_x = int(len(axis_x_grid))
    n_y = int(len(axis_y_grid))
    n_ic = len(ic_list_first)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    arr = _allocate_arrays(n_x, n_y, n_ic, ny, nx)
    best_E = np.full((n_x, n_y), np.inf)
    seen = np.zeros((n_x, n_y, n_ic), dtype=bool)
    n_records = 0
    for part_path in parts:
        with np.load(part_path, allow_pickle=True) as d:
            _check_required(d, part_path)
            # Cross-partial consistency.
            if str(d['axis_x_name']) != axis_x_name:
                raise RuntimeError(
                    f'{part_path!r}: axis_x_name '
                    f'{str(d["axis_x_name"])!r} != '
                    f'{axis_x_name!r}.'
                )
            if str(d['axis_y_name']) != axis_y_name:
                raise RuntimeError(
                    f'{part_path!r}: axis_y_name '
                    f'{str(d["axis_y_name"])!r} != '
                    f'{axis_y_name!r}.'
                )
            if not np.allclose(d['axis_x_grid'], axis_x_grid):
                raise RuntimeError(
                    f'{part_path!r}: axis_x_grid disagrees '
                    f'with the first partial.'
                )
            if not np.allclose(d['axis_y_grid'], axis_y_grid):
                raise RuntimeError(
                    f'{part_path!r}: axis_y_grid disagrees '
                    f'with the first partial.'
                )
            ic_list_this = _ic_list_from(d)
            if ic_list_this != ic_list_first:
                raise RuntimeError(
                    f'{part_path!r}: IC list disagrees '
                    f'with first partial. Got '
                    f'{ic_list_this}, expected '
                    f'{ic_list_first}.'
                )
            for key in _material_scalar_keys:
                v = float(d[key])
                if not np.isclose(v, float(material[key])):
                    raise RuntimeError(
                        f'{part_path!r}: {key} = {v} '
                        f'disagrees with first partial '
                        f'{float(material[key])}.'
                    )
            m_top_d = np.asarray(d['m_top'])
            if m_top_d.shape[1] != ny or m_top_d.shape[2] != nx:
                raise RuntimeError(
                    f'{part_path!r}: lattice '
                    f'{m_top_d.shape[1:3]} != ({ny}, {nx}).'
                )
            n_rec = int(np.asarray(d['i']).size)
            for r in range(n_rec):
                i = int(d['i'][r])
                j = int(d['j'][r])
                k = int(d['k'][r])
                if seen[i, j, k]:
                    raise RuntimeError(
                        f'Duplicate record for '
                        f'(i={i}, j={j}, k={k}) in '
                        f'{part_path!r}.'
                    )
                seen[i, j, k] = True
                arr['E'][i, j, k] = float(d['E'][r])
                arr['Q'][i, j, k] = float(d['Q'][r])
                arr['mz_top'][i, j, k] = float(
                    d['mz_top'][r]
                )
                arr['mz_bot'][i, j, k] = float(
                    d['mz_bot'][r]
                )
                arr['m_dot'][i, j, k] = float(d['m_dot'][r])
                arr['k_star'][i, j, k] = float(
                    d['k_star'][r]
                )
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
                arr['tau_max'][i, j, k] = float(
                    d['tau_max'][r]
                )
                arr['n_steps'][i, j, k] = int(
                    d['n_steps'][r]
                )
                conv = bool(d['converged'][r])
                arr['converged'][i, j, k] = conv
                lab = str(d['label'][r])
                if lab not in _phase_labels:
                    raise RuntimeError(
                        f'{part_path!r}: unknown phase '
                        f'label {lab!r} not in '
                        f'PHASE_LABELS.'
                    )
                arr['label_idx_per_ic'][i, j, k] = (
                    _phase_labels.index(lab)
                )
                if conv and float(d['E'][r]) < best_E[i, j]:
                    best_E[i, j] = float(d['E'][r])
                    arr['gs_idx'][i, j] = k
                    arr['gs_label_idx'][i, j] = (
                        _phase_labels.index(lab)
                    )
                    arr['gs_m_top'][i, j] = d['m_top'][r]
                    arr['gs_m_bot'][i, j] = d['m_bot'][r]
                n_records += 1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    n_expected = n_x * n_y * n_ic
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
        print(f'WARNING: {msg}', flush=True)
    if n_records != n_expected - n_missing:
        raise RuntimeError(
            f'Internal accounting error: '
            f'n_records={n_records}, expected '
            f'{n_expected - n_missing}.'
        )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Write final NPZ; schema matches sweep.sweep's output.
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        axis_x_name=np.array(axis_x_name, dtype=object),
        axis_x_values=axis_x_grid,
        axis_y_name=np.array(axis_y_name, dtype=object),
        axis_y_values=axis_y_grid,
        ic_names=np.array(
            [name for (name, _seed) in ic_list_first],
            dtype=object,
        ),
        labels=np.array(_phase_labels, dtype=object),
        **{key: material[key] for key in _material_scalar_keys},
        **{k: v for k, v in arr.items()},
    )
    print(
        f'Wrote {out_path} '
        f'({n_records} records from {len(parts)} partials, '
        f'{n_missing} missing).',
        flush=True,
    )


# -----------------------------------------------------------------------------
def _allocate_arrays(n_x, n_y, n_ic, ny, nx):
    """Allocate per-IC and ground-state output arrays."""
    return {
        'E': np.full((n_x, n_y, n_ic), np.nan),
        'Q': np.zeros((n_x, n_y, n_ic)),
        'mz_top': np.zeros((n_x, n_y, n_ic)),
        'mz_bot': np.zeros((n_x, n_y, n_ic)),
        'm_dot': np.zeros((n_x, n_y, n_ic)),
        'k_star': np.zeros((n_x, n_y, n_ic)),
        'P_2': np.zeros((n_x, n_y, n_ic)),
        'P_6': np.zeros((n_x, n_y, n_ic)),
        'P_iso': np.zeros((n_x, n_y, n_ic)),
        'peak_over_bg': np.zeros((n_x, n_y, n_ic)),
        'n_periods': np.zeros((n_x, n_y, n_ic)),
        'q_per_period': np.zeros((n_x, n_y, n_ic)),
        'tau_max': np.full((n_x, n_y, n_ic), np.nan),
        'n_steps': np.zeros(
            (n_x, n_y, n_ic), dtype=np.int32,
        ),
        'converged': np.zeros(
            (n_x, n_y, n_ic), dtype=bool,
        ),
        'label_idx_per_ic': np.full(
            (n_x, n_y, n_ic), -1, dtype=np.int8,
        ),
        'gs_idx': np.full((n_x, n_y), -1, dtype=np.int8),
        'gs_label_idx': np.full(
            (n_x, n_y), -1, dtype=np.int8,
        ),
        'gs_m_top': np.zeros(
            (n_x, n_y, ny, nx, 3), dtype=np.float32,
        ),
        'gs_m_bot': np.zeros(
            (n_x, n_y, ny, nx, 3), dtype=np.float32,
        ),
    }


# -----------------------------------------------------------------------------
def main():
    """Read the User Configuration block and call `aggregate`.

    The axes, lattice size, and material scalars are
    recovered from the partial NPZs themselves (no need to
    duplicate them here -- the aggregator will refuse to
    combine partials that disagree).
    """
    # ================ User Configuration ================
    # Directory containing the per-element partial NPZs.
    # Must hold at least one `part_*.npz`. Typical layout:
    # `output/phase_diagram/<axis_x>_<axis_y>_partials/`.
    partial_dir = (
        'output/phase_diagram/D_H_z_partials'
    )
    # Output NPZ. Typical layout:
    # `output/phase_diagram/<axis_x>_<axis_y>.npz`.
    out_path = (
        'output/phase_diagram/D_H_z.npz'
    )
    # If True, raise on any missing (i, j, k) record. If
    # False, leave the missing entries at their allocation
    # defaults and emit one summary line.
    require_all_records = True
    # ============ End User Configuration =================
    aggregate(
        partial_dir=partial_dir,
        out_path=out_path,
        require_all_records=require_all_records,
    )


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
