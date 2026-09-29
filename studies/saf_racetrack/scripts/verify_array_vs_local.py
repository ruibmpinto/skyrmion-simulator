"""Verify array + aggregate produce the same NPZ as local.

Runs the `test` grid at small nx/ny once with the
local-mode `sweep()` and once by calling
`sweep_array_partial()` in a loop (one call per
SLURM_ARRAY_TASK_ID), then aggregates the partials and
diffs every key in the resulting NPZ. Exits non-zero on any
mismatch.

Usage
-----
    PYTHONPATH=. python studies/saf_racetrack/scripts/verify_array_vs_local.py
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
import shutil
import sys
# Third-party
import numpy as np
# Local
from skyrmion_simulator.phase_diagram.aggregate import aggregate
from skyrmion_simulator.phase_diagram.sweep import (
    build_tasks,
    sweep,
    sweep_array_partial,
)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _diff_npz(local_path, array_path):
    """Diff two NPZs. Return list of mismatching keys.

    Parameters
    ----------
    local_path : str
        Path to the local-mode NPZ.
    array_path : str
        Path to the array+aggregate NPZ.

    Returns
    -------
    mismatches : list[str]
        Keys whose contents differ. Empty if the two NPZs
        are bit-for-bit equivalent within tolerance.
    """
    with np.load(local_path, allow_pickle=True) as a, \
            np.load(array_path, allow_pickle=True) as b:
        keys_a = set(a.files)
        keys_b = set(b.files)
        if keys_a != keys_b:
            print(
                f'key set differs:\n'
                f'  only in local: {sorted(keys_a - keys_b)}\n'
                f'  only in array: {sorted(keys_b - keys_a)}'
            )
            return sorted(keys_a.symmetric_difference(keys_b))
        mismatches = []
        for key in sorted(keys_a):
            va = a[key]
            vb = b[key]
            if va.dtype.kind == 'O':
                ok = list(va.ravel()) == list(vb.ravel())
            elif np.issubdtype(va.dtype, np.floating):
                # E and energies may differ in their last
                # bits due to non-deterministic FFT order
                # in the process pool. Use a loose rtol that
                # still catches genuine bugs.
                ok = np.allclose(
                    va, vb, rtol=1e-6, atol=1e-10,
                    equal_nan=True,
                )
            else:
                ok = np.array_equal(va, vb)
            if not ok:
                mismatches.append(key)
                if np.issubdtype(va.dtype, np.floating):
                    diff = np.abs(va - vb)
                    print(
                        f'  {key}: shape={va.shape} '
                        f'max|diff|={np.nanmax(diff):.3e}'
                    )
                else:
                    print(f'  {key}: dtype={va.dtype} differ')
    return mismatches


def main():
    """Run local and array modes and diff outputs."""
    # ================ User Configuration ================
    grid_name = 'test'
    nx = 32
    ny = 32
    max_steps = 500
    tol_torque = 1e-4
    tol_dE = 1e-7
    alpha_relax = 1.0
    workers = 4
    sims_per_task = 1
    out_dir = os.path.join('output', 'phase_diagram')
    local_out = os.path.join(out_dir, f'{grid_name}_local.npz')
    array_out = os.path.join(out_dir, f'{grid_name}_array.npz')
    partial_dir = os.path.join(
        out_dir, f'{grid_name}_partials_verify',
    )
    # ============ End User Configuration =================
    os.makedirs(out_dir, exist_ok=True)
    if os.path.isdir(partial_dir):
        shutil.rmtree(partial_dir)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Local mode
    print('=== local mode ===', flush=True)
    sweep(
        grid_name=grid_name, nx=nx, ny=ny,
        workers=workers, out_path=local_out,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
        verbose=False,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Array mode: one call per task, sequentially
    tasks, _, _ = build_tasks(
        grid_name, nx=nx, ny=ny,
        max_steps=max_steps,
        tol_torque=tol_torque, tol_dE=tol_dE,
        alpha_relax=alpha_relax,
    )
    n_tasks = len(tasks)
    print(
        f'=== array mode ({n_tasks} array elements) ===',
        flush=True,
    )
    for tid in range(n_tasks):
        sweep_array_partial(
            grid_name=grid_name, nx=nx, ny=ny,
            max_steps=max_steps,
            tol_torque=tol_torque, tol_dE=tol_dE,
            alpha_relax=alpha_relax,
            array_task_id=tid,
            sims_per_task=sims_per_task,
            partial_dir=partial_dir,
        )
    print('=== aggregate ===', flush=True)
    aggregate(
        grid_name=grid_name, nx=nx, ny=ny,
        partial_dir=partial_dir, out_path=array_out,
        require_all_records=True,
    )
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Diff
    print('=== diff ===', flush=True)
    mismatches = _diff_npz(local_out, array_out)
    if mismatches:
        print(
            f'\nFAIL: {len(mismatches)} key(s) differ: '
            f'{mismatches}',
            flush=True,
        )
        return 1
    print('\nPASS: every key matches.', flush=True)
    return 0


# =============================================================================
if __name__ == '__main__':
    sys.exit(main())
