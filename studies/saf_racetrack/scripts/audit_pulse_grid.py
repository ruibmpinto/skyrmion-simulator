"""Audit a pulsed-sweep output directory for missing trajectories.

Rebuilds the driver's task grid exactly as `sweep_pulse_shape.cpp`
enumerates it, compares that against the files on disk, and prints both
the missing (shape, T, peak J, ens) tuples and the SLURM array indices
that would produce them. The index list can be handed straight back to
sbatch as a sparse array, so only the gaps are re-run.

This exists because the grid enumeration order is part of the
task-to-trajectory contract: rebuilding the driver with a different
loop order while an array job is in flight makes early and late tasks
disagree about which cell they own, leaving gaps. Re-deriving the
mapping from the current source order and diffing against disk is the
reliable way to repair that.

The enumeration mirrored here is, per shape, then per substrate
temperature, then per peak current, then per ensemble member, with a
single member at T = 0 because a deterministic run has no ensemble.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m studies.saf_racetrack.scripts.audit_pulse_grid
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _build_grid(shapes, t_sub_list, peak_j_list, j_caps, n_ens):
    """Rebuild the driver's flattened task grid.

    Parameters
    ----------
    shapes : list[str]
        Shape names in driver order.
    t_sub_list : list[float]
        Substrate temperatures in driver order.
    peak_j_list : list[float]
        Peak-current ladder in driver order (A/m^2).
    j_caps : {dict, None}
        Per-shape caps in units of 1e11 A/m^2, one entry per
        temperature, or None for the uncapped (pilot) grid.
    n_ens : int
        Members per finite-temperature cell.

    Returns
    -------
    grid : list[tuple]
        One (shape, T_sub, peak_j, ens) per array index, in index
        order.
    """
    grid = []
    for shape in shapes:
        for t_idx, t_sub in enumerate(t_sub_list):
            if j_caps is None:
                cap = float('inf')
            else:
                cap = j_caps[shape][t_idx]*1.0e11 + 0.05e11
            for peak_j in peak_j_list:
                if peak_j > cap:
                    continue
                members = 1 if t_sub == 0.0 else n_ens
                for ens in range(members):
                    grid.append((shape, t_sub, peak_j, ens))
    return grid


# -----------------------------------------------------------------------------
def _missing(grid, case_dir):
    """Array indices whose trajectory file is absent.

    Parameters
    ----------
    grid : list[tuple]
        Output of `_build_grid`.
    case_dir : str
        Directory holding the trajectories.

    Returns
    -------
    gaps : list[tuple]
        (index, shape, T_sub, peak_j, ens) for each absent file.
    """
    if not os.path.isdir(case_dir):
        raise RuntimeError(f'_missing: {case_dir!r} is not a directory.')
    gaps = []
    for idx, (shape, t_sub, peak_j, ens) in enumerate(grid):
        name = '%s_T%05.1f_j%.2e_ens%03d.npz' % (shape, t_sub, peak_j, ens)
        if not os.path.isfile(os.path.join(case_dir, name)):
            gaps.append((idx, shape, t_sub, peak_j, ens))
    return gaps


# -----------------------------------------------------------------------------
def main():
    """Audit each configured case directory and report the gaps."""
    # =========================== User Configuration =========================
    stage_root = 'output/sweeps_driving_T/pulse_shape'
    stage = 'pilot'                # pilot | pilot_hi | production
    box_tags = [
        'hk36_D0p72_350x500',
        'hk36_D0p72_700x500',
        'hk36_D0p72_1400x500',
    ]
    # Driver order; must match sweep_pulse_shape.cpp.
    shapes = [
        'square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
        'tri_symmetric', 'gaussian',
    ]
    t_sub_list = [0.0, 10.0, 50.0, 100.0]
    if stage == 'pilot':
        peak_j_list = [2.0e11, 3.0e11, 4.0e11, 5.0e11, 6.0e11, 8.0e11]
        j_caps = None
        n_ens = 5
    elif stage == 'pilot_hi':
        # Extension above the first pilot's ceiling; disjoint ladder.
        peak_j_list = [9.0e11, 1.0e12, 1.2e12, 1.5e12]
        j_caps = None
        n_ens = 5
    else:
        peak_j_list = [0.5e11, 1.0e11, 2.0e11, 3.0e11, 4.0e11,
                       5.0e11, 6.0e11, 8.0e11]
        j_caps = {
            'square': [6, 6, 5, 4],
            'halfsine': [8, 8, 8, 5],
            'tri_sharprise': [8, 8, 6, 5],
            'tri_sharpfall': [8, 8, 8, 6],
            'tri_symmetric': [8, 8, 8, 6],
            'gaussian': [8, 8, 6, 5],
        }
        n_ens = 25
    # ======================= End User Configuration =========================
    grid = _build_grid(shapes, t_sub_list, peak_j_list, j_caps, n_ens)
    print('stage %s: grid is %d tasks' % (stage, len(grid)))
    for tag in box_tags:
        case_dir = os.path.join(stage_root, stage, tag)
        if not os.path.isdir(case_dir):
            print('=== %s: not present, skipped ===' % tag)
            continue
        gaps = _missing(grid, case_dir)
        n_have = len(grid) - len(gaps)
        print('=== %s: %d/%d present, %d missing ==='
              % (tag, n_have, len(grid), len(gaps)))
        if not gaps:
            continue
        by_shape = {}
        for idx, shape, t_sub, peak_j, ens in gaps:
            by_shape.setdefault(shape, []).append(
                (t_sub, peak_j, ens, idx))
        for shape in shapes:
            entries = by_shape.get(shape)
            if not entries:
                continue
            print('  %s:' % shape)
            for t_sub, peak_j, ens, idx in entries:
                print('    T=%5.1f  J=%.2e  ens=%03d  -> task %d'
                      % (t_sub, peak_j, ens, idx))
        indices = ','.join(str(g[0]) for g in gaps)
        print('  sparse re-run array for %s:' % tag)
        print('    --array=%s' % indices)


# =============================================================================
if __name__ == '__main__':
    main()
