"""Classify the pulsed-drive pilot and derive the usable-current table.

Reads every trajectory of the Phase-2 pilot, classifies its final
top-layer configuration with the same field classifier the racetrack
campaign uses, and reports two things per box:

- the class composition of every (shape, T_sub, peak J) cell, so mixed
  cells are visible rather than hidden behind a majority label. Classes
  are S compact, E elongated, P spanning (the domain bridges the track
  or reaches its own periodic image, so it is not a localized object),
  L labyrinth, A annihilated and R reversed background (the track
  switched, so the reversed-domain mask selects the background and the
  shape metrics describe it instead);
- the largest peak J at which no realization of a cell became a
  labyrinth or annihilated, which is the usable-current table that sets
  the production grids. That table covers finite temperatures only: a
  T = 0 point is a single deterministic realization and cannot support
  the same all-members-survived criterion.

The pilot deliberately sweeps peak J past the expected boundary, so
cells failing at the top of the range are the intended outcome, not a
problem.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m scripts.analyze_pulse_pilot
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
import re
# Third-party
import numpy as np
# Local
from src.stochastic_llgs.stability import classify_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _classify_dir(case_dir, cells):
    """Classify every trajectory in one box directory into `cells`.

    Accumulates into the caller's dict so the low-current and
    high-current stages, which use disjoint current ladders, compose
    into one table per box.

    Parameters
    ----------
    case_dir : str
        Directory holding `<shape>_T<T>_j<J>_ens<n>.npz` trajectories.
    cells : dict
        Accumulator, mapping (shape, T_sub, peak_j) to the list of
        class codes of its realizations. Mutated in place.

    Returns
    -------
    cells : dict
        The same accumulator, for convenience.
    """
    pattern = re.compile(
        r'(.+)_T(\d+\.\d)_j([0-9.e+]+)_ens(\d+)\.npz')
    paths = sorted(glob.glob(
        os.path.join(case_dir, '*_T*_j*_ens*.npz')))
    if not paths:
        raise RuntimeError(
            f'_classify_dir: no trajectories in {case_dir!r}.')
    for path in paths:
        base = os.path.basename(path)
        match = pattern.match(base)
        if match is None:
            raise RuntimeError(
                f'_classify_dir: cannot parse {base!r}; the naming '
                f'convention changed.')
        shape = match.group(1)
        t_sub = float(match.group(2))
        peak_j = float(match.group(3))
        with np.load(path, allow_pickle=True) as d:
            mz = np.asarray(d['mz_final_top'], dtype=float)
            q_abs = abs(float(d['Q'][-1]))
            d1 = float(d['D1_top'][-1])
            d2 = float(d['D2_top'][-1])
            l_x = float(d['L_x'])
        code, _metrics = classify_field(mz, q_abs, d1, d2, l_x)
        cells.setdefault((shape, t_sub, peak_j), []).append(code)
    return cells


# -----------------------------------------------------------------------------
def _report_composition(cells, shapes, temps, js):
    """Print the per-cell class composition.

    Parameters
    ----------
    cells : dict
        Output of `_classify_dir`.
    shapes : list[str]
        Shape names in reporting order.
    temps, js : list[float]
        Grid axes actually present.
    """
    header = 'shape           T   '
    for peak_j in js:
        header = header + '{:>12s}'.format('%.3g' % (peak_j*1e-11))
    print(header)
    for shape in shapes:
        for t_sub in temps:
            line = '{:15s}{:4.0f}'.format(shape, t_sub)
            for peak_j in js:
                codes = cells.get((shape, t_sub, peak_j))
                if codes is None:
                    line = line + '{:>12s}'.format('-')
                    continue
                # Full composition, not just the dominant class: a
                # cell reported as "S3/5" hides whether the other two
                # members elongated or broke up, which is exactly the
                # distinction the usable-J test turns on.
                classes = ('S', 'E', 'P', 'L', 'A', 'R')
                counts = {k: codes.count(k) for k in classes}
                n_known = sum(counts.values())
                if n_known != len(codes):
                    raise RuntimeError(
                        'unknown class code in %s T=%.1f J=%.2e'
                        % (shape, t_sub, peak_j))
                tag = ''.join('%s%d' % (k, counts[k])
                              for k in classes if counts[k])
                line = line + '{:>12s}'.format(tag)
            print(line)
        print()


# -----------------------------------------------------------------------------
def _report_usable(cells, shapes, temps, js):
    """Print the largest fully surviving peak current per shape and T.

    T = 0 is excluded: a deterministic point has a single realization,
    so "every realization survived" is a far weaker statement there than
    over an ensemble, and it would sit in the same table as finite-T
    entries backed by five members each. Caps are therefore defined at
    finite temperature only.

    A cell counts as usable only if EVERY realization stayed a skyrmion
    (compact or elongated); one labyrinth member disqualifies it, since
    the production grids must not straddle the boundary.

    The reported value is the largest peak J such that this cell AND
    every lower current also survived -- the contiguous window, not the
    maximum over usable cells. Survival is not always monotonic in J
    here: a cell can fail while a higher current survives, because at
    very high drive the texture can be swept into a different final
    state rather than progressively degrading. A '*' marks any shape
    whose usable set is non-contiguous, since a bare threshold would
    misrepresent it.

    Parameters
    ----------
    cells : dict
        Output of `_classify_dir`.
    shapes : list[str]
        Shape names in reporting order.
    temps, js : list[float]
        Grid axes actually present.
    """
    finite_temps = [t for t in temps if t > 0.0]
    if not finite_temps:
        raise RuntimeError(
            '_report_usable: no finite-temperature cells; caps are '
            'undefined at T = 0 alone.')
    header = '{:15s}'.format('shape')
    for t_sub in finite_temps:
        header = header + '{:>12s}'.format('T=%.0f' % t_sub)
    print(header)
    for shape in shapes:
        line = '{:15s}'.format(shape)
        for t_sub in finite_temps:
            survived = []
            for peak_j in js:
                codes = cells.get((shape, t_sub, peak_j))
                survived.append(
                    codes is not None
                    and all(c in ('S', 'E') for c in codes))
            # Contiguous run from the lowest current upward.
            n_run = 0
            while n_run < len(js) and survived[n_run]:
                n_run = n_run + 1
            gap = any(survived[n_run:])
            if n_run == 0:
                text = 'none'
            else:
                text = '%.3g' % (js[n_run - 1]*1e-11)
                if gap:
                    text = text + '*'
            line = line + '{:>12s}'.format(text)
        print(line)
    print('  * usable set is non-contiguous: a higher current survived '
          'above a failing one.')


# -----------------------------------------------------------------------------
def main():
    """Classify each pilot box and report its usable-current table."""
    # =========================== User Configuration =========================
    stage_root = 'output/sweeps_driving_T/pulse_shape'
    # Stages are merged per box: their current ladders are disjoint, so
    # together they give one contiguous window per (shape, T).
    stages = ['pilot', 'pilot_hi', 'pilot_hi_all']
    box_tags = [
        'hk36_D0p72_350x500',
        'hk36_D0p72_700x500',
        'hk36_D0p72_1400x500',
    ]
    shapes = [
        'square', 'halfsine', 'tri_sharprise', 'tri_sharpfall',
        'tri_symmetric', 'gaussian',
    ]
    # ======================= End User Configuration =========================
    for tag in box_tags:
        cells = {}
        used = []
        for stage in stages:
            case_dir = os.path.join(stage_root, stage, tag)
            if not os.path.isdir(case_dir):
                continue
            if not glob.glob(os.path.join(case_dir, '*_ens*.npz')):
                continue
            _classify_dir(case_dir, cells)
            used.append(stage)
        if not cells:
            print('=== %s: no data, skipped ===' % tag)
            continue
        print('    (stages merged: %s)' % ', '.join(used))
        temps = sorted({k[1] for k in cells})
        js = sorted({k[2] for k in cells})
        n_traj = sum(len(v) for v in cells.values())
        print('===== %s: %d trajectories, %d cells ====='
              % (tag, n_traj, len(cells)))
        # Cells short of their full ensemble mean the box is still
        # running; say so rather than presenting a partial count as
        # final.
        sizes = sorted({(k[1], len(v)) for k, v in cells.items()})
        print('members per cell by T: %s' % sizes)
        print()
        _report_composition(cells, shapes, temps, js)
        print('--- usable J: largest peak with NO labyrinth or '
              'annihilated member (10^11 A/m^2), finite T only ---')
        _report_usable(cells, shapes, temps, js)
        print()


# =============================================================================
if __name__ == '__main__':
    main()
