"""Banerjee 2014 critical-field cross-check.

Quantitative gate for the FM <-> SkX critical field. Loads the
Güngördü-sweep phase map (`gungordu_K_H.npz`, produced by
`run_gungordu_sweep.py` on the cluster), and at a few fixed
anisotropy slices a_s extracts the simulator's FM<->SkX
transition field h_c (the h_g at which the ground-state phase
flips between SkX and FM as the field increases), then compares
it to Banerjee 2014 Fig. 1(b).

Reference: S. Banerjee et al., Phys. Rev. X 4, 031045 (2014),
Fig. 1(b), via `banerjee_reference.ban_fm_skx_hc`. Banerjee's
circular-cell variational ansatz underestimates SkX stability,
so the simulator (full relaxation) is expected to sustain SkX
to AT LEAST Banerjee's h_c -- i.e. the simulated h_c should sit
on or above the Banerjee boundary, agreeing within the figure-
read tolerance.

This is post-processing on the #5 Güngördü sweep output -- no
new simulation. The phase map must already exist; if absent the
test reports INCONCLUSIVE.

Functions
---------
extract_fm_skx_hc
    Find the simulator FM<->SkX critical field at one a_s slice.
main
    Load the sweep NPZ, compare h_c at the chosen slices to
    Banerjee, report pass/fail.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
# Local
from src.phase_diagram.validation.banerjee_reference import (
    ban_fm_skx_hc)

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================
# Phase-label families (substring match against the classifier
# labels stored in the NPZ).
_FM_TAGS = ('FM',)
_SKX_TAGS = ('SkX',)


def _is_fm(label):
    """True if a phase label is a ferromagnetic family member."""
    return any(tag in label for tag in _FM_TAGS)


def _is_skx(label):
    """True if a phase label is the skyrmion-crystal phase."""
    return any(tag == label for tag in _SKX_TAGS)


def extract_fm_skx_hc(a_s_axis, h_g_axis, label_grid, labels,
                      a_s_target):
    """Simulator FM<->SkX critical field at one anisotropy slice.

    Picks the a_s column nearest `a_s_target`, walks the field
    axis from low to high, and returns the h_g midway between
    the last SkX cell and the first FM cell above it (the
    field-driven SkX -> FM destruction). Returns NaN if the
    column has no clean SkX-below / FM-above ordering.

    Parameters
    ----------
    a_s_axis : numpy.ndarray(1d)
        Anisotropy coordinate per x-index (length n_x).
    h_g_axis : numpy.ndarray(1d)
        Field coordinate per y-index (length n_y).
    label_grid : numpy.ndarray(2d)
        Ground-state label index, shape (n_x, n_y).
    labels : list[str]
        Label strings indexed by `label_grid`.
    a_s_target : float
        Anisotropy slice to probe.

    Returns
    -------
    a_s_used : float
        The a_s of the nearest column actually used.
    h_c : float
        FM<->SkX critical field h_g, or NaN if undefined.
    """
    i = int(np.argmin(np.abs(a_s_axis - a_s_target)))
    col = label_grid[i, :]
    col_labels = [labels[k] for k in col]
    # Find the highest-field SkX cell and the first FM cell
    # above it. Field increases with the y-index.
    skx_rows = [j for j, lbl in enumerate(col_labels)
                if _is_skx(lbl)]
    if not skx_rows:
        return float(a_s_axis[i]), float('nan')
    j_top_skx = max(skx_rows)
    fm_above = [j for j in range(j_top_skx + 1, len(col_labels))
                if _is_fm(col_labels[j])]
    if not fm_above:
        return float(a_s_axis[i]), float('nan')
    j_fm = min(fm_above)
    h_c = 0.5 * (h_g_axis[j_top_skx] + h_g_axis[j_fm])
    return float(a_s_axis[i]), float(h_c)


def main():
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    in_path = 'output/phase_diagram/validation/gungordu_K_H.npz'
    # Easy-plane SkX slices where the FM<->SkX boundary is
    # well defined and SkX is robust.
    a_s_slices = [0.0, 0.5, 1.0]
    rtol = 0.20
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print('banerjee_critical_field (FM<->SkX h_c vs Banerjee '
          '2014 Fig.1b):')
    if not os.path.exists(in_path):
        print(f'  NO sweep output at {in_path!r}: run the '
              f'Güngördü sweep (cluster) and aggregate first.')
        print('  status: INCONCLUSIVE (no sweep data)')
        return False
    z = np.load(in_path, allow_pickle=True)
    K_values = np.asarray(z['axis_x_values'], dtype=float)
    H_values = np.asarray(z['axis_y_values'], dtype=float)
    labels = list(z['labels'])
    # Ground state = MIN-ENERGY candidate per cell (re-derived
    # from per-IC energies, matching Güngördü's energy
    # comparison). The stored gs_label_idx additionally
    # requires strict torque convergence, which the textured
    # states rarely reach -> all-FM. See plot_gungordu_overlay.
    E = np.asarray(z['E'])
    lpi = np.asarray(z['label_idx_per_ic'], dtype=int)
    Em = np.where(np.isfinite(E), E, np.inf)
    kmin = np.argmin(Em, axis=2)
    label_grid = np.take_along_axis(
        lpi, kmin[:, :, None], axis=2)[:, :, 0]
    A_ex = float(z['A_ex'])
    Ms = float(z['Ms'])
    mu0 = float(z['mu0'])
    D = float(z['D_run'])
    # Physical axes -> Güngördü dimensionless coords (demag off,
    # bare K, K = -A_s_phys): a_s = -K A_ex / D^2.
    a_s_axis = -K_values * A_ex / (D * D)
    h_g_axis = Ms * H_values * A_ex / (D * D)
    # a_s_axis may be descending (K descending -> a_s ascending
    # or vice versa); sort ascending for the slice search.
    order = np.argsort(a_s_axis)
    a_s_axis = a_s_axis[order]
    label_grid = label_grid[order, :]
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    all_ok = True
    n_checked = 0
    for a_s_t in a_s_slices:
        a_s_used, h_c_sim = extract_fm_skx_hc(
            a_s_axis, h_g_axis, label_grid, labels, a_s_t)
        h_c_ban = ban_fm_skx_hc(a_s_used)
        if np.isnan(h_c_sim) or np.isnan(h_c_ban):
            print(f'  a_s={a_s_used:+.2f}: h_c_sim='
                  f'{h_c_sim:.3f}, h_c_Banerjee={h_c_ban:.3f} '
                  f'-> undefined (no clean SkX->FM column or '
                  f'out of Banerjee range); skipped')
            continue
        n_checked += 1
        rel = abs(h_c_sim - h_c_ban) / max(h_c_ban, 1e-9)
        ok = rel < rtol
        all_ok = all_ok and ok
        print(f'  a_s={a_s_used:+.2f}: h_c_sim={h_c_sim:.3f}, '
              f'h_c_Banerjee={h_c_ban:.3f}, rel.err='
              f'{rel*100:.1f}% {"PASS" if ok else "FAIL"}')
    if n_checked == 0:
        print('  status: INCONCLUSIVE (no slice yielded a '
              'clean FM<->SkX critical field)')
        return False
    print(f'  status: {"PASS" if all_ok else "FAIL"} '
          f'(rtol={rtol*100:.0f}%, {n_checked} slices)')
    return all_ok


# =============================================================================
if __name__ == '__main__':
    ok = main()
    raise SystemExit(0 if ok else 1)
