"""Field-based skyrmion stability classification.

Classifies a single relaxed/driven SAF configuration into compact
skyrmion / elongated skyrmion / labyrinth / annihilated from the
raw top-layer m_z field, the topological charge and the LCC
ellipse axes. Shared by the track-width plot script (per-cell ens0
classification) and the aggregator (per-realization survival
criterion).

Functions
---------
periodic_x_components
    Connected components of a Boolean field with periodic x,
    free y.
classify_field
    Classify one configuration from the raw field.
decide_class
    Decide the stability class from the per-cell metrics.
"""
#
#                                                                       Modules
# =============================================================================
# Third-party
import numpy as np
import scipy.ndimage as ndi
from scipy.spatial import ConvexHull

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def periodic_x_components(mask, min_cells):
    """Connected components of a Boolean field with periodic x, free y.

    Labels the 4-connected components of `mask`, merges labels that touch
    across the periodic seam (column 0 <-> column nx-1) via a union-find,
    and DROPS components smaller than `min_cells` -- at finite T the
    reversed-domain mask is peppered with single-cell thermal speckles
    that would otherwise be counted as domains. Returns the retained
    component count, the largest-component area fraction, whether any
    retained component wraps the x seam (connects to its own periodic
    image -> spanning stripe), and the total retained area (cells).

    Parameters
    ----------
    mask : numpy.ndarray(2d, bool)
        Reversed-domain mask, shape (ny, nx). x is the periodic axis.
    min_cells : int
        Minimum component size (cells) to count as a real domain.

    Returns
    -------
    n_comp : int
        Number of retained components after the seam merge + size filter.
    f_max : float
        Largest retained component cell count / total cells.
    x_percolates : bool
        True if a retained component spans the periodic-x seam.
    area : int
        Total retained (domain) area in cells.
    dx_cells : int
        Periodic-aware x-extent (columns) of the largest retained
        component -- the span along the periodic axis, used for the
        loss-of-periodicity gate (this is D_x, not the ellipse major
        axis D_1 which may lie along free-y).
    solidity : float
        Area / convex-hull-area of the largest retained component
        (periodic-unwrapped in x). Near 1 for a convex blob or stripe
        (skyrmion); low for a serpentine, hull-underfilling labyrinth.
    """
    lab, n = ndi.label(mask)
    if n == 0:
        return 0, 0.0, False, 0, 0, 1.0
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Union-find over labels; merge across the periodic-x seam.
    parent = list(range(n + 1))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)
    left = lab[:, 0]
    right = lab[:, -1]
    for a, b in zip(left, right):
        if a > 0 and b > 0:
            union(int(a), int(b))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Merged sizes; drop speckles below min_cells; seam-wrap test on the
    # retained (large) components only.
    roots = np.array([find(i) for i in range(n + 1)])
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    merged = {}
    for label in range(1, n + 1):
        r = roots[label]
        merged[r] = merged.get(r, 0) + int(sizes[label])
    big = {r: s for r, s in merged.items() if s >= min_cells}
    if not big:
        return 0, 0.0, False, 0, 0, 1.0
    n_comp = len(big)
    area = int(sum(big.values()))
    f_max = max(big.values()) / float(mask.size)
    seam = ({roots[int(a)] for a in left if a > 0}
            & {roots[int(b)] for b in right if b > 0})
    x_percolates = any(r in big for r in seam)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Largest retained component; its periodic x-extent (D_x) = nx minus
    # the largest cyclic gap of empty columns, and its solidity on the
    # x-unwrapped mask (rolled so the gap sits at the seam).
    r_max = max(big, key=big.get)
    comp = (roots[lab] == r_max)
    occ = np.any(comp, axis=0)
    nx = mask.shape[1]
    if occ.all():
        dx_cells = nx
        comp_uw = comp
    else:
        idx = np.flatnonzero(occ)
        gaps = np.diff(idx) - 1
        wrap_gap = idx[0] + nx - idx[-1] - 1
        max_gap = int(max(gaps.max() if gaps.size else 0, wrap_gap))
        dx_cells = nx - max_gap
        if gaps.size and gaps.max() >= wrap_gap:
            g = int(gaps.argmax())
            comp_uw = np.roll(comp, nx - int(idx[g + 1]), axis=1)
        else:
            comp_uw = comp
    pts = np.column_stack(np.nonzero(comp_uw))
    try:
        solidity = comp_uw.sum() / float(ConvexHull(pts).volume)
    except Exception:
        solidity = 1.0
    return n_comp, f_max, x_percolates, area, dx_cells, solidity


# -----------------------------------------------------------------------------
def classify_field(mz, q_abs, D1, D2, L_x):
    """Classify one relaxed/driven configuration from the raw field.

    Combines the three physical signals: the topological charge |Q|
    (skyrmion present vs collapsed vs multi-domain), the periodic-x
    connected-component / percolation structure of the reversed domain
    (single vs fragmented vs spanning), and the D_1 / L_x periodic gate
    (a domain longer than half the track can bridge its own x-image, so
    the single-skyrmion ellipse is invalid).

    The loss-of-periodicity gate uses D_x, the field-measured x-extent
    of the reversed domain (periodic axis), NOT the ellipse major axis
    D_1 -- a skyrmion elongated along free-y has large D_1 but small
    D_x and does not bridge its periodic-x image.

    Parameters
    ----------
    mz : numpy.ndarray(2d)
        Top-layer m_z of the final drive frame, shape (ny, nx).
    q_abs : float
        |Q| (absolute topological charge) of that frame.
    D1, D2 : float
        LCC ellipse major / minor axes (m), used only for the D_1/D_2
        elongation ratio (per-cell mean or per-realization value).
    L_x : float
        Track length along the periodic axis (m).

    Returns
    -------
    code : str
        'S' compact skyrmion, 'E' elongated / spanning skyrmion,
        'L' labyrinth / multi-domain, 'A' annihilated (ferromagnetic).
    metrics : dict
        Diagnostic values used by the decision (for the report table).
    """
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Field-analysis thresholds (physical; local, not global state).
    core_thresh = -0.5  # m_z below this = reversed domain (not T noise)
    min_frac = 0.002    # component size below this frac of box = speckle
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Reversed domain, de-speckled: threshold hard (m_z < -0.5) and drop
    # components below min_frac of the box, so thermal fluctuations near
    # m_z = 0 are not counted as domains.
    core = mz < core_thresh
    min_cells = max(20, int(min_frac * mz.size))
    n_comp, f_max, x_perc, area, dx_cells, solidity = \
        periodic_x_components(core, min_cells)
    core_frac = area / float(mz.size)
    # Loss-of-periodicity uses the field x-extent D_x (= dx_cells / nx),
    # not the ellipse major axis. D1/L_x is kept only for reference.
    dx_lx = dx_cells / float(mz.shape[1])
    d1_lx = D1 / L_x if (L_x > 0.0 and np.isfinite(D1)) else 0.0
    ratio = D1 / D2 if (D2 > 0.0 and np.isfinite(D1)
                        and np.isfinite(D2)) else 1.0
    metrics = {
        'q_abs': q_abs, 'core_frac': core_frac, 'n_comp': n_comp,
        'f_max': f_max, 'x_perc': bool(x_perc), 'dx_lx': dx_lx,
        'd1_lx': d1_lx, 'ratio': ratio, 'solidity': solidity,
    }
    return decide_class(metrics), metrics


# -----------------------------------------------------------------------------
def decide_class(m):
    """Decide the stability class from the per-cell metrics.

    Topology first, then shape. A single-winding reversed domain
    ($\\Q\\approx1$) is a skyrmion (compact or elongated) only if it is
    convex-like (high solidity); a serpentine, hull-underfilling domain
    is a labyrinth however its charge integrates. Neither the reversed
    fraction nor a lone satellite domain demotes a genuine skyrmion.

    Parameters
    ----------
    m : dict
        Metrics from `classify_field`: q_abs, n_comp, core_frac,
        x_perc, dx_lx, ratio, solidity.

    Returns
    -------
    code : str
        'S', 'E', 'L', or 'A'.
    """
    q_sk = 0.5          # |Q| above -> a topological skyrmion is present
    q_multi = 1.6       # |Q| above -> more than one skyrmion
    core_min = 0.005    # domain-area frac below -> no domain (FM)
    x_gate = 0.5        # D_x/L_x above -> periodic-image break (index p)
    r_elong = 1.7       # D1/D2 above -> elongated
    n_dom = 3           # this many domains -> labyrinth (multi-domain)
    s_min = 0.7         # solidity below -> serpentine labyrinth
    q = m['q_abs']
    n_comp = m['n_comp']
    # No reversed domain (and no sub-floor winding): ferromagnetic.
    if n_comp == 0 or (m['core_frac'] < core_min and q < q_sk):
        return 'A'
    # Genuine labyrinth: more than one winding, or many domains.
    if q >= q_multi or n_comp >= n_dom:
        return 'L'
    # Chargeless texture (|Q| ~ 0): not a skyrmion -> stripe / labyrinth.
    if q < q_sk:
        return 'L'
    # One winding: a convex-like blob/stripe is a skyrmion; a serpentine,
    # hull-underfilling domain is a labyrinth.
    if m['solidity'] < s_min:
        return 'L'
    # Skyrmion: elongated/spanning (E) vs compact (S).
    if m['x_perc'] or m['dx_lx'] >= x_gate or m['ratio'] >= r_elong:
        return 'E'
    return 'S'
