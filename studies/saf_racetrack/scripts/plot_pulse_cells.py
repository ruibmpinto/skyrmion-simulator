"""Final top-layer configurations of selected pulsed-drive cells.

Renders every realization of every cell up to one current beyond the
cap, as its final m_z map labelled with the class the field classifier
assigned it, so each classification decision can be checked by eye.
One figure per (box, shape, temperature): rows are ascending peak
current, ending on the first rung that breaks the window.

Rows are cells, columns are ensemble members. A deterministic cell
(T = 0) has one member and leaves the rest of its row blank.

No argparse; configure the run via the variables at the top of
`main()`.

Run with:
    python -m studies.saf_racetrack.scripts.plot_pulse_cells
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
# Local
from skyrmion_simulator.stochastic_llgs.stability import classify_field

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def _find_members(stage_root, stages, box_tag, shape, t_sub, peak_j):
    """Locate every realization of one cell across the merged stages.

    Parameters
    ----------
    stage_root : str
        Root holding the per-stage directories.
    stages : list[str]
        Stage names to search, in order.
    box_tag : str
        Box directory name.
    shape : str
        Pulse-shape name.
    t_sub : float
        Substrate temperature (K).
    peak_j : float
        Peak current density (A/m^2).

    Returns
    -------
    paths : list[str]
        Trajectory paths, sorted by ensemble index.
    """
    paths = []
    for stage in stages:
        pattern = os.path.join(
            stage_root, stage, box_tag,
            '%s_T%05.1f_j%.2e_ens*.npz' % (shape, t_sub, peak_j))
        paths.extend(glob.glob(pattern))
    return sorted(paths)


# -----------------------------------------------------------------------------
def _load_cell(path):
    """Final m_z map and class of one realization.

    Parameters
    ----------
    path : str
        Trajectory NPZ path.

    Returns
    -------
    mz : numpy.ndarray(2d)
        Final top-layer m_z.
    code : str
        Class code from the field classifier.
    metrics : dict
        Classifier metrics, for annotation.
    """
    with np.load(path, allow_pickle=True) as d:
        mz = np.asarray(d['mz_final_top'], dtype=float)
        q_abs = abs(float(d['Q'][-1]))
        d1 = float(d['D1_top'][-1])
        d2 = float(d['D2_top'][-1])
        l_x = float(d['L_x'])
    code, metrics = classify_field(mz, q_abs, d1, d2, l_x)
    metrics['q_abs'] = q_abs
    metrics['ratio'] = d1/d2 if d2 > 0.0 else float('nan')
    return mz, code, metrics


# -----------------------------------------------------------------------------
def _plot_box(stage_root, stages, box_tag, cells, a, n_col, out_path):
    """Render every requested cell of one box.

    Parameters
    ----------
    stage_root : str
        Root holding the per-stage directories.
    stages : list[str]
        Stage names to search.
    box_tag : str
        Box directory name.
    cells : list[tuple]
        (shape, T_sub, peak_j) entries to render.
    a : float
        Lattice constant (m).
    n_col : int
        Ensemble members per row.
    out_path : str
        Output PNG path.
    """
    a_nm = a*1e9
    n_row = len(cells)
    fig, axes = plt.subplots(
        n_row, n_col, figsize=(3.1*n_col, 1.95*n_row), squeeze=False)
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    for r, (shape, t_sub, peak_j) in enumerate(cells):
        paths = _find_members(
            stage_root, stages, box_tag, shape, t_sub, peak_j)
        for c in range(n_col):
            ax = axes[r][c]
            ax.set_xticks([])
            ax.set_yticks([])
            if c >= len(paths):
                ax.axis('off')
                continue
            if not paths:
                # Cell absent (e.g. a repair still in flight): say so
                # in place rather than aborting the whole sweep.
                ax.text(0.5, 0.5, 'not run', transform=ax.transAxes,
                        ha='center', va='center', fontsize=9,
                        color='0.5')
                continue
            mz, code, met = _load_cell(paths[c])
            ny, nx = mz.shape
            ax.imshow(mz, origin='lower', cmap='RdBu_r', norm=norm,
                      extent=[0.0, nx*a_nm, 0.0, ny*a_nm],
                      aspect='equal')
            ax.set_title(
                '%s  ens%d\n|Q|=%.2f r=%.2f sol=%.2f Dx/Lx=%.2f'
                % (code, c, met['q_abs'], met['ratio'],
                   met['solidity'], met['dx_lx']),
                fontsize=7.5)
            if c == 0:
                ax.set_ylabel(
                    '%s\nT=%.0f K  J=%.3g' % (shape, t_sub,
                                              peak_j*1e-11),
                    fontsize=8.5)
    fig.suptitle('final top-layer $m_z$ — %s, %s, $T=%.0f$ K'
                 % (box_tag, cells[0][0], cells[0][1]), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    fig.savefig(out_path, dpi=125)
    print('Saved: %s' % out_path)
    plt.close(fig)



# -----------------------------------------------------------------------------
def _plot_t0_ladder(stage_root, stages, box_tag, shape, ladder, a,
                    n_col, out_path):
    """Render the deterministic (T = 0) current ladder of one shape.

    A T = 0 cell has one realization, so the member axis is degenerate;
    currents are laid across the grid instead and the whole ladder fits
    in one figure.

    Parameters
    ----------
    stage_root : str
        Root holding the per-stage directories.
    stages : list[str]
        Stage names to search.
    box_tag : str
        Box directory name.
    shape : str
        Pulse-shape name.
    ladder : list[float]
        Peak currents to show (A/m^2).
    a : float
        Lattice constant (m).
    n_col : int
        Panels per row.
    out_path : str
        Output PNG path.
    """
    a_nm = a*1e9
    n_row = int(np.ceil(len(ladder)/n_col))
    fig, axes = plt.subplots(n_row, n_col, squeeze=False,
                             figsize=(3.1*n_col, 2.1*n_row))
    norm = matplotlib.colors.Normalize(vmin=-1.0, vmax=1.0)
    for idx in range(n_row*n_col):
        ax = axes[idx//n_col][idx % n_col]
        ax.set_xticks([])
        ax.set_yticks([])
        if idx >= len(ladder):
            ax.axis('off')
            continue
        peak_j = ladder[idx]
        paths = _find_members(stage_root, stages, box_tag, shape, 0.0,
                              peak_j)
        if not paths:
            ax.text(0.5, 0.5, 'not run', transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='0.5')
            ax.set_title('J=%.3g' % (peak_j*1e-11), fontsize=9)
            continue
        mz, code, met = _load_cell(paths[0])
        ny, nx = mz.shape
        ax.imshow(mz, origin='lower', cmap='RdBu_r', norm=norm,
                  extent=[0.0, nx*a_nm, 0.0, ny*a_nm], aspect='equal')
        ax.set_title(
            '%s  J=%.3g\n|Q|=%.2f r=%.2f sol=%.2f Dx/Lx=%.2f'
            % (code, peak_j*1e-11, met['q_abs'], met['ratio'],
               met['solidity'], met['dx_lx']), fontsize=8)
    fig.suptitle('final top-layer $m_z$ — %s, %s, $T=0$ (deterministic)'
                 % (box_tag, shape), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=125)
    print('Saved: %s' % out_path)
    plt.close(fig)

# -----------------------------------------------------------------------------
def main():
    """Render the configured cells for each box."""
    # =========================== User Configuration =========================
    stage_root = 'output/sweeps_driving_T/pulse_shape'
    stages = ['pilot', 'pilot_hi', 'pilot_hi_all']
    out_dir = 'output/figures_driving_T/cell_configs'
    a = 2.0e-9                 # m, lattice constant
    n_col = 5                  # ensemble members per row
    # Boundary pairs per (box, shape, T): the capped current and the
    # first failing rung above it, one row each. Taken from the
    # reclassified finite-T cap tables.
    ladder = [2.0e11, 3.0e11, 4.0e11, 5.0e11, 6.0e11, 8.0e11,
              9.0e11, 1.0e12, 1.2e12, 1.5e12]
    caps = {
        'hk36_D0p72_350x500': {
            'square':        {10: 6.0e11, 50: 5.0e11, 100: 0.0},
            'halfsine':      {10: 8.0e11, 50: 6.0e11, 100: 0.0},
            'tri_sharprise': {10: 9.0e11, 50: 6.0e11, 100: 0.0},
            'tri_sharpfall': {10: 1.0e12, 50: 9.0e11, 100: 0.0},
            'tri_symmetric': {10: 9.0e11, 50: 6.0e11, 100: 0.0},
            'gaussian':      {10: 9.0e11, 50: 6.0e11, 100: 0.0},
        },
        'hk36_D0p72_700x500': {
            'square':        {10: 6.0e11, 50: 5.0e11, 100: 3.0e11},
            'halfsine':      {10: 8.0e11, 50: 6.0e11, 100: 4.0e11},
            'tri_sharprise': {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
            'tri_sharpfall': {10: 1.0e12, 50: 9.0e11, 100: 6.0e11},
            'tri_symmetric': {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
            'gaussian':      {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
        },
        'hk36_D0p72_1400x500': {
            'square':        {10: 6.0e11, 50: 5.0e11, 100: 3.0e11},
            'halfsine':      {10: 8.0e11, 50: 6.0e11, 100: 4.0e11},
            'tri_sharprise': {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
            'tri_sharpfall': {10: 1.0e12, 50: 9.0e11, 100: 6.0e11},
            'tri_symmetric': {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
            'gaussian':      {10: 9.0e11, 50: 6.0e11, 100: 5.0e11},
        },
    }
    # Every current from the bottom of the ladder up to one rung past
    # the cap, so the whole approach to the boundary is visible and not
    # just its last two steps.
    requests = []
    for box_tag, per_shape in caps.items():
        for shape in ('square', 'halfsine', 'tri_sharprise',
                      'tri_sharpfall', 'tri_symmetric', 'gaussian'):
            for t_sub, cap_j in sorted(per_shape[shape].items()):
                if cap_j <= 0.0:
                    continue          # no usable window in this box
                idx = min(range(len(ladder)),
                          key=lambda i: abs(ladder[i] - cap_j))
                last = min(idx + 1, len(ladder) - 1)
                rows = [(shape, float(t_sub), ladder[i])
                        for i in range(last + 1)]
                requests.append((box_tag, shape, t_sub, rows))
    # ======================= End User Configuration =========================
    os.makedirs(out_dir, exist_ok=True)
    # One figure per (box, shape): six rows, being the cap row and the
    # first failing row above it at each of the three temperatures.
    # Splitting by shape keeps each figure readable -- all shapes in
    # one image would be 36 rows.
    # One figure per (box, shape, T): a single temperature's ladder is
    # 6-9 rows, which stays readable; pooling temperatures would not.
    for box_tag, shape, t_sub, rows in requests:
        out_path = os.path.join(
            out_dir,
            'cells_%s_%s_T%03d.png' % (box_tag, shape, int(t_sub)))
        _plot_box(stage_root, stages, box_tag, rows, a, n_col,
                  out_path)
    # T = 0 for every box and shape: the full ladder, one member each.
    for box_tag in caps:
        for shape in ('square', 'halfsine', 'tri_sharprise',
                      'tri_sharpfall', 'tri_symmetric', 'gaussian'):
            out_path = os.path.join(
                out_dir, 'cells_%s_%s_T000.png' % (box_tag, shape))
            _plot_t0_ladder(stage_root, stages, box_tag, shape, ladder,
                            a, n_col, out_path)


# =============================================================================
if __name__ == '__main__':
    main()
