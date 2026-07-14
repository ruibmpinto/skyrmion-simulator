"""Pair-potential analysis: reconstruct V(r) up to a mobility
factor (Q3 interactions branch).

Consumes
    output/stochastic_llgs/pair_potential/aggregate.npz
(written by `scripts.aggregate_sllg pair_potential`). Pools
ensemble-averaged dr/dt across initial separations, bins by
current separation r, and integrates `int dr/dt dr` to give
`V(r) / mu`, where `mu` is the (unknown) pair mobility.
Shape, sign, and characteristic length scale are preserved;
absolute magnitude requires an independent mobility
calibration.

Run with:
    python -m scripts.analyze_sllg_pair_potential
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import os
# Third-party
import numpy as np
import matplotlib.pyplot as plt

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rui_pinto@brown.edu)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================

# Project-wide matplotlib defaults (S4X style).
plt.rcParams['figure.dpi'] = 360
plt.rcParams['axes.labelsize'] = 18
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
plt.rcParams['legend.fontsize'] = 14
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['lines.linewidth'] = 1.5


def main():
    """Load the pair-separation aggregate, bin dr/dt by current
    separation, integrate to V(r)/mu, and write the pair-potential
    figure and the V_of_r NPZ."""
    # =========================== User Configuration =========================
    in_npz   = (
        'output/stochastic_llgs/pair_potential/aggregate.npz')
    fig_dir  = 'output/figures_sllg'
    fig_path = os.path.join(fig_dir, 'pair_potential.png')
    out_npz  = (
        'output/stochastic_llgs/pair_potential/V_of_r.npz')
    n_r_bins = 30
    # PBC contamination cutoff: drop r_init >= r_pbc_cut nm
    # because the second skyrmion's wrap-around image becomes
    # closer than the partner inside the box. With a 256 x 256
    # lattice at a=2 nm the box is 512 nm; r >= 256 nm sees
    # image-mediated attraction that masquerades as physics.
    r_pbc_cut_nm = 256.0
    # ======================= End User Configuration =========================
    os.makedirs(fig_dir, exist_ok=True)
    if not os.path.isfile(in_npz):
        raise RuntimeError(
            f'{in_npz!r} not found. Run '
            f'`python -m scripts.aggregate_sllg '
            f'pair_potential` first.'
        )
    d = np.load(in_npz, allow_pickle=True)
    r0_arr = d['r0_arr']
    t_sample = d['t_sample']
    r_mean_grid = d['r_mean_grid']
    n_alive = d['n_alive']
    # Filter mask for the PBC cut.
    pbc_mask = r0_arr * 1e9 < r_pbc_cut_nm
    print(f'Using {int(pbc_mask.sum())}/{r0_arr.size} '
          f'r_init cells (PBC cut at {r_pbc_cut_nm:.0f} nm).')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Pool dr/dt vs r across surviving, PBC-safe initial
    # separations.
    r_all = []
    dr_dt_all = []
    for i in range(r0_arr.size):
        if not pbc_mask[i]:
            continue
        if int(n_alive[i]) == 0:
            continue
        r_t = r_mean_grid[i, :]
        if np.all(np.isnan(r_t)) or t_sample.size < 4:
            continue
        dt = t_sample[1:] - t_sample[:-1]
        dr = r_t[1:] - r_t[:-1]
        v_radial = dr / dt
        r_mid = 0.5 * (r_t[1:] + r_t[:-1])
        good = np.isfinite(v_radial) & np.isfinite(r_mid)
        r_all.append(r_mid[good])
        dr_dt_all.append(v_radial[good])
    if r_all:
        r_all = np.concatenate(r_all)
        dr_dt_all = np.concatenate(dr_dt_all)
        r_lo, r_hi = float(r_all.min()), float(r_all.max())
        edges = np.linspace(r_lo, r_hi, n_r_bins + 1)
        r_centers = 0.5 * (edges[:-1] + edges[1:])
        dr_dt_binned = np.full(n_r_bins, np.nan)
        for i in range(n_r_bins):
            mask = (
                (r_all >= edges[i])
                & (r_all < edges[i + 1])
            )
            if int(mask.sum()) >= 2:
                dr_dt_binned[i] = float(
                    np.nanmean(dr_dt_all[mask])
                )
        # Trapezoidal integration with the correct
        # Smoluchowski sign convention:
        #   over-damped:  mu * d r / d t = -d V / d r
        #   =>  V(r) / mu = - integral_{r_min}^r <dr/dt> dr'.
        # The earlier version omitted the leading minus sign,
        # so V_over_mu came out flipped (rising for repulsion).
        V_over_mu = np.full(n_r_bins, np.nan)
        cum = 0.0
        r_prev = None
        f_prev = None
        for i in range(n_r_bins):
            if not np.isfinite(dr_dt_binned[i]):
                continue
            if r_prev is None:
                V_over_mu[i] = 0.0
            else:
                # Note the minus: cum -= 0.5*(f+f_prev)*dr.
                cum -= 0.5 * (dr_dt_binned[i] + f_prev) \
                    * (r_centers[i] - r_prev)
                V_over_mu[i] = cum
            r_prev = r_centers[i]
            f_prev = dr_dt_binned[i]
    else:
        r_centers = np.array([], dtype=float)
        dr_dt_binned = np.array([], dtype=float)
        V_over_mu = np.array([], dtype=float)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Persist the reconstructed V(r)/mu and its inputs.
    np.savez_compressed(
        out_npz,
        r_centers=r_centers,
        dr_dt_binned=dr_dt_binned,
        V_over_mu=V_over_mu,
    )
    print(f'Saved {out_npz}')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Two panels: r(t) traces per r_init (left) and the
    # reconstructed mobility-scaled V(r) (right).
    fig, axes = plt.subplots(1, 2)
    for i, r_init in enumerate(r0_arr):
        if int(n_alive[i]) == 0:
            continue
        # PBC-contaminated cells plotted faded; the V(r)
        # reconstruction excludes them.
        is_pbc = not pbc_mask[i]
        axes[0].plot(
            t_sample * 1e9, r_mean_grid[i, :] * 1e9,
            label=(
                rf'$r_\mathrm{{init}}={r_init*1e9:.0f}$ nm'
                + (' (PBC, excluded)' if is_pbc else '')
            ),
            alpha=(0.35 if is_pbc else 1.0),
            linestyle=('--' if is_pbc else '-'),
        )
    axes[0].set_xlabel(r'$t$ (ns)')
    axes[0].set_ylabel(r'$\langle r(t) \rangle$ (nm)')
    axes[0].legend(loc='best', frameon=False)
    axes[0].set_box_aspect(1)
    if r_centers.size > 0:
        axes[1].plot(
            r_centers * 1e9, V_over_mu, 'o-')
    axes[1].set_xlabel(r'$r$ (nm)')
    axes[1].set_ylabel(r'$V(r) / \mu$ (m$^2$/s)')
    axes[1].set_title('Pair potential (mobility-scaled)')
    axes[1].set_box_aspect(1)
    fig.tight_layout()
    fig.savefig(fig_path)
    print(f'Saved {fig_path}')


# =============================================================================
if __name__ == '__main__':
    main()
