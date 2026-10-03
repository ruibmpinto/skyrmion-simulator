"""Aggregate the finite-T skyrmion-size scan (benchmark #9b).

Collects the per-trajectory NPZs written by
`scan_skyrmion_size_thermal` into one small summary file, one
(R_sk_mean, R_sk_std) pair per trajectory, so only one file needs
to be pulled from the cluster. `test_skyrmion_size_thermal` reads
this aggregate when present.

Reads  : <in_dir>/*.npz
Writes : <out_path> with per-trajectory H, T, seed, R_sk_mean,
         R_sk_std and the shared dot radius R_dot.

Run with:
    python -m \
    skyrmion_simulator.phase_diagram.validation.aggregate_skyrmion_size_thermal

Functions
---------
aggregate_skyrmion_size_thermal
    Collect the per-trajectory NPZs into one summary NPZ.
main
    Aggregate the default scan output directory.
"""
#
#                                                                       Modules
# =============================================================================
# Standard
import glob
import os
# Third-party
import numpy as np

#
#                                                          Authorship & Credits
# =============================================================================
__author__ = 'Rui Barreira (rbarreira@ethz.ch)'
__credits__ = ['Rui Barreira']
__status__ = 'Development'

# =============================================================================
#
# =============================================================================


def aggregate_skyrmion_size_thermal(in_dir, out_path):
    """Collect the #9b per-trajectory NPZs into one summary NPZ.

    Parameters
    ----------
    in_dir : str
        Directory holding the per-trajectory NPZ files.
    out_path : str
        Path of the aggregate NPZ to write.

    Returns
    -------
    n_files : int
        Number of trajectories aggregated.
    """
    fs = sorted(glob.glob(os.path.join(in_dir, '*.npz')))
    if not fs:
        raise RuntimeError(
            f'aggregate_skyrmion_size_thermal: no NPZ in {in_dir!r}; '
            f'run the thermal scan array first.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-trajectory records
    H = np.empty(len(fs), dtype=float)
    T = np.empty(len(fs), dtype=float)
    seed = np.empty(len(fs), dtype=np.int64)
    R_sk_mean = np.empty(len(fs), dtype=float)
    R_sk_std = np.empty(len(fs), dtype=float)
    for i, f in enumerate(fs):
        with np.load(f) as z:
            H[i] = float(z['H'])
            T[i] = float(z['T'])
            seed[i] = int(z['seed'])
            R_sk_mean[i] = float(z['R_sk_mean'])
            R_sk_std[i] = float(z['R_sk_std'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared dot radius, identical in every file of the scan
    with np.load(fs[0]) as z0:
        R_dot = float(z0['R_dot'])
    np.savez_compressed(
        out_path, H=H, T=T, seed=seed, R_sk_mean=R_sk_mean,
        R_sk_std=R_sk_std, R_dot=R_dot, n_files=len(fs))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-(field, temperature) summary
    print(f'aggregated {len(fs)} trajectories -> {out_path}')
    for Hv in np.unique(H):
        for Tv in np.unique(T[H == Hv]):
            sel = (H == Hv) & (T == Tv)
            gm = float(np.mean(R_sk_mean[sel]))
            print(f'  H={Hv*1e3:4.0f} mT  T={Tv:6.1f} K: '
                  f'{int(sel.sum())} seeds, <R_sk>={gm*1e9:5.1f} nm')
    return len(fs)


# -----------------------------------------------------------------------------
def main():
    """Aggregate the #9b scan output directory."""
    in_dir = 'output/phase_diagram/validation/skyrmion_size_thermal'
    out_path = ('output/phase_diagram/validation/'
                'skyrmion_size_thermal_agg.npz')
    aggregate_skyrmion_size_thermal(in_dir, out_path)


# =============================================================================
if __name__ == '__main__':
    main()
