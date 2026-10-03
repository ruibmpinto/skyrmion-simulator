"""Aggregate the skyrmion Neel-Arrhenius scan (benchmark #8).

Collects the per-trajectory NPZs written by
`scan_skyrmion_arrhenius` into one small summary file, so only one
file needs to be pulled from the cluster for the Arrhenius fit.
`test_skyrmion_arrhenius` reads this aggregate when present.

Reads  : <in_dir>/T*.npz
Writes : <out_path> with per-trajectory T_sub, t_collapse,
         alive_at_end, flip_index and the shared run settings.

Run with:
    python -m \
    skyrmion_simulator.stochastic_llgs.validation.aggregate_skyrmion_arrhenius

Functions
---------
aggregate_skyrmion_arrhenius
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


def aggregate_skyrmion_arrhenius(in_dir, out_path):
    """Collect the #8 per-trajectory NPZs into one summary NPZ.

    Parameters
    ----------
    in_dir : str
        Directory holding the per-trajectory `T*.npz` files.
    out_path : str
        Path of the aggregate NPZ to write.

    Returns
    -------
    n_files : int
        Number of trajectories aggregated.
    """
    fs = sorted(glob.glob(os.path.join(in_dir, 'T*.npz')))
    if not fs:
        raise RuntimeError(
            f'aggregate_skyrmion_arrhenius: no T*.npz found in '
            f'{in_dir!r}; run the scan array first.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-trajectory records
    T_sub = np.empty(len(fs), dtype=float)
    t_collapse = np.empty(len(fs), dtype=float)
    alive = np.empty(len(fs), dtype=bool)
    flip_index = np.empty(len(fs), dtype=int)
    for i, f in enumerate(fs):
        with np.load(f) as z:
            T_sub[i] = float(z['T_sub'])
            t_collapse[i] = float(z['t_collapse'])
            alive[i] = bool(z['alive_at_end'])
            flip_index[i] = int(z['flip_index'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared run settings, identical in every file of the scan
    with np.load(fs[0]) as z0:
        n_drive = int(z0['n_drive'])
        dt = float(z0['dt'])
        sample_every = int(z0['sample_every'])
        H_ext = np.asarray(z0['H_ext'], dtype=float)
        D = float(z0['D'])
    np.savez_compressed(
        out_path, T_sub=T_sub, t_collapse=t_collapse,
        alive_at_end=alive, flip_index=flip_index, n_drive=n_drive,
        dt=dt, sample_every=sample_every, H_ext=H_ext, D=D,
        n_files=len(fs))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-temperature summary
    print(f'aggregated {len(fs)} trajectories -> {out_path}')
    t_max = n_drive * dt
    for T in np.unique(T_sub):
        sel = T_sub == T
        n_ev = int(np.isfinite(t_collapse[sel]).sum())
        print(f'  T={T:6.1f} K: {int(sel.sum()):4d} traj, '
              f'{n_ev:4d} collapsed (window {t_max*1e9:.0f} ns)')
    return len(fs)


# -----------------------------------------------------------------------------
def main():
    """Aggregate the #8 scan output directory."""
    in_dir = 'output/stochastic_llgs/validation/skyrmion_arrhenius'
    out_path = ('output/stochastic_llgs/validation/'
                'skyrmion_arrhenius_agg.npz')
    aggregate_skyrmion_arrhenius(in_dir, out_path)


# =============================================================================
if __name__ == '__main__':
    main()
