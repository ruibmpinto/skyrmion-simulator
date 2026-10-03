"""Aggregate the skyrmion collapse DMI sweep (benchmark #8b).

Collects the per-trajectory NPZs written by
`scan_skyrmion_arrhenius_dsweep` into one small summary file, so
only one file needs to be pulled from the cluster.
`test_skyrmion_arrhenius_dtrend` reads this aggregate when present.

Reads  : <in_dir>/*.npz
Writes : <out_path> with per-trajectory D, T_sub, t_collapse,
         alive_at_end and the shared window settings.

Run with:
    python -m \
    skyrmion_simulator.stochastic_llgs.validation.aggregate_skyrmion_arrhenius_dsweep

Functions
---------
aggregate_skyrmion_arrhenius_dsweep
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


def aggregate_skyrmion_arrhenius_dsweep(in_dir, out_path):
    """Collect the #8b per-trajectory NPZs into one summary NPZ.

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
            f'aggregate_skyrmion_arrhenius_dsweep: no NPZ found in '
            f'{in_dir!r}; run the DMI-sweep array first.')
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-trajectory records
    D = np.empty(len(fs), dtype=float)
    T_sub = np.empty(len(fs), dtype=float)
    t_collapse = np.empty(len(fs), dtype=float)
    alive = np.empty(len(fs), dtype=bool)
    for i, f in enumerate(fs):
        with np.load(f) as z:
            D[i] = float(z['D'])
            T_sub[i] = float(z['T_sub'])
            t_collapse[i] = float(z['t_collapse'])
            alive[i] = bool(z['alive_at_end'])
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Shared window settings, identical in every file of the scan
    with np.load(fs[0]) as z0:
        n_drive = int(z0['n_drive'])
        dt = float(z0['dt'])
        sample_every = int(z0['sample_every'])
        H_ext = np.asarray(z0['H_ext'], dtype=float)
    np.savez_compressed(
        out_path, D=D, T_sub=T_sub, t_collapse=t_collapse,
        alive_at_end=alive, n_drive=n_drive, dt=dt,
        sample_every=sample_every, H_ext=H_ext, n_files=len(fs))
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Per-DMI summary
    print(f'aggregated {len(fs)} trajectories -> {out_path}')
    t_max = n_drive * dt
    for Dv in np.unique(D):
        sel = D == Dv
        n_ev = int(np.isfinite(t_collapse[sel]).sum())
        print(f'  D={Dv*1e3:.2f} mJ/m^2: {int(sel.sum()):4d} traj, '
              f'{n_ev:4d} collapsed (window {t_max*1e9:.0f} ns)')
    return len(fs)


# -----------------------------------------------------------------------------
def main():
    """Aggregate the #8b scan output directory."""
    in_dir = 'output/stochastic_llgs/validation/skyrmion_arrhenius_dsweep'
    out_path = ('output/stochastic_llgs/validation/'
                'skyrmion_arrhenius_dsweep_agg.npz')
    aggregate_skyrmion_arrhenius_dsweep(in_dir, out_path)


# =============================================================================
if __name__ == '__main__':
    main()
