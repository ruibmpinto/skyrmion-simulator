#!/bin/bash
#SBATCH --job-name=sky_arr_agg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4096

# Aggregate the skyrmion Neel-Arrhenius scan (#8) per-trajectory
# NPZs into a single small summary file, so only one file needs
# to be pulled to the local machine for the Arrhenius fit.
#
# Run AFTER the submit_skyrmion_arrhenius.sh array completes,
# either on a login node:
#     bash scripts/aggregate_skyrmion_arrhenius.sh
# or as a dependent job:
#     sbatch --dependency=afterok:<arrayjobid> \
#            scripts/aggregate_skyrmion_arrhenius.sh
#
# Reads  : output/stochastic_llgs/validation/skyrmion_arrhenius/T*.npz
# Writes : output/stochastic_llgs/validation/skyrmion_arrhenius_agg.npz
#          (per-trajectory T_sub, t_collapse, alive_at_end,
#          flip_index + the shared run settings). Pure NumPy,
#          no matplotlib. test_skyrmion_arrhenius.py reads this
#          aggregate when present, else falls back to the
#          per-trajectory glob.

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 - <<'PY'
import glob
import os
import numpy as np

d = 'output/stochastic_llgs/validation/skyrmion_arrhenius'
fs = sorted(glob.glob(os.path.join(d, 'T*.npz')))
if not fs:
    raise RuntimeError(
        f'aggregate_skyrmion_arrhenius: no T*.npz found in '
        f'{d!r}; run the scan array first.')

T_sub = np.empty(len(fs), dtype=float)
t_collapse = np.empty(len(fs), dtype=float)
alive = np.empty(len(fs), dtype=bool)
flip_index = np.empty(len(fs), dtype=int)
for i, f in enumerate(fs):
    z = np.load(f)
    T_sub[i] = float(z['T_sub'])
    t_collapse[i] = float(z['t_collapse'])
    alive[i] = bool(z['alive_at_end'])
    flip_index[i] = int(z['flip_index'])

# Shared run settings (taken from the first record; the scan
# writes identical values to every file).
z0 = np.load(fs[0])
out = os.path.join(
    'output/stochastic_llgs/validation',
    'skyrmion_arrhenius_agg.npz')
np.savez_compressed(
    out,
    T_sub=T_sub,
    t_collapse=t_collapse,
    alive_at_end=alive,
    flip_index=flip_index,
    n_drive=int(z0['n_drive']),
    dt=float(z0['dt']),
    sample_every=int(z0['sample_every']),
    H_ext=np.asarray(z0['H_ext'], dtype=float),
    D=float(z0['D']),
    n_files=len(fs))

# Quick per-temperature summary to the log.
print(f'aggregated {len(fs)} trajectories -> {out}')
t_max = int(z0['n_drive']) * float(z0['dt'])
for T in np.unique(T_sub):
    sel = T_sub == T
    n_ev = int(np.isfinite(t_collapse[sel]).sum())
    print(f'  T={T:6.1f} K: {int(sel.sum()):4d} traj, '
          f'{n_ev:4d} collapsed (window {t_max*1e9:.0f} ns)')
PY
