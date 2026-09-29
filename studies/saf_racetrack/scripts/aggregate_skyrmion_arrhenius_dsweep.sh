#!/bin/bash
#SBATCH --job-name=sky_dsw_agg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4096

# Aggregate the skyrmion collapse DMI-sweep (#8b) per-trajectory
# NPZs into one small summary file, so only one file is pulled.
#
# Run AFTER submit_skyrmion_arrhenius_dsweep.sh completes:
#     bash studies/saf_racetrack/scripts/aggregate_skyrmion_arrhenius_dsweep.sh
#
# Reads  : output/stochastic_llgs/validation/
#          skyrmion_arrhenius_dsweep/*.npz
# Writes : output/stochastic_llgs/validation/
#          skyrmion_arrhenius_dsweep_agg.npz
#          (per-trajectory D, T_sub, t_collapse, alive_at_end +
#          the shared window settings). Pure NumPy.
#          test_skyrmion_arrhenius_dtrend.py reads this aggregate
#          when present, else falls back to the per-file glob.

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 - <<'PY'
import glob
import os
import numpy as np

d = 'output/stochastic_llgs/validation/skyrmion_arrhenius_dsweep'
fs = sorted(glob.glob(os.path.join(d, '*.npz')))
if not fs:
    raise RuntimeError(
        f'aggregate_..._dsweep: no NPZ found in {d!r}; run the '
        f'DMI-sweep array first.')

D = np.empty(len(fs), dtype=float)
T_sub = np.empty(len(fs), dtype=float)
t_collapse = np.empty(len(fs), dtype=float)
alive = np.empty(len(fs), dtype=bool)
for i, f in enumerate(fs):
    z = np.load(f)
    D[i] = float(z['D'])
    T_sub[i] = float(z['T_sub'])
    t_collapse[i] = float(z['t_collapse'])
    alive[i] = bool(z['alive_at_end'])

z0 = np.load(fs[0])
out = os.path.join(
    'output/stochastic_llgs/validation',
    'skyrmion_arrhenius_dsweep_agg.npz')
np.savez_compressed(
    out, D=D, T_sub=T_sub, t_collapse=t_collapse,
    alive_at_end=alive,
    n_drive=int(z0['n_drive']), dt=float(z0['dt']),
    sample_every=int(z0['sample_every']),
    H_ext=np.asarray(z0['H_ext'], dtype=float),
    n_files=len(fs))

print(f'aggregated {len(fs)} trajectories -> {out}')
t_max = int(z0['n_drive']) * float(z0['dt'])
for Dv in np.unique(D):
    sel = D == Dv
    n_ev = int(np.isfinite(t_collapse[sel]).sum())
    print(f'  D={Dv*1e3:.2f} mJ/m^2: {int(sel.sum()):4d} traj, '
          f'{n_ev:4d} collapsed (window {t_max*1e9:.0f} ns)')
PY
