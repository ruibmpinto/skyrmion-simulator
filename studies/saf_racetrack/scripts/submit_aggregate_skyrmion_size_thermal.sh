#!/bin/bash
#SBATCH --job-name=sky_sizeT_agg
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:20:00
#SBATCH --mem-per-cpu=4096

# Aggregate the finite-T skyrmion-size scan (#9b) per-trajectory
# NPZs into one small summary file (one (R_sk_mean, R_sk_std)
# pair per trajectory), so only one file is pulled.
#
# Run AFTER submit_skyrmion_size_thermal.sh completes:
#     bash studies/saf_racetrack/scripts/submit_aggregate_skyrmion_size_thermal.sh
#
# Reads  : output/phase_diagram/validation/
#          skyrmion_size_thermal/*.npz
# Writes : output/phase_diagram/validation/
#          skyrmion_size_thermal_agg.npz (per-trajectory H, T,
#          seed, R_sk_mean, R_sk_std). Pure NumPy.
#          test_skyrmion_size_thermal.py reads this aggregate
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

d = 'output/phase_diagram/validation/skyrmion_size_thermal'
fs = sorted(glob.glob(os.path.join(d, '*.npz')))
if not fs:
    raise RuntimeError(
        f'aggregate_..._size_thermal: no NPZ in {d!r}; run the '
        f'thermal scan array first.')

H = np.empty(len(fs), dtype=float)
T = np.empty(len(fs), dtype=float)
seed = np.empty(len(fs), dtype=np.int64)
R_sk_mean = np.empty(len(fs), dtype=float)
R_sk_std = np.empty(len(fs), dtype=float)
for i, f in enumerate(fs):
    z = np.load(f)
    H[i] = float(z['H'])
    T[i] = float(z['T'])
    seed[i] = int(z['seed'])
    R_sk_mean[i] = float(z['R_sk_mean'])
    R_sk_std[i] = float(z['R_sk_std'])

z0 = np.load(fs[0])
out = os.path.join(
    'output/phase_diagram/validation',
    'skyrmion_size_thermal_agg.npz')
np.savez_compressed(
    out, H=H, T=T, seed=seed, R_sk_mean=R_sk_mean,
    R_sk_std=R_sk_std, R_dot=float(z0['R_dot']),
    n_files=len(fs))

print(f'aggregated {len(fs)} trajectories -> {out}')
for Hv in np.unique(H):
    for Tv in np.unique(T[H == Hv]):
        sel = (H == Hv) & (T == Tv)
        gm = float(np.mean(R_sk_mean[sel]))
        print(f'  H={Hv*1e3:4.0f} mT  T={Tv:6.1f} K: '
              f'{int(sel.sum())} seeds, '
              f'<R_sk>={gm*1e9:5.1f} nm')
PY
