#!/bin/bash
#SBATCH --job-name=tw_equil
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-599%1000

set -euo pipefail
mkdir -p logs

# Stage 2 of the track-width campaign. The J=0 thermal equilibrium is
# current-independent, so it is computed ONCE per (T_sub, ens):
# 6 T x 100 ens = 600 tasks. Each loads the shared T=0 m_eq.npz
# (submit_relax_track_width.sh), equilibrates at its T until the LCC
# size plateaus, and caches m_thermal_T{T}_ens{ens}.npz. The drive
# stage reuses each cached state across all 6 current values.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Match the version
# present when you built (`module list` in the build shell).
module load fftw/3.3.9

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

BIN=src/simulator_cpp/build/equilibrate_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
