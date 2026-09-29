#!/bin/bash
#SBATCH --job-name=tw_equil_ts
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-499

set -euo pipefail
mkdir -p logs

# T6 stage A: equilibrate the intermediate temperatures {55,65,75,85,95}
# K at n_ens=100 for the selected box, seeding the transition study.
# J=0 thermal, so the seeds are shape-independent. 5 T x 100 ens =
# 500 tasks -> array 0-499. Reuses the box's existing m_eq. Writes
# m_thermal_T0{55,65,75,85,95}.0_ens{000..024}.npz into campaign/<tag>/
# (new filenames, no collision with the T={10,50,100,...} seeds).
# Built into build_t6 so in-flight builds are untouched.
# Required exports:
#   TW_CASE=3 (700x500)   TW_TSCAN=1
#   sbatch --export=ALL,TW_CASE=3,TW_TSCAN=1 \
#          studies/saf_racetrack/scripts/submit_equilibrate_tscan.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (0-6)}"
: "${TW_TSCAN:?TW_TSCAN must be exported (1)}"

BIN=src/simulator_cpp/build_ttrans/equilibrate_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
