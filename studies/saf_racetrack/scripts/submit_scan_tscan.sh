#!/bin/bash
#SBATCH --job-name=tw_scan_ts
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-499

set -euo pipefail
mkdir -p logs

# Transition-temperature DC study: constant J = 3e11 drive over the
# intermediate temperatures {105,110,115,120,125} K for the selected
# box, to locate the temperature at which the skyrmion collapses.
# 5 T x 1 J x 100 ens = 500 trajectories -> array 0-499. Seeds are the
# TW_TSCAN equilibrate outputs in campaign/<tag>/; drives are written to
# campaign/<tag>/ttrans/ (a separate dir, so aggregate_sllg sees a clean
# 5 T x 1 J grid). Built into build_ttrans.
# Required exports:
#   TW_CASE=3 (700x500)   TW_TSCAN=1
#   sbatch --export=ALL,TW_CASE=3,TW_TSCAN=1 studies/saf_racetrack/scripts/submit_scan_tscan.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (0-6)}"
: "${TW_TSCAN:?TW_TSCAN must be exported (1)}"

BIN=src/simulator_cpp/build_ttrans/scan_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
