#!/bin/bash
#SBATCH --job-name=tw_scan
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-3599%1000

set -euo pipefail
mkdir -p logs

# Stage 3 of the track-width campaign. 6 J x 6 T x 100 ens = 3600
# trajectories; one SLURM array task per trajectory, up to 1000
# concurrent. Each task loads the cached per-(T,ens) thermal state
# m_thermal_T{T}_ens{ens}.npz (written by submit_equilibrate_track_width.sh),
# then drives at its current with newell demag using OpenMP across the
# 16 allocated cores -- no equilibration here. The C++ binary reads
# SLURM_ARRAY_TASK_ID to select its trajectory.
# Select the (K_top, D, box) case with the exported TW_CASE env var
# (must match the stage-2 submission it depends on):
#   sbatch --export=ALL,TW_CASE=<0..3> scripts/submit_scan_track_width.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Match the version
# present when you built (build_tw was built against fftw/3.3.10).
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (0-3)}"

BIN=src/simulator_cpp/build_tw/scan_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"

# Stage 1 is skipped: the relaxed-box snapshot is staged directly as
# campaign/<tag>/m_eq_<nx>x<ny>.npz. Submit stages 2 -> 3 chained, per
# case (TW_CASE in 0..3):
#   for i in 0 1 2 3; do
#     j2=$(sbatch --parsable --export=ALL,TW_CASE=$i \
#          scripts/submit_equilibrate_track_width.sh)
#     sbatch --dependency=afterok:$j2 --export=ALL,TW_CASE=$i \
#          scripts/submit_scan_track_width.sh
#   done
# After all tasks finish, aggregate + plot on a login node:
#   python3 -m scripts.aggregate_sllg scan_track_width
#   python3 -m scripts.plot_track_width
