#!/bin/bash
#SBATCH --job-name=tw_scan
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
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
# Select the (K_top, D, box, a) case with the exported TW_CASE env var
# (must match the stage-2 submission it depends on):
#   sbatch --export=ALL,TW_CASE=<0..6> studies/saf_racetrack/scripts/submit_scan_track_width.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Match the version
# present when you built (build_tw was built against fftw/3.3.10).
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (0-6)}"

BIN=src/simulator_cpp/build_tw/scan_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"

# Full 3-stage chain for one case (relax -> equilibrate -> scan). The
# relax stage now writes directly into campaign/<tag>/, so it is part of
# the chain (no manual staging). Example: the a=3 nm lattice test at
# TW_CASE=6:
#   C=6
#   j1=$(sbatch --parsable --export=ALL,TW_CASE=$C \
#        studies/saf_racetrack/scripts/submit_relax_track_width.sh)
#   j2=$(sbatch --parsable --dependency=afterok:$j1 \
#        --export=ALL,TW_CASE=$C studies/saf_racetrack/scripts/submit_equilibrate_track_width.sh)
#   sbatch --dependency=afterok:$j2 --export=ALL,TW_CASE=$C \
#        studies/saf_racetrack/scripts/submit_scan_track_width.sh
# After all tasks finish, aggregate + plot on a login node:
#   python3 -m studies.saf_racetrack.scripts.aggregate_sllg scan_track_width
#   python3 -m studies.saf_racetrack.scripts.plot_track_width
