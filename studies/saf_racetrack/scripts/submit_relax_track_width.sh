#!/bin/bash
#SBATCH --job-name=tw_relax
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4000

set -euo pipefail
mkdir -p logs

# Stage 1 of the track-width campaign. The T=0, J=0 racetrack-demag
# equilibrium is identical for every (T_sub, j) cell, so it is relaxed
# ONCE here (64 cores) and written directly into the campaign dir the
# later stages read: campaign/<tag>/m_eq_<nx>x<ny>.npz. Stages 2 (equil)
# and 3 (scan) load that field.
# Select the (K_top, D, box, a) case with the exported TW_CASE env var
# (must match the stage-2/3 submissions it feeds):
#   sbatch --export=ALL,TW_CASE=<idx> studies/saf_racetrack/scripts/submit_relax_track_width.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Match the version
# present when you built (build_tw was built against fftw/3.3.10).
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (case index)}"

# Build once on a login node before submitting (all three stages share
# build_tw):
#   cd src/simulator_cpp && cmake -S . -B build_tw \
#     && cmake --build build_tw -j --target \
#          relax_track_width equilibrate_track_width scan_track_width
BIN=src/simulator_cpp/build_tw/relax_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
