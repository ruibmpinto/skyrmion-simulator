#!/bin/bash
#SBATCH --job-name=tw_relax
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048

set -euo pipefail
mkdir -p logs

# Stage 1 of the track-width campaign. The T=0, J=0 newell-demag
# equilibrium is identical for every (T_sub, j) cell, so it is relaxed
# ONCE here (64 cores) and written to
# output/stochastic_llgs/scan_track_width/m_eq.npz. The drive array
# (submit_scan_track_width.sh) loads that field.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Use the SAME
# version that was present when you built; check `module spider fftw`
# and `module list` in your build shell, then pin it here.
module load fftw/3.3.9

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-64}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

# Build once on a login node before submitting:
#   cd src/simulator_cpp && cmake -S . -B build \
#     && cmake --build build -j --target relax_track_width scan_track_width
BIN=src/simulator_cpp/build/relax_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
