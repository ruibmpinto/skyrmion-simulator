#!/bin/bash
#SBATCH --job-name=tw_stopgo
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-3599

set -euo pipefail
mkdir -p logs

# T2 stop-and-go drive. Reuses the campaign thermal seeds (stage 2) for
# the selected box and drives them with three 1 ns square pulses at
# 0/2/4 ns (on/off/on/off/on = 5 ns). 6 J x 6 T x 100 ens = 3600
# trajectories; output goes to scan_track_width/stopgo/<tag>/, never
# overwriting the 2 ns DC campaign. Built into build_tw_t2 so the
# in-flight build_tw (T1) is untouched.
# Select the box with TW_CASE (e.g. 3 = hk36_D0p72_700x500) and set
# TW_DRIVE=stopgo:
#   sbatch --export=ALL,TW_CASE=3,TW_DRIVE=stopgo \
#          scripts/submit_scan_track_width_stopgo.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
# FFTW must be loaded at RUNTIME too (libfftw3.so.3). Match the version
# present when build_tw_t2 was built.
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores
: "${TW_CASE:?TW_CASE must be exported (0-6)}"
: "${TW_DRIVE:?TW_DRIVE must be exported (stopgo)}"

BIN=src/simulator_cpp/build_tw_t2/scan_track_width
if [[ ! -x "${BIN}" ]]; then
    echo "error: ${BIN} not built." >&2
    exit 1
fi

srun --unbuffered "${BIN}"
