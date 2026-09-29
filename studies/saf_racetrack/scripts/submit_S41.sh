#!/bin/bash
#SBATCH --job-name=sk_S41
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-0

set -euo pipefail
mkdir -p logs

# Run from the directory in which sbatch was invoked
# (must be the project root containing src/).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# Single-trajectory sweep: no internal multiprocessing.
# The 16 cores are used by scipy.fft workers=-1 inside
# demag_field for the per-step FFTs.
export SWEEP_NPROC=1

python3 -m studies.saf_racetrack.scripts.sweep_S41_v_time
