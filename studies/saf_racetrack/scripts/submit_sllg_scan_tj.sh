#!/bin/bash
#SBATCH --job-name=sllg_tj
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=[0-1349]%1500

set -euo pipefail
mkdir -p logs

# Run from the directory in which sbatch was invoked
# (must be the project root containing src/).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# scan_tj has 9 T_sub x 5 j x 30 ensemble = 1350 trajectories.
# Each SLURM array task runs ONE trajectory. Demag is OFF for
# this scan, so the per-step work has no FFTs to thread; one
# core per task is enough. The %1000 cap throttles concurrency
# so the queue does not flood.
export SWEEP_NPROC=1

python3 -m studies.saf_racetrack.experiments.scan_tj
