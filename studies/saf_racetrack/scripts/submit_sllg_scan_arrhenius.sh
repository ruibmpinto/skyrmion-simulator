#!/bin/bash
#SBATCH --job-name=sllg_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=[0-1799]%2000

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# scan_arrhenius has 9 T x 200 ensemble = 1800 trajectories.
# Each SLURM array task runs ONE zero-drive trajectory; demag is
# OFF so one core per task is enough. The %1000 cap throttles
# concurrency. This is the longest scan (10 ns observation
# window) and the cluster time budget dominates total wall time.
export SWEEP_NPROC=1

python3 -m studies.saf_racetrack.experiments.scan_arrhenius
