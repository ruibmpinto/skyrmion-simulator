#!/bin/bash
#SBATCH --job-name=sllg_pair
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-119

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# pair_potential has 4 r_init x 30 ensemble = 120 trajectories
# with demag ON. Each task initialises two skyrmions and runs
# zero-current dynamics to record r(t).
export SWEEP_NPROC=1

python3 -m studies.saf_racetrack.experiments.pair_potential
