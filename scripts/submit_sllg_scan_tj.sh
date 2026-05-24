#!/bin/bash
#SBATCH --job-name=sllg_tj
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-249

set -euo pipefail
mkdir -p logs

# Run from the directory in which sbatch was invoked
# (must be the project root containing src/).
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# scan_tj has 5 T_sub x 5 j x 10 ensemble = 250 trajectories.
# Each SLURM array task runs ONE trajectory; the 8 cores are
# consumed by numpy/BLAS threading inside the per-step kernel
# operations. No internal multiprocessing on top of that.
export SWEEP_NPROC=1

python3 -m src.stochastic_llgs.production.scan_tj
