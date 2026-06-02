#!/bin/bash
#SBATCH --job-name=sllg_rad
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

# scan_radius has 4 (D, H_z) cells x 30 ensemble = 120
# trajectories with demag ON. Each per-step FFT in
# effective_field_demag_pair uses the 8 cores via the numpy
# FFT backend; no extra Python-level parallelism.
export SWEEP_NPROC=1

python3 -m src.stochastic_llgs.experiments.scan_radius
