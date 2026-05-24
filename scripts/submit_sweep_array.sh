#!/bin/bash
#SBATCH --job-name=skyrmion_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=36:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=[0-2699]%192

set -euo pipefail

mkdir -p logs

# Run from the directory in which sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 -m src.phase_diagram.sweep
