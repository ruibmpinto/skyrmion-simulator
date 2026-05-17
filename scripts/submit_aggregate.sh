#!/bin/bash
#SBATCH --job-name=skyrmion_aggregate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8192

set -euo pipefail

mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 -m src.phase_diagram.aggregate
