#!/bin/bash
#SBATCH --job-name=sky_sizeT
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --array=[0-167]%168

# Finite-T skyrmion-size scan (#9b): Tomasello 2018 Fig. 1(b)
# symbols (mean +/- std diameter vs temperature) in a confined
# dot, at H = 0/25/50 mT. Grid = 3 fields x 7 temperatures x 8
# seeds = 168 trajectories. Each array task runs ONE trajectory
# (m(T)-scaled parameters, free-BC dot, single-layer Brown-noise
# heun_stochastic_step, thermalise then sample R_sk(t)), reading
# SLURM_ARRAY_TASK_ID for the grid index. One core per task is
# optimal (roll/cross, no BLAS).
#
# Each task writes output/phase_diagram/validation/
# skyrmion_size_thermal/H<field>_T<temp>_ens<seed>.npz. After
# the array completes:
#   bash scripts/submit_aggregate_skyrmion_size_thermal.sh
#   python3 -m \
#     src.phase_diagram.validation.test_skyrmion_size_thermal

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 -m src.phase_diagram.validation.scan_skyrmion_size_thermal
