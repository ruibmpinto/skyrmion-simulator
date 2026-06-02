#!/bin/bash
#SBATCH --job-name=sky_arr_dsw
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --array=[0-3199]%3200

# Skyrmion collapse Arrhenius DMI sweep (#8b): the barrier-vs-DMI
# trend of Rohart 2016 Fig. 5(b). Runs the single-layer collapse
# Arrhenius scan at 5 DMI values x 10 temperatures x 64 ensemble
# = 3200 trajectories, all at the same sub-critical 95 mT field.
# The narrow D grid [2.90..3.10] mJ/m^2 stays inside the
# tau-observable window at 95 mT (see scan d_list comment).
# Each array task runs ONE trajectory (60 ns window, 128x128
# lattice, demag off), reading SLURM_ARRAY_TASK_ID for the grid
# index. One core per task is optimal (roll/cross, no BLAS).
#
# Each task writes output/stochastic_llgs/validation/
# skyrmion_arrhenius_dsweep/D####u_T<temp>_ens<idx>.npz.
# After the array completes:
#   bash scripts/aggregate_skyrmion_arrhenius_dsweep.sh   # bundle
#   python3 -m \
#     src.stochastic_llgs.validation.test_skyrmion_arrhenius_dtrend

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 -m src.stochastic_llgs.validation.scan_skyrmion_arrhenius_dsweep
