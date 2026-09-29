#!/bin/bash
#SBATCH --job-name=sky_arr
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4096
#SBATCH --array=[0-1279]%1280

# Skyrmion Neel-Arrhenius collapse benchmark (#8).
#
# Follows Rohart 2016's Langevin protocol on a SINGLE FM layer:
# an isolated Neel skyrmion under a sub-critical destabilizing
# field (95 mT; Rohart's atomistic 250 mT is supercritical for
# this micromagnetic skyrmion) collapses thermally; tau(T) ->
# Arrhenius fit. The scan is 10 temperatures x 128 ensemble =
# 1280 trajectories. Each array task runs ONE trajectory (its
# own 60 ns observation window on a 128x128 lattice, demag off),
# reading SLURM_ARRAY_TASK_ID for the grid index. The %1280 cap
# lets all run at once if the partition allows; lower it to
# throttle. One core per task is optimal (per-step work is
# roll/cross, no BLAS -> no multi-thread speedup per traj).
#
# Each task writes output/stochastic_llgs/validation/
# skyrmion_arrhenius/T<temp>_ens<idx>.npz. After the array
# completes, fit + gate locally (no aggregation step needed --
# the test globs the per-trajectory NPZs directly):
#
#   python3 -m skyrmion_simulator.stochastic_llgs.validation.test_skyrmion_arrhenius

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

python3 -m skyrmion_simulator.stochastic_llgs.validation.scan_skyrmion_arrhenius
