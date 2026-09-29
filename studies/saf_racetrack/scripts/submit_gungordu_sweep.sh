#!/bin/bash
#SBATCH --job-name=gungordu_sweep
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=4048
#SBATCH --array=[0-2499]%2500

# Gungordu 2016 Fig. 3 phase-diagram benchmark (also the base
# map for the Banerjee 2014 critical-field cross-check).
#
# The sweep is a 50 x 50 grid in Gungordu's dimensionless
# coordinates (a_s = A_s J/D^2, h_g = H J/D^2), with the 4
# Gungordu candidate-phase ICs per cell (FM, SP, SkX, SC) ->
# 50*50*4 = 10000 tasks. Demag is OFF (Gungordu is a local
# anisotropy-only model); the run uses the production sweep /
# relax / effective_field_demag_pair path with a zero demag
# kernel (demag_kind='none', set inside the runner).
#
# SIMS_PER_TASK=4 makes each array element relax one full cell
# (its 4 ICs are consecutive tasks), so the array spans
# 10000 / 4 = 2500 elements (0-2499), throttled to 192 at once.
# Each element writes output/phase_diagram/validation/
# gungordu_partials/part_<id>.npz.
#
# After the array finishes, aggregate + plot (see the two
# commented commands at the bottom, or submit them as a
# dependent job).

set -euo pipefail

mkdir -p logs

# Run from the directory in which sbatch was invoked.
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# One cell (4 ICs) per array element. The runner reads
# SLURM_ARRAY_TASK_ID and SIMS_PER_TASK for the partition.
export SIMS_PER_TASK=4

python3 -m skyrmion_simulator.phase_diagram.validation.run_gungordu_sweep

# ---------------------------------------------------------------------
# Post-processing (run AFTER the whole array completes; e.g. as a
# dependent job: `sbatch --dependency=afterok:<jobid> ...`):
#
#   python3 -c "from skyrmion_simulator.phase_diagram.aggregate import aggregate; \
#     aggregate('output/phase_diagram/validation/gungordu_partials', \
#               'output/phase_diagram/validation/gungordu_K_H.npz')"
#   python3 -m skyrmion_simulator.phase_diagram.validation.plot_gungordu_overlay
# ---------------------------------------------------------------------
