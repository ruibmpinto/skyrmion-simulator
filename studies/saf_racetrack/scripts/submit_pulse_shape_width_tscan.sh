#!/bin/bash
#SBATCH --job-name=ps_wtscan
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-1499

set -euo pipefail
mkdir -p logs

# T6 stage B: square-only width sweep at the intermediate temperatures
# {55,65,75,85,95} K, driven at two FIXED currents (3e11 = 100 K cap,
# 1e11 = half) so temperature is the only variable across the sweep.
# Sweeps t_pulse {100,200,300,400,600,700} ps, n_ens=25:
# 5 T x 2 J x 6 widths x 25 = 1500 trajectories -> array 0-1499.
# Seeds from the campaign dir (stage A's m_thermal_T0{55..95}.0). Output
# -> pulse_shape/width_tscan/<tag>. Built into build_t6.
# Required exports:
#   PS_CASE=1  PS_STAGE=width_tscan
#   SEED_DIR=output/stochastic_llgs/scan_track_width/campaign/hk36_D0p72_700x500
#   sbatch --export=ALL,PS_CASE=1,PS_STAGE=width_tscan,SEED_DIR=<dir> \
#          studies/saf_racetrack/scripts/submit_pulse_shape_width_tscan.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

: "${PS_CASE:?PS_CASE must be exported}"
: "${PS_STAGE:?PS_STAGE must be exported}"
: "${SEED_DIR:?SEED_DIR must be exported}"

echo "task ${SLURM_ARRAY_TASK_ID} case=${PS_CASE} stage=${PS_STAGE}"
echo "seeds ${SEED_DIR}"
srun ./src/simulator_cpp/build_t6/sweep_pulse_shape
