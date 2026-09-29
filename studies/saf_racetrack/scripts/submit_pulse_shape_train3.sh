#!/bin/bash
#SBATCH --job-name=ps_train3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=24:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-3121

set -euo pipefail
mkdir -p logs

# T3: three back-to-back pulses (period = t_pulse) instead of one, on
# the full production grid for the selected box. Reuses the campaign
# seeds (T=0 m_eq, T>0 m_thermal); drives 3*t_pulse then a t_pulse
# settle (4*t_pulse = 2 ns window). Output -> pulse_shape/train3/<tag>.
# Built into build_ps_train3 so in-flight pulse builds are untouched.
#
# The array size matches PS_CASE=1 (700x500): 3122 trajectories, so
# --array=0-3121. A different box has a different grid size.
# Required exports:
#   PS_CASE=1  PS_STAGE=train3
#   SEED_DIR=output/stochastic_llgs/scan_track_width/campaign/hk36_D0p72_700x500
# e.g.
#   sbatch --export=ALL,PS_CASE=1,PS_STAGE=train3,SEED_DIR=<dir> \
#          studies/saf_racetrack/scripts/submit_pulse_shape_train3.sh
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/2024-06 gcc/12.2.0
module load fftw/3.3.10

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-2}"

: "${PS_CASE:?PS_CASE must be exported}"
: "${PS_STAGE:?PS_STAGE must be exported}"
: "${SEED_DIR:?SEED_DIR must be exported}"

echo "task ${SLURM_ARRAY_TASK_ID} case=${PS_CASE} stage=${PS_STAGE}"
echo "seeds ${SEED_DIR}"
srun ./src/simulator_cpp/build_ps_train3/sweep_pulse_shape
