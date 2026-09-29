#!/bin/bash
#SBATCH --job-name=sk_breathing
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=2048
#SBATCH --array=0-6

set -euo pipefail
mkdir -p logs

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

# One D value per SLURM array task; scipy.fft workers use
# all 16 cores for the per-step demag FFTs (relax phase uses
# the K_eff path so demag is local; FFT workers still help for
# any future demag relax).
export SWEEP_NPROC=1

python3 -m studies.saf_racetrack.scripts.sweep_breathing
