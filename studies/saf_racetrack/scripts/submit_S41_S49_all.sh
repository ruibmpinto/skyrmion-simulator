#!/bin/bash
# Submit every S41-S49 sweep to the SLURM cluster as separate array jobs.
# Run from the project root: bash studies/saf_racetrack/scripts/submit_S41_S49_all.sh
#
# All array jobs are independent; SLURM will schedule them in
# whatever order. Logs go to logs/sk_<S##>_<jobid>_<arrayidx>.{out,err}.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

for fig in S41 S42 S43a S43b S44 S46a S46b S47 S48 S49; do
    script="studies/saf_racetrack/scripts/submit_${fig}.sh"
    if [[ ! -f "${script}" ]]; then
        echo "WARN: missing ${script}, skipping" >&2
        continue
    fi
    echo ">>> sbatch ${script}"
    sbatch "${script}"
done

echo
echo "All sweeps submitted. Track with:"
echo "  squeue -u \$USER"
echo "Or follow one log live:"
echo "  tail -f logs/sk_S41_*.out"
