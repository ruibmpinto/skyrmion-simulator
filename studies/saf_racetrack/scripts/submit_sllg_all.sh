#!/bin/bash
# Submit every stochastic-LLGS production scan (Q1-Q4) to the SLURM cluster
# as separate array jobs. Run from the project root:
#   bash studies/saf_racetrack/scripts/submit_sllg_all.sh
#
# Each array job is independent; SLURM schedules them
# in whatever order. Logs land at
# logs/sllg_<scan>_<jobid>_<arrayidx>.{out,err}.
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
mkdir -p logs

for scan in scan_tj scan_radius pair_potential scan_arrhenius; do
    script="studies/saf_racetrack/scripts/submit_sllg_${scan}.sh"
    if [[ ! -f "${script}" ]]; then
        echo "WARN: missing ${script}, skipping" >&2
        continue
    fi
    echo ">>> sbatch ${script}"
    sbatch "${script}"
done

echo
echo "All stochastic-LLGS scans submitted. Track with:"
echo "  squeue -u \$USER"
echo "Follow one log live:"
echo "  tail -f logs/sllg_tj_*.out"
