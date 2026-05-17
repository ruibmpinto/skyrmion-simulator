#!/bin/bash

# Submit the array sweep, then chain the aggregator so it
# runs after every array element finishes successfully.
#
# Edit the User Configuration block in
#     src/phase_diagram/sweep.py        and
#     src/phase_diagram/aggregate.py
# (they must agree on GRID_NAME, NX, NY) and the --array
# range in scripts/submit_sweep_array.sh, then run:
#
#     bash scripts/submit_pipeline.sh

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ARRAY_JID=$(sbatch --parsable "${HERE}/submit_sweep_array.sh")
echo "submitted array job ${ARRAY_JID}"

AGG_JID=$(sbatch --parsable \
    --dependency=afterok:${ARRAY_JID} \
    "${HERE}/submit_aggregate.sh")
echo "submitted aggregate job ${AGG_JID} (after array ${ARRAY_JID})"
