#!/bin/bash
#SBATCH --job-name=gungordu_aggregate
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --mem-per-cpu=8192

# Aggregate the 2500 Gungordu sweep partials into one NPZ, then
# inject the run's D (the sweep does not store it). Run this on
# the cluster AFTER the submit_gungordu_sweep.sh array finishes;
# then pull the single gungordu_K_H.npz to your Mac and run
# plot_gungordu_overlay / banerjee_critical_field locally.

set -euo pipefail
mkdir -p logs
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load stack/.2024-06-silent gcc/12.2.0
module load python/3.11.6

PARTDIR=output/phase_diagram/validation/gungordu_partials
OUT=output/phase_diagram/validation/gungordu_K_H.npz

# require_all_records=False: each cell ran the 4-IC Gungordu set
# (FM, SP, SkX, SC), but the partial writer tags ic_names with
# the generic 7-IC default, so the aggregator would otherwise
# expect 2500*7 records and flag 2500*3 as "missing". The
# ground state is the min-energy CONVERGED record per cell, so
# it is correct from the 4 real ICs regardless; the phantom IC
# slots are simply never filled.
python3 -c "
from skyrmion_simulator.phase_diagram.aggregate import aggregate
aggregate('${PARTDIR}', '${OUT}', require_all_records=False)
"

# The sweep NPZ does not carry D; the overlay/critical-field
# tools need z['D_run']. Inject the run's fixed Gungordu value
# D = 1.5 mJ/m^2 (MUST match D in run_gungordu_sweep.py).
python3 -c "
import numpy as np
f='${OUT}'
z=dict(np.load(f, allow_pickle=True))
z['D_run']=np.float64(1.5e-3)
np.savez_compressed(f, **z)
print('injected D_run=1.5e-3 into', f)
"

echo 'Aggregation complete:' "${OUT}"
echo 'Pull it to your Mac, then run:'
echo '  python3 -m skyrmion_simulator.phase_diagram.validation.plot_gungordu_overlay'
echo '  python3 -m skyrmion_simulator.phase_diagram.validation.banerjee_critical_field'
