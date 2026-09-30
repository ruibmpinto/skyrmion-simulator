# SAF racetrack study

A research application built on the `skyrmion_simulator` package: current-
and temperature-driven skyrmion motion in a Pt/Co/Ru synthetic
antiferromagnet racetrack. It contains the campaign-specific geometry,
drive protocols, stability classification, sweep drivers, analysis,
figures and SLURM submission. None of it is installed with the package.

## Layout

| Folder | Contents |
| --- | --- |
| `orchestrator/` | Single-trajectory driver, integrator factories, per-frame observers, NPZ I/O, shared machinery for the finite-temperature pulsed sweeps |
| `experiments/` | Stochastic production runs: single trajectory, (T, j) scan, radius scan, Arrhenius scan, pair potential, track-width scan |
| `phase_diagram/` | Study-specific phase-diagram runs: the (D, K_top) sweep at zero field and IC and phase visualizations |
| `plots/` | Shared plotting helpers and figure styles |
| `scripts/` | Sweep drivers (`sweep_*`), analysis (`analyze_*`, `aggregate_*`), figures (`plot_*`, `animate_*`) and SLURM submission (`submit_*.sh`) |

## Running

Install the package, then run every study entry point as a module from
the repository root:

```bash
pip install -e .
python -m studies.saf_racetrack.experiments.run_single
python -m studies.saf_racetrack.scripts.figures.plot_track_width
```

Run configuration is set as plain variables at the top of each `main()`.

## Data

Scripts write to and read from `output/` at the repository root. Keep
large campaign data elsewhere by making `output` a symbolic link.

## Cluster runs

The `scripts/submit_*.sh` files are SLURM batch scripts, submitted from
the repository root. Their `module load` lines match the cluster the
campaign ran on; adapt them to yours. `scripts/submit_build.sh` builds the
C++ `relax_track_width` and `scan_track_width` binaries.

## Documentation

- `docs/reproduction_S41_S49.md`: reproduction of the Pham et al. (2024)
  supplementary figures S41 to S49.
- `docs/stochastic_llgs/stochastic_llgs.tex`: stochastic production scans
  and their post-processing pipeline.
- `docs/stability_classification.tex`: the skyrmion versus labyrinth
  stability discriminant.
