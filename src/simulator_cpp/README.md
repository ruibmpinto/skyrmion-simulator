# skyrmion C++ simulator

C++17 port of `src/simulator/`. Outputs are numpy `.npz` archives consumed
by the Python plotting and animation scripts under `scripts/`.

## Dependencies

- CMake >= 3.20
- C++17 compiler (AppleClang, GCC >= 9)
- FFTW3 (double precision). On macOS: `brew install fftw`.
  On Linux: `apt install libfftw3-dev`. FFTW threads optional.
- libnpy (vendored at `src/aux/libnpy`).
- OpenMP optional; the CMake script auto-detects it.

## Build

```bash
cd src/simulator_cpp
cmake -S . -B build
cmake --build build -j
```

The binary is `build/skyrmion_sim`.

## Run

```bash
cd build
./skyrmion_sim
```

Run-time configuration sits at the top of `src/main.cpp`. To change
lattice size, drive current, snapshot cadence, etc., edit the values
between the `Run configuration` comment markers and rebuild. For
parameter sweeps, write a parallel entry-point file that sets the
`Params` fields and links against `skyrmion_core`.

## Output

`./output/snapshots.npz` is written at the end of a run. Keys:

| key            | shape                  | dtype   | meaning                                  |
| -------------- | ---------------------- | ------- | ---------------------------------------- |
| `m_top`        | `(n_frames, ny, nx, 3)`| float64 | top-layer magnetization at each frame    |
| `m_bot`        | `(n_frames, ny, nx, 3)`| float64 | bottom-layer magnetization               |
| `pos_top`      | `(ny, nx, 3)`          | float64 | top-layer site positions (m)             |
| `pos_bot`      | `(ny, nx, 3)`          | float64 | bottom-layer site positions (m)          |
| `step`         | `(n_frames,)`          | int64   | LLGS step index of the frame             |
| `time_s`       | `(n_frames,)`          | float64 | physical time (s) at the frame           |
| `phase_id`     | `(n_frames,)`          | int32   | 0 = relax (J=0), 1 = drive (J on)        |
| `Q_top`        | `(n_frames,)`          | float64 | top-layer topological charge             |
| `Q_bot`        | `(n_frames,)`          | float64 | bottom-layer topological charge          |
| `cx_top`       | `(n_frames,)`          | float64 | top-layer skyrmion center x (m)          |
| `cy_top`       | `(n_frames,)`          | float64 | top-layer skyrmion center y (m)          |
| `diameter_top` | `(n_frames,)`          | float64 | top-layer skyrmion diameter (m)          |
| `D1_top`       | `(n_frames,)`          | float64 | major-axis diameter from second moments  |
| `D2_top`       | `(n_frames,)`          | float64 | minor-axis diameter                      |
| `theta_top`    | `(n_frames,)`          | float64 | ellipse major-axis angle (rad)           |
| `psi_top`      | `(n_frames,)`          | float64 | right-DW in-plane angle (rad)            |
| `params_json`  | `(N,)`                 | uint8   | serialized Params struct                  |

Phase contract:

- `phase_id = 0`, `step = 0`: initial unrelaxed configuration.
- `phase_id = 0`, last frame: relaxed equilibrium.
- `phase_id = 1`, `step = 0`: same magnetization as the last phase-0 frame.
- `phase_id = 1`, last frame: final driven state.

## Visualizing

After a run, generate an mp4 of the m_z field:

```bash
python scripts/animate_simulation.py
```

(see the Python scripts' headers for configuration variables).

## Demag

Three options via `Params.demag_kind`:

| value             | meaning                                                                            |
| ----------------- | ---------------------------------------------------------------------------------- |
| `DemagKind::None` | Default. Local-K_eff path (matches the Python `rhs_local_keff` production run).    |
| `DemagKind::Slab` | Analytic thin-film slab kernel applied via FFTW r2c/c2r.                           |
| `DemagKind::Newell` | mumax3-style finite-prism Newell kernel with Aharoni (0,0) override + convergence gate. |

The Newell kernel matches the Python `src/simulator/demag_newell.py`
to ~1e-13 relative error (verified by `tools/compare_newell_kernel.py`).
`Params.demag_accuracy` (default 8) sets the mumax3 quadrature density
and `Params.demag_tol_conv` (default 2e-2) is the convergence gate
between the requested accuracy and 2x that. Looser tol_conv may be
needed at very small lattices.

To cross-validate a kernel built at custom parameters, build and run
`dump_newell_kernel`:

```bash
cmake --build build --target dump_newell_kernel
./build/dump_newell_kernel              # writes newell_kernel.npz
python src/simulator_cpp/tools/compare_newell_kernel.py
```

## Threading

OpenMP parallelism is applied to per-site loops when available. To
control thread count at run time:

```bash
OMP_NUM_THREADS=8 OMP_PROC_BIND=close OMP_PLACES=cores ./skyrmion_sim
```

FFTW threading is enabled if `libfftw3_threads` is present at build
time.

## Sweep analyses

One executable per `scripts/sweep_*.py` analysis, built alongside the
main simulator:

```
sweep_S41_v_time  sweep_D_S41        sweep_D_S41_local  sweep_S42_J
sweep_S43a_FWHM   sweep_S43b_J_two_D sweep_S44_deformation
sweep_S46a_HRKKY  sweep_S46b_FWHM    sweep_S47_J_config sweep_S48_inertia
sweep_S49_TSH     sweep_breathing
```

Each binary has its grid + run config as named variables at the top of
`main()` (no argparse). It writes the same output layout as the Python
sweeps (`output/sweeps_S41_S49/<analysis>/…npz`), so the existing
`scripts/analyze_*.py` and `scripts/plot_*.py` read the results
unchanged (after the one-line `src/orchestrator/io.py::load_trace` patch that
accepts the C++ uint8-bytes metadata).

Each grid point emits:

- `<grid_id>.npz` — per-frame observable trace (the Python `Trace`
  schema: `t`, `cx_top`, `d_top`, `Q_top`, …).
- `<grid_id>_snapshots.npz` — field snapshots for animation (same
  schema as the simulator's `snapshots.npz`), **only when**
  `dump_snapshots = true` in the binary's config. Replay with
  `scripts/animate_simulation.py`.

Run one grid point or the whole grid:

```bash
# Whole grid, serial:
./build/sweep_S42_J

# One grid point (SLURM array, or local):
SLURM_ARRAY_TASK_ID=3 ./build/sweep_S42_J

# Local parallelism across points:
seq 0 8 | xargs -P 4 -I {} env SLURM_ARRAY_TASK_ID={} ./build/sweep_S42_J
```

The default field model is full FFT demag (Newell kernel) with a
convergence-stop relaxation, matching the Python `use_full_demag=True`
default. Set `use_full_demag = false` (or, for `sweep_D_S41`,
`use_demag = false`) for the faster local-K_eff path. The S49 analytic
Thiele curves and the `unwrap_trajectory` post-processing remain on the
Python side. Stochastic-LLG sweeps are a planned follow-up (the
`sweep::Stepper` interface already accommodates a Heun stochastic
backend with no driver changes).

## Tests

`tests/` contains one parity test per C++ module. Each test loads a
Python-computed reference (`tests/reference/references.npz`) and calls
the analogous C++ function on the same inputs; the per-element
difference must satisfy `max|a - b| <= atol + rtol * max|b|`
(`atol = 1e-12`, `rtol = 1e-10`).

```bash
python src/simulator_cpp/tests/run_tests.py
```

This regenerates the references via the Python implementations,
configures and builds, then runs ctest. Coverage:

| suite                        | functions covered                                                        |
| ---------------------------- | ------------------------------------------------------------------------ |
| `test_pulses`                | ConstantPulse, SquarePulse, GaussianPulse, SuperpositionPulse            |
| `test_parameters`            | precompute, bare_anis_prefactors, effective_anisotropy, critical_dmi, pma_anisotropy_field |
| `test_lattice`               | lattice_positions                                                        |
| `test_initial_conditions`    | skyrmion_profile (both polarities), uniform_state, saf_skyrmion          |
| `test_demag_slab`            | precompute_demag_slab (10 kernel components)                             |
| `test_demag_newell`          | precompute_demag_newell (10 kernel components)                           |
| `test_demag_field`           | DemagState::compute (slab + Newell)                                      |
| `test_fields`                | effective_field, effective_field_demag                                   |
| `test_energy`                | total_energy (slab + Newell)                                             |
| `test_integrator`            | normalize_inplace, llgs_rhs, RHSLocalKeff, RHSDemag, rk4_step (×2 modes) |
| `test_observables`           | topological_charge, skyrmion_center, skyrmion_diameter, skyrmion_ellipse, dw_angle |
| `test_io_npz`                | SnapshotBuffer round-trip                                                |
| `test_simulation`            | run() end-to-end (50 relax + 20 drive steps vs Python)                   |
| `test_sweep`                 | skyrmion_center_pbc, sweep::observe_state (16 scalars) vs Python         |
| `test_edge_cases`            | explicit-raise contract + boundary values for every function (42 cases) |

Two Python harnesses run after ctest (driven by `run_tests.py`):

- `tests/edge_contract.py` — asserts the Python error contract matches
  the C++ `test_edge_cases` raises (where both should raise) and
  documents the cases where C++ is intentionally stricter than Python.
- `tools/sweep_parity.py` — diffs the C++ sweep driver (`run_trace`)
  against Python `run_one` across all 17 observable arrays.

### Explicit-raise policy

No function silently swallows errors. Every invalid input raises (no
default fallbacks). Three inputs are handled more strictly than the
Python reference, which silently returns NaN there:

- `skyrmion_profile`: polarity not in {+1, -1} (Python: no validation).
- `skyrmion_center`: empty core mask, `ws == 0` (Python: NaN).
- `uniform_state`: zero direction vector (Python: NaN).
