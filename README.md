# Skyrmion simulator

Micromagnetic and atomistic-style simulator for magnetic skyrmions in
synthetic antiferromagnets (SAF), built to study current-driven skyrmion
motion on a racetrack at finite temperature.

The physical system is a Pt/Co/Ru/Pt/Co/Ru stack: two ferromagnetic Co
layers coupled antiferromagnetically through RKKY exchange. Skyrmions in
the two layers carry opposite topological charge, so the Magnus force
cancels and the skyrmion Hall angle vanishes.

The code solves the Landau-Lifshitz-Gilbert-Slonczewski (LLGS) equation
on a two-layer square lattice with symmetric exchange, interfacial
Dzyaloshinskii-Moriya interaction, uniaxial anisotropy, RKKY interlayer
coupling, dipolar (demagnetizing) fields, Zeeman coupling and
spin-orbit-torque driving. Finite temperature is handled by Brown's
stochastic LLGS equation with a multiplicative Gaussian white-noise
thermal field, integrated in the Stratonovich sense by a stochastic Heun
predictor-corrector.

## Repository layout

Two implementations of the same physics live side by side. The Python
package is the reference: readable, validated against the literature, and
the place where new physics is prototyped. The C++ port is the production
engine used for cluster campaigns.

| Path | Contents |
| --- | --- |
| `src/simulator/` | Python reference solver: lattice, fields, energy, demag (Newell and slab kernels), integrator, pulses, observables |
| `src/stochastic_llgs/` | Finite-temperature layer: thermal field, stochastic Heun integrator, Joule heating, skyrmion tracking, stability classifier |
| `src/phase_diagram/` | Ground-state relaxation sweeps and phase classification |
| `src/orchestrator/` | Run driver, observers and I/O for pulsed-drive experiments |
| `src/plots/` | Shared plotting helpers and figure styles |
| `src/simulator_cpp/` | C++17 port (CMake, 61 targets, 24 CTest suites). See its own `README.md` |
| `src/aux/libnpy/` | Vendored third-party npy/npz I/O. See `src/aux/README.md` |
| `scripts/` | Sweep drivers, analysis and plotting scripts, SLURM submission scripts |
| `tests/` | Cross-implementation regression tests |
| `docs/` | Theory notes, benchmark report and study write-ups (LaTeX and Markdown sources) |

Simulation output is written to `output/` and is not tracked; reference
literature lives in `refs/` and is likewise untracked.

## Requirements

Python side: Python >= 3.10 with `numpy`, `scipy`, `matplotlib` and
`pytest`.

C++ side: CMake >= 3.20, a C++17 compiler, FFTW3 (double precision) and
optionally OpenMP. Full build instructions are in
[`src/simulator_cpp/README.md`](src/simulator_cpp/README.md).

## Running

Deterministic Python run, configured by the variables at the top of
`main()`:

```bash
python -m src.simulator.main
```

Finite-temperature run:

```bash
python -m src.stochastic_llgs.experiments.run_single
```

C++ production run:

```bash
cd src/simulator_cpp
cmake -S . -B build && cmake --build build -j
./build/skyrmion_sim
```

Run configuration is set as plain variables at the top of each entry
point rather than through command-line flags; edit them and re-run. The
C++ sweep and scan binaries follow the same convention.

## Tests and validation

```bash
pytest tests src/simulator/validation src/stochastic_llgs/validation \
    src/phase_diagram/validation
cd src/simulator_cpp && python tests/run_tests.py
```

`tests/` holds cross-implementation regression tests (demag kernel
equivalence, observables, pulse refactor parity). The `validation/`
directories hold physics benchmarks that assert against published
results: muMAG standard problems #4 and #5, the Cortes-Ortuno DMI
standard problem, Bogdanov-Hubert and Rohart-Thiaville skyrmion profiles,
1D domain-wall profiles, FMR dispersion, Rohart 2013 confinement, Rohart
2016 Arrhenius collapse, Tomasello 2018 skyrmion size versus temperature,
Gungordu-Banerjee phase diagrams, and Thiele velocity against Pham 2024.
Thermal correctness is checked by Brown reversal rates, equipartition and
Langevin diffusion tests.

The C++ suite runs 24 CTest binaries that compare against Python-computed
references at `atol=1e-12`, `rtol=1e-10`.

## Documentation

| Document | Subject |
| --- | --- |
| `docs/theory.tex` | Hamiltonian, LLGS equation, discretization, thermal noise |
| `docs/benchmarks/benchmarks.tex` | Validation report against the literature benchmarks above |
| `docs/cpp_port.tex` | Design and verification of the C++ port |
| `docs/phase_diagram.tex` | Ground-state phase diagram methodology |
| `docs/stability_classification.tex` | Skyrmion versus labyrinth discriminant |
| `docs/stochastic_llgs/stochastic_llgs.tex` | Stochastic solver derivation and convergence |
| `docs/reproduction_S41_S49.md` | Reproduction of Pham 2024 supplementary figures S41-S49 |

Compiled PDFs are build products and are not tracked; rebuild them from
the `.tex` sources.

## Code style

Python follows the project style guide: 80-column lines, NumPy-style
docstrings on every module, class and function, no type hints in
signatures, no module-level globals. `flake8` is configured in `.flake8`
and enforced in CI.

## License

MIT. See [LICENSE](LICENSE).

`src/aux/libnpy/` is vendored third-party code under its own license; see
`src/aux/libnpy/LICENSE`.
