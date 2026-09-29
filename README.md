# Skyrmion simulator

A micromagnetic solver for magnetic skyrmions.

It integrates the Landau-Lifshitz-Gilbert-Slonczewski (LLGS) equation on
a square lattice with symmetric exchange, interfacial Dzyaloshinskii-
Moriya interaction, uniaxial anisotropy, dipolar (demagnetizing) fields,
Zeeman coupling and spin-orbit-torque driving. Finite temperature uses
Brown's stochastic LLGS equation with a multiplicative Gaussian
white-noise thermal field, integrated in the Stratonovich sense by a
stochastic Heun predictor-corrector.

The solver runs a single ferromagnetic layer or a coupled bilayer. The
bilayer case adds one term, the RKKY interlayer field, and the study the
code was written for is a synthetic antiferromagnet (SAF): a
Pt/Co/Ru/Pt/Co/Ru stack whose two Co layers carry skyrmions of opposite
topological charge, so the Magnus force cancels and the skyrmion Hall
angle vanishes.

## Code organisation

The code separates into three layers of decreasing generality.

**Solver.** `src/simulator/` and `src/stochastic_llgs/` implement
general micromagnetics. Each field term acts on one layer —
`exchange_field(m, ...)`, `dmi_field(m, ...)`, `anisotropy_field(m, ...)`,
`zeeman_field(...)` — and `effective_field` assembles them for one
layer. `llgs_rhs` and `rk4_step_single` advance one layer. No module in
this layer references the racetrack geometry or the campaign protocols.

**Bilayer.** The interlayer coupling is one term, `rkky_field(m_other,
H_RKKY)`, together with the pair wrappers
(`effective_field_demag_pair`, `rk4_step`, `total_energy`) and the
two-block demag kernel. Passing `None` as the partner layer reduces
these to the single-layer solver; passing an array gives the bilayer.

**Application.** `src/phase_diagram/`, `src/orchestrator/`, most of
`scripts/`, and the racetrack geometry, track-width campaign, pulse
protocols, equilibration-to-plateau criterion and stability classifier
constitute one study built on the solver rather than part of it.

The generality of the solver layer is established by the single-layer
literature benchmarks listed below, which run with no partner layer and
no RKKY coupling. `src/simulator/validation/_helpers.py` provides the
corresponding single-layer entry points `make_single_fm_params`,
`relax_single_fm` and `integrate_single_fm`.

| Path | Layer | Contents |
| --- | --- | --- |
| `src/simulator/` | solver | Lattice, field terms, energy, demag (Newell and slab kernels), RK4 integrator, pulse waveforms, observables |
| `src/stochastic_llgs/` | solver | Thermal field, stochastic Heun integrator, Joule heating, skyrmion tracking |
| `src/phase_diagram/` | application | Ground-state relaxation sweeps and phase classification |
| `src/orchestrator/` | application | Run driver, observers and I/O for pulsed-drive experiments |
| `src/plots/` | application | Shared plotting helpers and figure styles |
| `src/simulator_cpp/` | production | C++17 port of the SAF path, for cluster campaigns. See its own `README.md` |
| `src/aux/libnpy/` | vendored | Third-party npy/npz I/O. See `src/aux/README.md` |
| `scripts/` | application | Sweep drivers, analysis and plotting, SLURM submission |
| `tests/` | — | Cross-implementation regression tests |
| `docs/` | — | Theory notes, benchmark report, study write-ups (LaTeX and Markdown) |

The Python package is the reference implementation: readable, validated
against the literature, and where new physics is prototyped. The C++ port
covers the SAF production path only and is pinned to Python by parity
tests; it is not a general solver. See `src/simulator_cpp/README.md` for
what it does and does not support.

Simulation output is written to `output/` and is not tracked; reference
literature lives in `refs/` and is likewise untracked.

## Single layer or bilayer

The partner layer is the second argument everywhere it appears. `None`
means there is no second layer:

```python
# Single ferromagnetic layer: no RKKY term, self-demag only.
H = effective_field(m, None, p.C_ex, p.C_dmi, p.C_anis_top,
                    p.H_ext, p.H_RKKY)
H_top, _ = effective_field_demag_pair(m_top, None, p, kernels)

# Coupled bilayer: RKKY plus the inter-layer demag block.
H_top, H_bot = effective_field_demag_pair(m_top, m_bot, p, kernels)
```

The parameter names carry the SAF stack the code was built for
(`K_top`/`K_bot`, `t_Co`, `d_Ru`, `H_RKKY`), and `default_params()`
returns that stack's measured values. They are defaults, not
assumptions: override them for any other material.

## Requirements

Everything for both halves, including CMake and FFTW3:

```bash
conda env create -f environment.yml
conda activate skyrmion-simulator
```

Python only:

```bash
pip install -r requirements.txt
```

The Python side needs `numpy`, `scipy`, `matplotlib` and `pytest`. The
C++ side additionally needs CMake >= 3.20, a C++17 compiler, FFTW3
(double precision, >= 3.3.8) and optionally OpenMP; `pip` cannot supply
those, so use the conda environment or a system package manager. Full
build instructions are in
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
python -m pytest tests src/simulator/validation \
    src/stochastic_llgs/validation src/phase_diagram/validation
cd src/simulator_cpp && python tests/run_tests.py
```

Invoke pytest as `python -m pytest` from the repository root: the tests
import `src.*` and rely on the root being on `sys.path`.

`tests/` holds cross-implementation regression tests (demag kernel
equivalence, observables, pulse refactor parity). The `validation/`
directories hold physics benchmarks that assert against published
results.

Single ferromagnetic layer, no RKKY coupling. These exercise the solver
layer alone:

| Benchmark | Reference | Single-layer route |
| --- | --- | --- |
| `test_mumag_sp4.py` | muMAG standard problem #4 | `rk4_step_single` |
| `test_mumag_sp5.py` | muMAG standard problem #5, STT vortex | `rk4_step_single` |
| `test_dmi_standard_problem.py` | Cortes-Ortuno 2018 DMI standard problem | `H_RKKY = 0` |
| `test_skyrmion_profile_bh.py` | Bogdanov-Hubert / Rohart-Thiaville profile | `relax_single_fm` |
| `test_dw_profile_1d.py` | Analytic 1D domain-wall width | `rk4_step_single` |
| `test_fmr_dispersion.py` | Kittel FMR dispersion | `integrate_single_fm` |
| `test_confined_skyrmion_radius_rt2013.py` | Rohart 2013 confinement in a dot | `H_RKKY = 0`, disk mask |
| `test_skyrmion_size_vs_T_tomasello2018.py` | Tomasello 2018 size versus temperature | reuses the Rohart 2013 dot |
| `test_skyrmion_arrhenius.py` | Rohart 2016 Arrhenius collapse | single FM layer |

Bilayer: `test_thiele_v_sot.py` (Thiele velocity against Pham 2024) and
the Gungordu-Banerjee phase diagrams.

Thermal correctness: Brown reversal rates, equipartition and Langevin
diffusion, on a macrospin.

The C++ suite runs 24 CTest binaries. These are parity tests against
Python-computed references at `atol=1e-12`, `rtol=1e-10`. They are not
independent literature benchmarks; the literature comparisons above
exist only in the Python implementation.

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

## Citation

If you use this code in academic work, please cite it:

```bibtex
@software{barreira_skyrmion_simulator_2026,
  author  = {Barreira, Rui},
  title   = {{skyrmion-simulator}: Micromagnetic and stochastic {LLGS}
             simulator for skyrmions in synthetic antiferromagnets},
  year    = {2026},
  version = {1.0.0},
  url     = {https://github.com/ruibmpinto/skyrmion-simulator},
  license = {MIT}
}
```

The same metadata is in [`CITATION.cff`](CITATION.cff), which GitHub
exposes through the "Cite this repository" button.

## License

MIT. See [LICENSE](LICENSE).

`src/aux/libnpy/` is vendored third-party code under its own license; see
`src/aux/libnpy/LICENSE`.

The MIT license does not cover the published figures reproduced for
benchmark comparison. They remain the copyright of their publishers and
are included solely for scientific comparison:

| File | Source |
| --- | --- |
| `docs/benchmarks/refs/ref_nist_sp4.png` | muMAG Standard Problem #4, NIST, <https://www.ctcms.nist.gov/~rdm/mumag.org.html> |
| `docs/benchmarks/refs/ref_pham_s49.png` | V. T. Pham et al., Science 384, 307 (2024), Fig. S49, [doi:10.1126/science.add5751](https://doi.org/10.1126/science.add5751) |
| `docs/benchmarks/refs/ref_gungordu_fig3.png` | U. Güngördü et al., Phys. Rev. B 93, 064428 (2016), Fig. 3, [doi:10.1103/PhysRevB.93.064428](https://doi.org/10.1103/PhysRevB.93.064428) |
| `docs/benchmarks/refs/ref_tomasello_fig1b.png`, `docs/figures/validation/tomasello_fig1b_digitization.png`, `docs/figures/validation/tomasello_fig2_digitization.png` | R. Tomasello et al., Phys. Rev. B 97, 060402(R) (2018), Figs. 1(b) and 2, [doi:10.1103/PhysRevB.97.060402](https://doi.org/10.1103/PhysRevB.97.060402) |
