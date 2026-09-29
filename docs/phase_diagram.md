# Phase-diagram subsystem

Single source of truth for the (D, H_z) phase-diagram
generator added to the SAF skyrmion simulator.

## 1. Overview

The phase-diagram subsystem produces T=0 ground-state phase
diagrams of the Pt/Co/Ru/Co synthetic antiferromagnet (SAF)
in the (DMI strength, perpendicular field) plane. It
re-uses the existing micromagnetic primitives in
`src/skyrmion_simulator/simulator/` and adds:

- explicit two-dimensional Fourier-space demagnetizing
  fields (replacing the thin-film K_eff approximation
  baked into `parameters._precompute()`),
- a corrected total-energy module,
- two new initial conditions (random and stripe),
- a convergence-driven LLGS relaxation routine,
- an FFT-based phase classifier,
- a parameter-sweep driver and plotting tools.

The package lives in `src/skyrmion_simulator/phase_diagram/` plus two
shared simulator-layer modules `src/skyrmion_simulator/simulator/demag.py`
and `src/skyrmion_simulator/simulator/energy.py`.

## 2. Theory

### 2.1 Total energy at T=0

With all fields measured in Tesla (so the magnetic
constant μ₀ is already absorbed into the conversion
H_T = μ₀ H_(A/m)), the per-site self-energy of a
bilinear contribution H is

    e = -½ Ms (m · H_T)   [J / m³]

and the Zeeman contribution to an external field is

    e_ext = -Ms (m · H_ext_T)   [J / m³].

The total energy summed over both layers is therefore

    E = -½ V Ms Σ_layers m·H_internal
        -   V Ms Σ_layers m·H_Zeeman,

where V = t_Co · a² is the per-site volume of one Co
layer and H_internal aggregates exchange, DMI, anisotropy
(bare K), RKKY, and demag.

The 0.5 prefactor on H_internal prevents double counting
of bilinear self-interactions; it is *absent* on the
Zeeman term because the external field is independent of
the magnetization. Verified: substituting
H_anis_T = (2K/Ms) m_z ẑ recovers the standard density
e_anis = -K m_z².

### 2.2 Demagnetizing field

Each layer is treated as a slab of thickness t_Co with
periodic in-plane lattice. The 2D Fourier-space self
tensor uses the thin-film shape function

    f(k, t) = (1 - exp(-|k|·t)) / (|k|·t),

with limits f → 1 (uniform shape anisotropy
−μ₀Ms m_z ẑ) and f → 0 (no demag for textures much
shorter than the thickness). Cross-layer coupling at gap
d_Ru uses f² · exp(-|k|·d_Ru) and vanishes at k=0 (an
infinite slab produces no field outside itself).

### 2.3 Order parameters and phase signatures

| Phase   | Signature |
|---------|-----------|
| FM_anti | \|⟨m_z⟩\| > 0.95 or \|⟨m_top · m_bot⟩\| > 0.9, *and* m_top · m_bot < −0.9 (antiparallel SAF; H_z-blind) |
| FM_par+ | parallel FM aligned +z: m_top · m_bot > +0.9 and ⟨m_z⟩ > 0 |
| FM_par- | parallel FM aligned −z: m_top · m_bot > +0.9 and ⟨m_z⟩ < 0 |
| iSk     | \|Q\| ∈ [0.5, 1.5] (single or few isolated skyrmions) |
| SkX     | 6-fold FFT angular harmonic *and* \|Q\| / N_periods ≥ 0.5 |
| BX      | 6-fold FFT angular harmonic *but* \|Q\| / N_periods < 0.5 (bubble lattice, topologically trivial) |
| SS      | dominant FFT peak with 2-fold angular harmonic |
| Lab     | dominant FFT ring with no clean angular order |

Q is the topological charge of the top layer (reused from
`skyrmion_simulator.simulator.main.topological_charge`).

## 3. Method

### 3.1 (D, H_z) grid sweep

Grid resolutions are selected by setting `grid_name` at the top of `sweep.main()`:

- `test`   :  4 ×  4 (development / smoke tests)
- `quick`  :  8 ×  8 (between-test-and-coarse fast pass)
- `coarse` : 20 × 20 (production coarse scan)
- `medium` : 40 × 40 (between coarse and fine)
- `fine`   : 50 × 50 (production fine scan)

All presets span `D ∈ [0, 4] mJ/m²` and `H_z ∈ [-0.5, +0.5] T`, sized so that `D/D_c ∈ [0, 1.40]` for the `K = 1.60 MJ/m³` material used in the present runs (`D_c = 2.86 mJ/m²`).

### 3.2 Initial-condition ensemble

Seven ICs per (D, H_z) point:

1. Random sphere, seed 11.
2. Random sphere, seed 22.
3. Random sphere, seed 33.
4. `fm_anti` — antiparallel SAF, top aligned with sign(H_z).
5. `fm_par` — parallel FM, both layers aligned with sign(H_z).
6. SAF skyrmion seed (`saf_skyrmion`).
7. Helical stripe with period λ = 4π·A_ex/D
   (floored at 4·a when D is small).

The random and skyrmion / stripe ICs use antiparallel pairing
so the antiferromagnetic RKKY ground state is respected at
the start. `fm_par` is the only seed that explores the
parallel-FM basin; it is required to find the spin-flop
transition at \|H_z\| > H_RKKY ≈ 0.21 T, above which the
parallel-FM branch beats antiparallel-FM (Zeeman gain >
RKKY cost).

### 3.3 Relaxation

`relax(m_top, m_bot, p, kernels, …)` runs an RK4 LLGS
integrator that calls `effective_field_demag_pair` on each
substep. SOT and `J_current` are forcibly zeroed inside
the loop and restored on exit. Convergence checks fire
every `check_every=1000` steps; both criteria must be
satisfied:

- max\|m × (m × H_eff)\| < `tol_torque` (default 1e-5 T),
- \|ΔE / E\| < `tol_dE` (default 1e-8) over the window.

Optional `alpha_relax` overrides Gilbert damping for an
over-damped quench (and refreshes the dependent
`gamma_p`).

### 3.4 Phase classification

Decision tree in `classifier.classify`:

1. \|⟨m_z⟩\| > 0.95 → FM±.
2. 0.5 ≤ \|Q\| ≤ 1.5 → iSk.
3. Otherwise: locate the dominant non-zero FFT peak at k*;
   sample azimuthal power on a Gaussian ring of width
   `1.5 · dk_grid` around k*; compute the n=2 and n=6
   angular harmonics relative to the mean ring power.
   * P_6 / P_iso > 2 and P_6 > P_2:
     * \|Q\| / N_periods ≥ 0.5 → SkX (true skyrmion lattice).
     * else → BX (bubble lattice; periodic but Q=0 per
       period, no topological protection).
   * P_2 / P_iso > 2 and P_2 > P_6 → SS.
   * Otherwise periodic-but-isotropic → Lab.
4. Fallbacks: \|Q\| > 1.5 without 6-fold ordering → iSk
   (multi-isolated skyrmions); \|Q\| ≈ 0 → Lab; else
   `undetermined`.

`N_periods = (k* L_x / 2π) · (k* L_y / 2π)` estimates the
number of principal periods of the dominant texture in the
field of view. Hexagonal skyrmion lattices satisfy
`|Q| / N_periods ≈ 1`; bubble lattices satisfy
`|Q| / N_periods ≈ 0`.

## 4. Implementation map

| File | Role | Equations |
|------|------|-----------|
| `src/skyrmion_simulator/simulator/demag.py` | Self/inter-layer 2D Fourier-space demag kernels and FFT field eval. | §2.2 |
| `src/skyrmion_simulator/simulator/energy.py` | `total_energy`, `bare_anis_prefactors`. | §2.1 |
| `src/skyrmion_simulator/phase_diagram/fields_demag.py` | `effective_field_demag_pair` composing per-term fields with bare K. | §2.1, §2.2 |
| `src/skyrmion_simulator/phase_diagram/initial_conditions_ext.py` | `random_state`, `stripe_state`. | §3.2 |
| `src/skyrmion_simulator/simulator/relaxation.py` | `relax`, local RK4 calling `llgs_rhs`. | §3.3 |
| `src/skyrmion_simulator/phase_diagram/classifier.py` | `order_parameters`, `classify`. | §3.4 |
| `src/skyrmion_simulator/simulator/params_helper.py` | `make_params(**overrides)`. | — |
| `src/skyrmion_simulator/phase_diagram/sweep.py` | Parallel sweep driver and CLI. | §3.1 |
| `src/skyrmion_simulator/phase_diagram/plot_phase_diagram.py` | Phase map / order-parameter / texture renderers. | §3 |

### 4.1 Bare-K convention

`parameters._precompute()` stores
C_anis = 2 K / Ms − μ₀ Ms (the K_eff thin-film
correction). With explicit demag this correction is
counted twice. Phase-diagram code therefore obtains the
bare prefactor via
`skyrmion_simulator.simulator.energy.bare_anis_prefactors(p)` =
`2 K_top / Ms`, `2 K_bot / Ms`.

The K_eff convention on `p.C_anis_*` is left untouched so
that `skyrmion_simulator.simulator.fields.effective_field` and any code
that calls it without demag continues to work unchanged.

K may be overridden at sweep time via three variables in
the "User Configuration" block at the top of
`skyrmion_simulator.phase_diagram.sweep.main()`:

- `K_top`, `K_bot` (J/m³) — explicit raw anisotropies.
- `Q_PMA` — target quality factor; resolves to
  `K = Q_PMA · (½ μ₀ Ms²) + (½ μ₀ Ms²)`. Mutually
  exclusive with explicit `K_top`/`K_bot`.

The resolved values are stored as `K_top_raw`,
`K_bot_raw` in the NPZ alongside `K_eff_*`, `D_c`, `H_K`.

### 4.2 Import-path correction

The pre-existing simulator modules previously imported
each other as `from src.X import …` while the files live
at `src/skyrmion_simulator/simulator/X.py`. Each broken import was rewritten
to `from skyrmion_simulator.simulator.X import …` (no behavioural change)
so the package is importable from the project root.

## 5. Usage

The sweep and plot entry points take no argparse arguments; configuration lives in the `User Configuration` block at the top of each module's `main()`. Edit the variables in source, then:

### Sweep

```bash
python -m skyrmion_simulator.phase_diagram.sweep                 # local
sbatch studies/saf_racetrack/scripts/submit_sweep_array.sh              # SLURM array
python -m skyrmion_simulator.phase_diagram.aggregate             # merge partials
```

Top-of-`main()` variables: `grid_name`, `nx`, `ny`, `max_steps`, `tol_torque`, `tol_dE`, `alpha_relax`, `workers`, `sims_per_task`, `out_path`, `partial_dir`, `K_top`, `K_bot`, `Q_PMA`. Default output: `output/phase_diagram/<grid_name>.npz`. SLURM-array mode is triggered automatically when `SLURM_ARRAY_TASK_ID` is set.

### Plot

```bash
python -m skyrmion_simulator.phase_diagram.plot_phase_diagram
```

Top-of-`main()` variables: `in_path` *or* `grid_name`, `out_dir`, `units`. With `units='reduced'` axes are `(D/D_c, H_z/H_K)` and the NPZ must contain the `D_c`, `H_K` scalars; with `units='absolute'` axes are mJ/m² and T.

Outputs three PNGs alongside the NPZ: `<grid>_phase_map_<units>.png`, `<grid>_order_params_<units>.png`, `<grid>_textures_<units>.png`.

### NPZ schema

| Key | Shape | Description |
|-----|-------|-------------|
| `D` | (n_D,) | DMI grid in J/m² |
| `H_z` | (n_H,) | Field grid in T |
| `ic_names` | (n_IC,) | IC labels |
| `labels` | (n_phase,) | Phase label table |
| `E`, `Q`, `mz_top`, `mz_bot`, `tau_max`, `n_steps`, `converged`, `label_idx_per_ic` | (n_D, n_H, n_IC) | Per-IC observables |
| `gs_idx`, `gs_label_idx` | (n_D, n_H) | Ground-state IC and phase index |
| `gs_m_top`, `gs_m_bot` | (n_D, n_H, ny, nx, 3) float32 | Ground-state textures |

## 6. Validation log

Each row: tolerance and observed value. Populated as the
corresponding test runs. Commit hashes link to the run.

| # | Test | Expected | Observed | Commit |
|---|------|----------|----------|--------|
| 1 | FM stability (D=0, H_z=±0.5 T, polarity-aligned FM IC, 32²) | label = FM±, \|⟨m_z⟩\|=1, τ=0 | label=FM+ at +H, FM- at -H, ⟨m_z⟩=±1.000, τ=0.00 | _pending_ |
| 2 | Skyrmion stability (default p, `saf_skyrmion` IC, 256²) | Q ≈ ±1, d ≈ 160 nm | _pending_ | |
| 3 | Stripe formation (D=1.5 mJ/m², H_z=0, random IC, 256²) | labyrinth λ ≈ 4π·A/D | _pending_ | |
| 4 | Energy conservation (α→0, short integration, 256²) | \|ΔE/E\| < 1e-4 | _pending_ | |
| 5 | Demag K_eff check (FM uniform, 64²) | ⟨H_demag_z⟩ = -μ₀ Ms = -1.79699 T | -1.796991 T (top), -1.796991 T (bot), match to 1e-6 T; in-plane components identically zero | _pending_ |
| 5a | Demag AP cross-term at k=0 | ⟨H_top⟩ = -μ₀Ms, ⟨H_bot⟩ = +μ₀Ms (independent of other layer at k=0) | -1.796991 / +1.796991 T | _pending_ |
| 6 | Size convergence (128² vs 256² vs 512²) | phase boundary stable | _pending_ | |
| 7 | Commensurability (240² vs 256², SkX) | Q quantized | _pending_ | |
| 8 | Synthetic stripe → SS classifier (128²) | label = SS | label=SS, P_2/P_iso=1.7e9, peak/bg=63 | _pending_ |
| 9 | Synthetic hex SkX classifier (128²) | label = SkX | label=SkX, P_6 ≫ P_2, peak/bg=60 | _pending_ |
| 10 | RKKY energy delta E_AP-E_FM (32², uniform) | -H_RKKY·Ms·V·N·2 = -3.122e-18 J | -3.1219e-18 J (matches to 4 sig figs) | _pending_ |

## 7. Literature anchors

- A. Bogdanov & A. Hubert, *J. Magn. Magn. Mater.* **138**, 255 (1994). Ground-state phase diagram of chiral magnets.
- J. Sampaio *et al.*, *Nat. Nanotechnol.* **8**, 839 (2013). Skyrmion stability in nanostructures.
- S. Rohart & A. Thiaville, *Phys. Rev. B* **88**, 184422 (2013). Skyrmion confinement in ultrathin film disks.
- F. Büttner, I. Lemesh, G. S. D. Beach, *Sci. Rep.* **8**, 4464 (2018). Theory of isolated skyrmions in thin films.
- A. Bernand-Mantel, C. B. Muratov, T. M. Simon, *Phys. Rev. B* **101**, 045416 (2020). Domain walls vs skyrmions in chiral magnets.
- A. Vansteenkiste *et al.*, *AIP Adv.* **4**, 107133 (2014). Mumax3 micromagnetic conventions.

## 8. Known limitations

- **Default material parameters give K_eff ≈ 0** (K_top − μ₀Ms²/2 ≈ 9 kJ/m³, `Q_PMA ≈ 0.007`). With explicit demag the antiparallel SAF "FM" state has zero Zeeman coupling (net moment cancels) and the marginal anisotropy is not enough to suppress demag-driven textures, so FM is *not* the ground state even at large \|H_z\|. To recover a textbook chiral-magnet phase diagram set `Q_PMA = 0.25` (or similar; the physical range for Pt/Co/Ru/Co with engineered interface anisotropy is `Q_PMA ∈ [0.2, 0.5]`) at the top of `skyrmion_simulator.phase_diagram.sweep.main()`. The sweep then resolves `K_top = K_bot ≈ 1.6 MJ/m³`, `D_c ≈ 2.86 mJ/m²`, `H_K ≈ 0.45 T`.
- The 7-IC ensemble explores both antiparallel (`fm_anti`) and parallel (`fm_par`) FM basins, exposing the spin-flop transition at `\|H_z\| > H_RKKY ≈ 0.21 T`. Without the `fm_par` seed the parallel branch is invisible to the sweep.
- T=0 only. Thermal fluctuations would require a stochastic LLG, not implemented.
- Thin-film demag uses the `(1 - e^{-kt})/(kt)` shape function, accurate when the in-plane texture varies on scales ≫ a but not exact for the discretized lattice. Future work: full Newell tensor.
- Periodic boundaries enforce commensurability of stripe / SkX states with the lattice; finite-size shifts of phase boundaries are expected (validation item 7).
- First-order phase boundaries are metastable: the 6-IC ensemble usually finds the ground state but in pathological regions a continuation IC (currently not implemented) would be needed.
- The single-pass FFT classifier uses only the top layer's m_z; it implicitly relies on the AFM SAF coupling being strong enough that `m_bot ≈ −m_top`, which is satisfied for the default `H_RKKY=0.205 T`. Failure mode: weak RKKY plus strong DMI could decouple layers; flagged as `undetermined`.
