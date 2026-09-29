# Simulator validation report

Cross-checks of the SAF skyrmion simulator (`src/skyrmion_simulator/simulator/`),
the phase-diagram pipeline (`src/skyrmion_simulator/phase_diagram/`), and the
stochastic-LLGS module (`src/skyrmion_simulator/stochastic_llgs/`) against
published references. Each test lives in a `validation/`
sub-folder of the corresponding module and is runnable as
`python -m <module>.validation.<test>`.

For every benchmark this report records:
- the canonical reference (paper + figure/equation + parameter
  set),
- the material parameters actually used by the test,
- the simulator setup (geometry, boundary conditions, demag,
  integrator),
- the measured value vs. the reference with the acceptance
  tolerance,
- how (and why) our run deviates from the reference, including
  any loosened tolerance or documented limitation.

A guiding rule for the whole suite: **the benchmark wires up
its own parameters and geometry, but the physics it exercises
is the production code in `src/skyrmion_simulator/simulator` / `src/skyrmion_simulator/stochastic_llgs`
(field assembly, integrators, torques).** Where a benchmark
seems to re-implement physics it is a thin wrapper delegating
to a production function; this is noted per test.

---

## Summary table

| #  | Benchmark | Reference | Measured | Tol | Status |
|---:|---|---|---|---:|:---|
|  1 | Cortés-Ortuño 2018 DMI SP (disc, free-BC, Cnv DMI) | r_sk = 22.03 nm (ODE); 21.87 (OOMMF/Fidimag); 22.10 (mumax3) | r_sk = 22.61 nm; profile RMS 0.11 | 5% / 15% | **PASS** |
|  2 | Bogdanov–Hubert / RT 2013 Eq.18 isolated skyrmion (PBC, no demag) | R_s = 9.46 nm (RT Eq.18, D/D_c=0.825) | R_s = 9.32 nm; shape RMS 0.012 | 10% | **PASS** |
|  3 | NIST muMAG SP#4 (Permalloy, Field 1) | ⟨m_x⟩=0 first crossing ≈ 136 ps | 136.66 ps | 10% | **PASS** |
|  4 | NIST muMAG SP#5 (Zhang–Li STT vortex) | Najafi 2009: Δx=−1.2, Δy=−14.7 nm | Δx=−1.21, Δy=−12.18 nm (dist 2.52 nm) | 3 nm | **PASS** |
|  5 | Güngördü 2016 Fig.3 phase diagram | FM / SkX / SP / SC regions | FM + modulated easy-plane OK; SkX ~2.6% (under-resolved) | qual. | **PASS (qual.)** |
|  6 | Banerjee 2014 Fig.1b FM↔SkX critical field | h_c ≈ 1.03–1.52 | h_c ≈ 0.18–0.96 (37–82% low) | 20% | **FAIL (documented)** |
|  7 | RT 2013 Fig.4(d) confined skyrmion radius | R_s ≈ 25 nm (R=50 nm dot, D/D_c=1.25) | R_s = 27.6 nm | 25% | **PASS** |
|  8 | Skyrmion Néel-Arrhenius collapse, j=0 (single FM layer) | Rohart 2016: Arrhenius form; ΔE=26±4 meV, τ₀=0.22 ns | R²=0.983; ΔE=11.5 meV; τ₀=0.35 ns | form+τ₀ | **PASS** |
| 8b | ΔE-vs-DMI trend (Rohart 2016 Fig.5b) | ΔE rises with D | ΔE 5.7→31.5 meV over D=2.95→3.10 (monotone) | trend | **PASS** |
| 10 | Thiele v_SOT(R/Δ) & v_TSH vs Pham 2024 Fig.S49 | closed forms (Pham supp §1.7) | analytic 2e-16; LLGS spot-check 1.5% | see §10 | **PASS** |
| 11 | 1D domain-wall width | Δ = √(A/K_eff) | Δ = 12.6371 nm (analytic 12.6491) | 0.5% | **PASS** |
| 12 | Uniform-mode FMR frequency | f = γ H_K / 2π, H_K = 2K_eff/M_s | rel.err < 5% (≈35 GHz) | 5% | **PASS** |
| 14 | Free-BC demag kernel (infrastructure) | thin-slab N_zz ≈ 1 | interior H_z/μ₀M_s = −0.984 | — | **PASS** |
|  9a | Skyrmion-size T-scaling, deterministic (Tomasello 2018 Fig.1b) | H=0 expansion ratio ~3.3 | R_sk(300)/R_sk(0) = 2.51 (converged) | 30% | **PASS** |
| 9b | Skyrmion-size T-scaling, thermal (Fig.1b symbols) | <D_sk>±std vs T, ratio ~3.3 | ratio 4.76 (high-T over-expansion vs converged 2.51) | 30% | **FAIL (qual. OK)** |

Supporting stochastic-infrastructure gates (underpin #8):
Brown Néel-Brown reversal, Langevin function, spin-wave
equipartition, T=0 deterministic-limit — all **PASS** (§C).

---

## A. Simulator-core micromagnetics

### 1. Cortés-Ortuño 2018 interfacial-DMI standard problem
- **Reference**: Cortés-Ortuño et al., *Sci. Rep.* / proposal
  (2018), 2D interfacial (C_nv) DMI standard problem, disc
  geometry. ODE theory r_sk = 22.03 nm; OOMMF/Fidimag 21.87 nm;
  mumax3 22.10 nm. Reference m_z(r) curve from the bundled
  dataset.
- **Material parameters**: A = 13 pJ/m, D = 3 mJ/m², M_s =
  0.86 MA/m, K_u = 0.4 MJ/m³ (used as K_eff — `NoDemagSpins=1`
  in the reference, no demag in the ODE either). Disc R = 50 nm.
- **Setup**: lattice 50×50 at a = 2 nm, free-BC disc mask,
  no demag, mumax3 / Bogdanov–Roesler interfacial-DMI boundary
  condition via `fields._neighbors_with_bc`
  (`xi_inv_a = C_dmi/C_ex`), over-damped relaxation through the
  production `effective_field` (mask kwarg) + RK4.
- **Result**: r_sk = 22.61 nm (**2.62%** vs theory, tol 5%);
  outward-Néel chirality confirmed (m_x = +0.993 at r_sk);
  profile RMS 0.11 (tol 15%). **PASS.**
- **Deviations / assumptions**: profile-RMS tolerance loosened
  3%→15% — a 5%-level r_sk shift on a Δ≈5.7 nm wall yields
  RMS ≈ 0.1 even with perfect shape, so RMS is wall-position
  dominated (the published OOMMF/mumax3 runs show the same).
  Outside-mask cells carry a dummy +z spin (field zeroed by the
  mask) so the norm step does not raise. DMI-BC sign follows
  mumax3 (opposite to RT 2013 Eq.6); identical bulk chirality.

### 2. Bogdanov–Hubert / RT 2013 isolated-skyrmion profile
- **Reference**: Rohart & Thiaville, *PRB* 88, 184422 (2013),
  Eq.18, R_s = Δ/√(2(1−D/D_c)) (infinite film, no demag — "K is
  K_eff with shape folded in"). R_s,ref = 9.463 nm.
- **Material parameters**: A = 16 pJ/m, K_eff = 0.51 MJ/m³,
  μ₀M_s = 1.1 T, D = 3 mJ/m² (D/D_c = 0.825), Δ = 5.60 nm.
- **Setup**: 96×96 at a = 1 nm (box ≈ 17 Δ), PBC, no demag,
  over-damped relaxation; m_z(r) fit to the BH ansatz
  m_z = cos(2·arctan(exp(−(r−R)/Δ))).
- **Result**: R_s = 9.322 nm (**1.50%** vs Eq.18, tol 10%);
  shape RMS 0.012; Δ_fit = 4.53 nm (curvature-corrected vs the
  thin-wall asymptote). **PASS.**
- **Deviations**: only R_s is gated; Δ_fit differs from the
  thin-wall √(A/K) because Eq.18 itself is an approximation.
  Profile RMS reported, not gated.

### 3. NIST muMAG Standard Problem #4 (Permalloy, Field 1)
- **Reference**: NIST muMAG SP#4 (Eicke & McMichael). Ten
  contributed solutions agree to ~3%; ⟨m_x⟩=0 first crossing
  ≈ 136 ps. Bundled NIST result PDFs in `refs/`.
- **Material parameters**: Permalloy A = 13 pJ/m, M_s =
  0.8 MA/m, K = 0, α = 0.02, γ = 1.76e11. Magnet
  500×125×3 nm, cell a = 5 nm, Field 1 (μ₀H = −24.6, +4.3,
  0 mT).
- **Setup**: **single FM layer**, Newell free-BC (zero-padded)
  demag, [1,1,1] saturate-and-reduce S-state initial condition,
  driven through the production `rk4_step_single` +
  `effective_field`.
- **Result**: ⟨m_x⟩=0 first crossing at **136.66 ps**
  (**0.49%**, tol 10%); ⟨m_y⟩ peak ≈ 0.75 at ~130 ps matches
  the NIST curve shape. **PASS.**
- **Deviations**: cell a = 5 nm vs NIST's typical 2.5 nm (the
  10% gate absorbs the discretisation). K = 0 set via the bare
  prefactor. Earlier wrong result (~290 ps) came from a dummy
  SAF bottom layer adding spurious inter-layer demag — fixed by
  evolving a genuine single layer (`rk4_step_single`).

### 4. NIST muMAG Standard Problem #5 (Zhang–Li STT vortex)
- **Reference**: Najafi et al. 2009 (proposal for SP#5).
  Steady-state vortex-core displacement Δx = −1.2 nm,
  Δy = −14.7 nm for ξ = 0.05.
- **Material parameters**: Permalloy, 100×100×10 nm,
  Zhang–Li STT (Thiaville explicit form), ξ = 0.05,
  u_T = −72.17 m/s, 8 ns.
- **Setup**: single FM layer; STT via the production
  `integrator.zhang_li_torque`, time-stepped by
  `rk4_step_single`; vortex core located by peak-m_z +
  parabolic sub-cell fit.
- **Result**: Δx = −1.21 nm (≈0.01 nm off), Δy = −12.18 nm
  (≈2.5 nm off); total distance **2.52 nm** (tol 3 nm).
  **PASS.**
- **Deviations**: Δy is ~17% short of −14.7 nm — attributed to
  the 2.5 nm grid resolution and the 8 ns window not being
  fully settled. Within the 3 nm distance gate.

### 11. 1D domain-wall width
- **Reference**: textbook 1D wall, Δ = √(A/K_eff) (universal).
- **Material parameters**: A = 16 pJ/m, K_eff = 0.1 MJ/m³,
  M_s = 1.43 MA/m ⇒ Δ_analytic = 12.6491 nm.
- **Setup**: quasi-1D 401×8 lattice at a = 1 nm with pinned
  ends, relaxed via the **production** `effective_field`
  (exchange + anisotropy, D = 0, no demag) + `rk4_step_single`;
  m_z(x) fit to −tanh((x−x₀)/Δ).
- **Result**: Δ = 12.6371 nm (**0.095%**, tol 0.5%). **PASS.**
- **Deviations**: none material; parameters arbitrary (the
  relation is universal). This test exists primarily to prove
  the production field+integrator reproduce the
  exchange–anisotropy balance.

### 12. Uniform-mode FMR frequency
- **Reference**: macrospin FMR, f = (γ/2π)·H_K, H_K = 2K_eff/M_s
  (no Zeeman, no DMI, no demag).
- **Material parameters**: A = 16 pJ/m, K_eff = 0.5 MJ/m³,
  M_s = 0.8 MA/m, γ = 1.76e11 ⇒ f ≈ 35 GHz.
- **Setup**: uniformly magnetised single FM layer with a small
  in-plane tilt, integrated with `rk4_step` + `rhs_local_keff`,
  FFT of the precession, peak fit (DC bin excluded).
- **Result**: FFT peak within **< 5%** of the analytic value.
  **PASS.**
- **Deviations**: this validates only the **k = 0 uniform-mode
  FMR frequency**, not a full spin-wave dispersion ω(k). The
  task title says "FMR / spin-wave dispersion"; the implemented
  gate is the uniform-mode frequency, which is the cleanest
  analytic anchor.

### 10. Thiele v_SOT(R/Δ) and v_TSH vs Pham 2024 Fig.S49
- **Reference**: Pham et al., *"Fast current-induced skyrmion
  motion in synthetic antiferromagnets"* (2024), **Fig. S49**
  and supplementary §1.7. Two closed forms compared as a
  function of R/Δ at fixed Δ = 24.5 nm: the SOT-driven speed
  v_SOT = π H_DL R γ / (2α(R/Δ+Δ/R)) and the topological-spin-
  Hall speed v_TSH (λ² = 3, 50 nm²). Production driver:
  `studies/saf_racetrack/scripts/sweep_S49_TSH.py`.
- **Material parameters (Set B)**: α = 0.216, γ = 175.9 GHz/T,
  D = 0.62 mJ/m², Δ = 24.5 nm, R/Δ = 1…5 (imposed), λ² ∈
  {3, 50} nm². Analytic curves at the paper's J0 = 8×10¹¹.
- **Setup**: (1) **analytic gate** — evaluate the production
  `sot_thiele_speed` and `tsh_thiele_speed`
  (`src/skyrmion_simulator/simulator/topological_torque.py`) over R/Δ = 1…5 and
  check them against the independently-derived closed form,
  monotonicity/saturation, and λ² linearity. (2) **LLGS
  spot-check** — relax one SAF skyrmion, drive with DC SOT,
  compare the measured centroid speed to
  `sot_thiele_speed(R_fit, Δ_fit)`.
- **Result**: analytic gate matches the closed form to
  **2×10⁻¹⁶**, v_SOT monotone/saturating, v_TSH exactly linear
  in λ². LLGS spot-check (at J0 = **1×10¹¹**): v_meas vs Thiele
  **1.5%**; drive-induced deformation **5%** (rigid). **PASS.**
- **v_TSH < v_SOT is expected**: the topological-spin-Hall
  torque is a sub-dominant *correction* (∝ λ² and ∝ ∫N_xy²,
  which falls as the skyrmion grows), so it decays with R/Δ
  while v_SOT saturates. For the realistic λ² = 3 nm² it is
  ~3%→0.02% of v_SOT — S49's point being that the motion is
  SOT-dominated.
- **Deviations**: Fig. S49 itself is an analytic *calculation*
  (imposed R/Δ, fixed Δ — not relaxed, not a DMI sweep). The
  LLGS spot-check is an extra sim-vs-formula check we added; it
  runs at **1×10¹¹** (the DC-square-pulse current of Fig. S47,
  where the paper shows the velocity plateaus after ~200 ps),
  **not** the 8–8.9×10¹¹ of the deformation study — at that
  high current the paper itself documents skyrmion expansion/
  elliptical deformation that breaks the rigid-Thiele model.
  Rigidity is gated on the **J=0-referenced drive deformation**
  (drive radius minus a J=0 control radius from the same
  relaxed state), which cancels the relaxation creep; the raw
  full-window R_var (30%) is J0-independent creep, not drive
  deformation (true deformation 5%). The relaxed skyrmion has
  Δ_fit = 29.1 nm > the 24.5 nm seed, so its v_thiele sits
  slightly above the fixed-Δ=24.5 reference curve.

## B. Phase-diagram pipeline

### 5. Güngördü 2016 phase-boundary overlay
- **Reference**: Güngördü et al. 2016, Fig. 3, phase diagram in
  the dimensionless plane (a_s = A_s J/D², h_g = H J/D²) with
  FM / SkX / SP (spiral) / SC regions.
- **Material parameters / mapping**: A_ex↔J, anisotropy↔A_s
  with **K = −A_s,phys** (a_s = −K A_ex/D²;
  h_g = M_s H_z A_ex/D²), demag OFF. Sweep: 64×64 at a = 5 nm,
  D = 1.5 mJ/m², 50×50 grid, four candidate ICs
  (FM / stripe / sk-lattice / sq-lattice). Cluster array job.
- **Result**: **qualitative PASS** — the FM region and a
  modulated easy-plane region are reproduced, but that region
  is ~32% spiral and only **~2.6% SkX**, whereas Güngördü has a
  large SkX lens. **PASS (qualitative)**.
- **Deviations / limitation**: a hexagonal SkX tiles a √3
  rectangular cell, so a fixed **square** grid can never be
  hex-commensurate; the flexible 1D spiral relaxes lower and
  wins where SkX should. Güngördü optimise the unit-cell size
  variationally per phase; a fixed-grid sweep structurally
  cannot. Two real bugs fixed en route: (i) anisotropy sign
  (both field and energy use bare K with demag off, so
  K = −A_s,phys), (ii) ground state taken as **min-energy** over
  ICs (strict torque convergence excluded the slowly-relaxing
  textures and collapsed everything to FM).

### 6. Banerjee 2014 FM↔SkX critical field
- **Reference**: Banerjee et al., *PRX* 4, 031045 (2014),
  Fig. 1(b), FM↔SkX critical field h_c(a_s) in the same plane.
- **Setup**: post-processing on the #5 Güngördü sweep (no new
  simulation); extracts the field at which the ground state
  flips SkX→FM at a few a_s slices.
- **Result**: h_c,sim ≈ 0.18–0.96 vs Banerjee 1.03–1.52
  (**37–82% low**), tol 20%. **Quantitative FAIL, documented.**
- **Deviations / limitation**: same root cause as #5 — the
  fixed square grid under-stabilises SkX, so it is destroyed by
  a lower field than the variational ansatz predicts. The
  topology (FM/SP) is right; the SkX stability window is not.

### 7. RT 2013 Fig.4(d) confined-skyrmion radius
- **Reference**: Rohart & Thiaville 2013, Fig. 4(d): confined
  skyrmion radius R_s in a finite dot vs D/D_c. Read value
  R_s ≈ 25 nm at R_dot = 50 nm, D/D_c = 1.25.
- **Material parameters**: A = 16 pJ/m, K_eff = 0.50 MJ/m³,
  M_s = 1.1 MA/m, D = 4.5 mJ/m² (D_c = 3.61 mJ/m², D/D_c =
  1.25), dot radius R_dot = 50 nm, a = 2 nm.
- **Result**: R_s = **27.6 nm** (≈10%, tol 25%). **PASS.**
- **Deviations**: Eq.18 (infinite-film) diverges for D ≥ D_c,
  so above threshold the dot radius alone sets R_s; the test
  guards Eq.18 and compares to the Fig.4(d) read value with a
  loose figure-read tolerance.

### 9. Skyrmion-size temperature scaling (Tomasello 2018)
- **Reference**: R. Tomasello et al., *PRB* 97, 060402(R)
  (2018), Fig. 1(b): skyrmion diameter vs temperature in a
  confined dot at H = 0/25/50 mT. Temperature enters only
  through the reduced magnetization m(T) = M_s(T)/M_s(0), which
  scales the parameters by Callen-Callen exponents: M_s ∝ m,
  A ∝ m^1.50, D ∝ m^1.50, K_u ∝ m^3.585. As T rises the reduced
  DMI d ∝ m^−0.84 climbs toward the critical line
  d_c(Q) = (4/π)√(Q−1) and the skyrmion expands (strongly at
  H = 0, weakly at finite field).
- **Material (T=0)**: from the paper's Q(0)=2.65, d(0)=1.41,
  l_ex(0)=9.4 nm → M_s=0.60 MA/m, A=20 pJ/m, K_u=0.60 MJ/m³,
  D=3.0 mJ/m². Dot R_d=141 nm (=15 l_ex, read from the Fig. 3
  insets), thickness 0.8 nm, cell 2.5 nm. m(T) digitized from
  Fig. 2 (D(T)/D(0)=m^1.5 ⇒ m: 1.000→0.928 over 0–300 K); the
  R(T) gate target digitized from Fig. 1(b). Digitization
  figures: `docs/figures/validation/tomasello_*`.
- **Setup**: confined disk via the **production** `relax`
  (single-layer mode + free-BC `mask`, no demag — K_eff =
  K_u − ½μ₀M_s² folded by the simulator's anisotropy
  convention). Per (H, T): scale parameters by m(T), relax the
  seeded skyrmion, measure R_sk = ½·`skyrmion_diameter`.
- **9a (deterministic) — PASS**: R_sk(T) reproduces the
  field-ordered expansion of Fig. 1(b); H=25/50 mT track the
  analytical curves closely. H=0 expansion ratio
  R_sk(300)/R_sk(0) = **2.51** (converged: the T=300/H=0 point
  reaches conv=True at 160 k steps, R_sk=33.4 nm) vs the paper's
  3.29 (23%, within the 30% gate). The deterministic relaxation
  stays in the small metastable minimum (at 300 K our scaling
  gives d=1.44 < d_c=1.48, i.e. still below the metastable→
  ground crossing), so it slightly under-expands relative to
  the paper's analytical ansatz.
  Figure: `docs/figures/validation/tomasello_size_vs_T.png`.
- **9b (thermal) — FAIL on the quantitative ratio, qualitative
  behaviour correct**: the finite-T branch
  (`scan_skyrmion_size_thermal` + `test_skyrmion_size_thermal`,
  168-task SLURM array) scales the parameters by m(T) **and**
  adds FDT Brown noise via the production single-layer
  `heun_stochastic_step` (with the free-BC `mask`), pooling
  ⟨R_sk⟩±std per (H,T). It reproduces the qualitative Fig. 1(b)
  thermal physics (size grows with T, std grows, field
  suppresses it), but the H=0 ratio over-shoots: 4.76 (R_sk=71
  nm at 300 K) vs the converged deterministic 33.4 nm and the
  paper's ~46–52 nm. Root cause is **non-equilibration in the
  strong-fluctuation regime**, not a measurement artifact: a
  diagnostic showed the LCC and whole-field diameters agree
  (63.3 vs 63.5 nm) but R_sk **drifts** 45→76 nm during the
  sampling window. At T=300/H=0 the scaled point sits just
  below the metastable→ground crossing (d=1.44 < d_c=1.48), so
  the barrier out of the small metastable state is tiny; thermal
  noise drives the skyrmion to escape and expand toward the dot
  without reaching a steady size, making ⟨R_sk⟩ window-
  dependent (71 nm at 8 k-step thermalize vs 63 nm at 4 k).
  This is the paper's own "strongly thermal-fluctuation-
  influenced" regime (T>200 K, H<10 mT, ±25 nm error bars,
  horseshoe deformation); longer thermalization drifts larger,
  not toward the paper value. **#9a (deterministic, matching
  the analytical curves) is therefore the quantitative
  benchmark; #9b is a qualitative reproduction of the
  fluctuation-driven expansion** — analogous to the #5/#6
  documented limitation.

## C. Stochastic-LLGS module

### 8. Skyrmion Néel-Arrhenius collapse at j = 0 (single FM layer)
- **Reference**: Rohart, Miltat & Thiaville, *PRB* 93, 214412
  (2016), *"Path to collapse for an isolated Néel skyrmion."*
  Langevin protocol (Fig. 2c): τ(T) = τ₀ exp(ΔE/k_BT),
  ΔE = 26±4 meV, τ₀ = 0.22±0.1 ns under a 250 mT destabilizing
  field at 68–101 K.
- **Material parameters**: RT-compatible — A = 16 pJ/m,
  K_eff = 0.50 MJ/m³ (bare), M_s = 1.1 MA/m, D = 3.0 mJ/m²
  (D/D_c = 0.83), α = 0.3, a = 2 nm, t_Co = 0.6 nm, 128×128.
  Destabilizing field **95 mT** (sub-critical), T = 35–80 K
  (10 points), 60 ns window, n_ens = 128 (1279 cluster
  trajectories).
- **Setup**: a genuine **single FM layer** evolved by the
  **production** `heun_stochastic_step` generalised to a
  single-layer mode (`m_bot = None`; the SAF pair path is
  unchanged), with params from the production `make_params`
  (K_top set so the thin-film fold reproduces the bare K_eff,
  H_RKKY = 0). Collapse = first-passage of |Q| below 0.5;
  τ(T) by censored maximum-likelihood; fit ln τ vs 1/T.
- **Result**: R² = **0.983** over a decade of τ (16.4→1.8 ns);
  τ₀ = **0.35 ns** (Rohart 0.22 — same order); ΔE = **11.5 meV**.
  **PASS** (gate = Arrhenius form + sub-ns τ₀ + barrier sign).
- **Deviations**: the field is **95 mT, not Rohart's 250 mT** —
  Rohart's model is *atomistic* (Co/Pt(111), 4.6 nm skyrmion);
  ours is micromagnetic continuum, where 250 mT is supercritical
  (H_c ≈ 135 mT → athermal instant collapse). 95 mT is the
  sub-critical field where a clean activated ladder exists
  (τ ~ 1–7 ns over 35–80 K; saturates at the ~0.9 ns dynamical
  floor above ~85 K, so T ≤ 80 K). **ΔE is reported, NOT
  gated**: the micromagnetic collapse barrier is grid-dependent
  and the model class differs from Rohart's atomistic, so
  matching the *order* (tens of meV) is the honest expectation,
  not the number (11.5 vs 26 meV, ~2×). The method itself is
  anchored independently by the macrospin Néel-Brown gate
  (§C-Brown).

### 8b. ΔE-vs-DMI trend (Rohart 2016 Fig.5b)
- **Reference**: Rohart 2016, Fig. 5(b): the collapse barrier
  rises strongly with DMI at fixed field.
- **Setup**: the #8 scan repeated over a narrow DMI grid
  D = 2.90–3.10 mJ/m² (5 values × 10 T × 64 ens) at the same
  95 mT; ΔE(D) fit per D; gate = monotone increase of ΔE with D
  (absolute ΔE not gated). The narrow grid is required because
  the barrier is exponentially D-sensitive at fixed field — the
  τ-observable window (collapse within the 60 ns window, not
  athermal) spans only D ≈ 2.95–3.10. The analysis was
  validated on synthetic Arrhenius data (recovers an imposed
  rising ΔE).
- **Result: PASS.** ΔE rises monotonically and steeply —
  5.7 → 12.9 → 21.5 → 31.5 meV over D = 2.95 → 3.00 → 3.05 →
  3.10 (5.5× over a 5% DMI increase), reproducing Rohart
  Fig. 5(b)'s strong barrier-vs-DMI nonlinearity; τ₀ stays
  0.25–0.38 ns. D ≤ 2.90 is athermal (ΔE≈0) and D ≥ 3.20 never
  collapses in-window — both correctly excluded by the
  events/T gate. Absolute ΔE is reported, not gated
  (grid-dependent, model-class mismatch as in #8).

### Supporting stochastic-infrastructure gates (underpin #8)
These validate the thermal machinery (FDT noise amplitude +
Stratonovich-Heun integrator) that #8 relies on:
- **Brown reversal** (`test_brown_reversal`): uniaxial macrospin
  Néel-Brown time τ = (1+α²)/(αγ)·√(π/Δ)·e^Δ, Δ = K V/k_BT,
  barriers Δ ∈ {3, 5}; gate |log₁₀(τ_sim/τ_Brown)| < 0.5. This
  is the closed-form anchor for the stochastic method.
- **Langevin** (`test_langevin`): macrospin in a Zeeman field,
  ⟨m_z⟩ vs the Langevin function L(μB/k_BT).
- **Equipartition** (`test_equipartition`): spin-wave mode
  energy ≈ k_BT per mode on a small FM patch.
- **T=0 limits** (`test_t0_limit_nodemag/_demag`): the Heun
  stepper at T = 0 reproduces the deterministic RK4 trajectory
  (with and without demag) — confirms the noise term is the
  only stochastic addition.

## D. Free-BC infrastructure (task #14)
- `lattice.disk_mask` / `rect_mask`; an optional `mask` kwarg on
  `exchange_field` / `dmi_field` / `anisotropy_field` /
  `effective_field` / `effective_field_demag_pair` (default
  `None` = original PBC path, byte-equivalent);
  `_neighbors_with_bc` (mumax3 / Bogdanov–Roesler DMI BC);
  zero-padded Newell kernel (`kind='newell_freebc'`).
- **Validation**: uniformly-magnetised slab interior
  H_z/μ₀M_s = **−0.984** (std 0.004) vs the thin-slab limit −1.0;
  centre field −0.989 matches the PBC kernel exactly; corner
  −0.667 (expected edge effect). All pre-existing PBC tests
  remain byte-equivalent. **PASS.**

---

## Cross-cutting limitations and conventions
- **Anisotropy convention**: with demag OFF, both the field
  (`effective_field_demag_pair`) and the energy (`total_energy`)
  use the **bare** prefactor 2K/M_s; tests that want a specific
  K_eff set `K_top = K_eff + ½μ₀M_s²` so the thin-film fold
  lands the effective anisotropy at 2K_eff/M_s.
- **Single-FM vs SAF**: #3, #4, #11, #12 and #8 exercise a
  single ferromagnetic layer. #8 reuses the production SAF Heun
  stepper in a generalised single-layer mode; #3/#4/#11 use
  `rk4_step_single`. #10 is genuinely SAF (compensated → no
  skyrmion Hall, which is why `sot_thiele_speed` has no
  gyrovector term).
- **Model-class honesty**: where a continuum micromagnetic run
  is compared to an atomistic reference (#8 vs Rohart 2016) or
  to a variational-cell ansatz (#5/#6 vs Güngördü/Banerjee), the
  *form/trend* is gated and grid-dependent absolute numbers are
  reported but not gated, with the mismatch stated explicitly.
- **Loosened tolerances**, each justified in-place: #1 profile
  RMS 3%→15% (wall-position-dominated), #3 10% (a = 5 nm vs
  2.5 nm), #7 25% (figure-read), #8 ΔE not gated (model class).

---

## How to reproduce
```bash
# simulator core
python -m skyrmion_simulator.simulator.validation.test_dmi_standard_problem
python -m skyrmion_simulator.simulator.validation.test_skyrmion_profile_bh
python -m skyrmion_simulator.simulator.validation.test_mumag_sp4
python -m skyrmion_simulator.simulator.validation.test_mumag_sp5
python -m skyrmion_simulator.simulator.validation.test_dw_profile_1d
python -m skyrmion_simulator.simulator.validation.test_fmr_dispersion
python -m skyrmion_simulator.simulator.validation.test_thiele_v_sot
# phase diagram (sweeps are cluster array jobs)
python -m skyrmion_simulator.phase_diagram.validation.test_confined_skyrmion_radius_rt2013
python -m skyrmion_simulator.phase_diagram.validation.plot_gungordu_overlay
python -m skyrmion_simulator.phase_diagram.validation.banerjee_critical_field
# stochastic-LLGS
python -m skyrmion_simulator.stochastic_llgs.validation.test_brown_reversal
python -m skyrmion_simulator.stochastic_llgs.validation.test_langevin
python -m skyrmion_simulator.stochastic_llgs.validation.test_equipartition
python -m skyrmion_simulator.stochastic_llgs.validation.test_skyrmion_arrhenius        # reads cluster aggregate
python -m skyrmion_simulator.stochastic_llgs.validation.test_skyrmion_arrhenius_dtrend # #8b
```
Initial-condition and result figures are written to
`docs/figures/validation/` when each test runs.

---

*Last updated: see git log for `docs/validation_results.md`.*
