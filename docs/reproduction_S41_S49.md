# Reproduction plan: Pham et al. (2024) supplementary figures S41–S49

Source: `refs/Fast current-induced skyrmion motion in synthetic
antiferromagnets.pdf`, supplementary sections 1.4–1.8 (pages ~30–40).

This document lists every figure to reproduce, the parameter set
each uses, the sweep variable(s), the observables computed, and
what the simulator (`src/simulator/`) still needs to support.

---

## Two parameter sets used by the paper's simulations

### Set A — "Table S2 + adjusted DMI" (used for S41–S47)

Matches `src/simulator/parameters.default_params()` *except* `D`:

| Symbol         | Value                  | Notes                          |
|----------------|------------------------|--------------------------------|
| Ms             | 1.43e6 A/m             | unchanged                      |
| A_ex           | 16e-12 J/m             | unchanged                      |
| **D**          | **0.85 mJ/m² (=8.5e-4 J/m²)** | **calibrated; default code has 0.62e-3** |
| K_top          | 1.294e6 J/m³           | unchanged                      |
| K_bot          | 1.31e6 J/m³            | unchanged                      |
| α              | 0.14                   | unchanged                      |
| t_Co           | 1.3e-9 m               | unchanged                      |
| d_Ru           | 0.8e-9 m               | unchanged                      |
| γ              | 194.8e9 rad/(s·T)      | unchanged                      |
| H_RKKY         | 0.205 T                | unchanged (varied in S46, S47, S48) |
| χ_DL           | 2.21e-14 T·A⁻¹·m²      | unchanged                      |
| χ_FL           | 0.53e-14 T·A⁻¹·m²      | unchanged                      |

Notes: the paper uses `D = 0.76 mJ/m²` instead of `0.62 mJ/m²`
"such that the skyrmion diameter is similar to the experimental
~215 nm" (supplementary §1.1). With our Newell finite-prism
kernel, the effective DMI threshold is higher, so we use
`D = 0.85 mJ/m²` to reproduce the same ~215 nm size. Both
adjustments lie within the ±39% error bar on D.

### Set B — "S40 / S48 specific" (used for S40 and S48 only)

| Symbol  | Value                | Difference vs Set A             |
|---------|----------------------|---------------------------------|
| α       | 0.216                | larger than 0.14                |
| γ       | 175.9e9 rad/(s·T)    | smaller than 194.8e9            |
| D       | 0.62 mJ/m²           | paper Table S2 nominal          |
| Hk_top  | 12.4 mT (→ K_top via H_K = 2K_eff/Ms) | unchanged from default |
| Hk_bot  | 35.4 mT              | unchanged from default          |

These differ from Set A. Only S48 is in scope here (S40 is static
charge-density maps; we already have those from
`plot_charge_density.py`).

---

## Figure-by-figure plan

### Figure S41 — Instantaneous velocity vs time (long DC pulse)

- **Purpose:** show the rise to steady state; verify
  `v_avg = ΔX/t_pulse` matches steady-state velocity.
- **Parameter set:** Set A.
- **Drive:** **DC square pulse**, J = 1×10¹¹ A/m², duration 2 ns.
- **Sweep:** none (single trajectory).
- **Observables:**
  - `v_inst(t)` = finite-difference velocity of skyrmion centre,
    sampled every ~5–10 ps.
  - `v_avg = ΔX / t_pulse` (single scalar; dashed black line in
    the figure).
- **Simulator status:** **supported** (the existing DC-J flow with
  longer `n_steps` and `dump_every` smaller suffices). No code
  changes needed.

### Figure S42 — Gaussian pulses of varying current density

- **Purpose:** show that `v_max` diverges from `v_avg` at high J
  (due to skyrmion deformation).
- **Parameter set:** Set A.
- **Drive:** **Gaussian pulse** `J(t) = J_0 exp(-(t-t_0)²/2σ²)`
  with FWHM = 500 ps.
- **Sweep:** `J_0 ∈ {1, 2, 3, 4, 5, 6, 7, 8, 8.9} ×10¹¹ A/m²`
- **Observables (per J_0):**
  - `v_inst(t)` (panel a, traces vs time).
  - `v_max` = max of `v_inst` (panel b, blue).
  - `v_avg = ΔX / FWHM` (panel b, red).
- **Simulator status:** **needs Gaussian-pulse drive**. Currently
  J_current is a constant. Add a callable `J(t)` or expose
  `H_DL(t), H_FL(t)` arrays in `integrator.rk4_step`.

### Figure S43 — Pulse-width and skyrmion-size sweeps

- **Panel (a) — `v_avg` vs FWHM**
  - Set A.
  - Gaussian pulse, FWHM ∈ {100, 150, 200, 250, 300, 350, 400, 450, 500, 530}
    ps.
  - Two J values: J = 4×10¹¹ A/m² and J = 8.9×10¹¹ A/m².
  - Total: 10 × 2 = **20 runs**.
- **Panel (b) — `v_avg` vs J for two skyrmion sizes**
  - Set A but with **two D values** producing two skyrmion radii.
  - D₁ = 0.85 mJ/m² (large skyrmion, ~215 nm).
  - D₂ ≈ 0.60 mJ/m² (smaller skyrmion).
  - Gaussian pulse FWHM = 500 ps.
  - J sweep: J ∈ {1, 2, 3, 4, 5, 6, 7, 8, 8.9} ×10¹¹ A/m².
  - Total: 9 × 2 = **18 runs**.
- **Observables:** `v_avg = ΔX / FWHM`.
- **Simulator status:** Gaussian pulse needed (same as S42).

### Figure S44 — Skyrmion deformation under 0.5 ns pulse

- **Purpose:** show elliptical deformation (D1, D2) and DW
  magnetization rotation under strong drive.
- **Parameter set:** Set A.
- **Drive:** Gaussian pulse, FWHM = 500 ps.
- **Panel A — diameter vs time**
  - Single J = 8.9×10¹¹ A/m².
  - `D_avg(t)` (current `skyrmion_diameter` is enough).
- **Panel B — velocity, D1, D2 vs J**
  - Sweep J ∈ {1, 2, 3, 4, 5, 6, 7, 8, 8.9} ×10¹¹ A/m² (~9 points).
  - `v_avg`, `D1_max` (major axis), `D2_max` (minor axis) at pulse
    peak.
- **Panel C — DW angle ψ_top, ψ_bot vs J**
  - Same sweep as Panel B.
  - DW magnetization angle ψ relative to −x̂ axis.
- **Simulator status:**
  - Gaussian pulse: needed.
  - **Ellipse fit:** add `skyrmion_ellipse(m, a)` returning
    (D1, D2) from second moments / PCA of the `mz<0` region.
  - **DW angle:** add `dw_angle(m, a)` returning ψ — typically
    the average direction of `(m_x, m_y)` along the
    `mz=0` contour.

### Figure S45 — DW vs skyrmion comparison

- **Purpose:** verify that at high J the skyrmion behaves like a
  single DW.
- **Parameter set:** Set A.
- **Drive:** Gaussian pulse, FWHM = 500 ps.
- **Sweep:** J ∈ {1, 2, 3, 4, 5, 6, 7, 8, 8.9} ×10¹¹ A/m² (~9
  points) **for two initial conditions:**
  - (i) skyrmion (same as S44).
  - (ii) single 1D Néel DW in a long track (new IC).
- **Observables:** v_avg vs J, ψ vs J for top/bottom layers.
- **Total:** 9 × 2 = **18 runs**.
- **Simulator status:**
  - **New initial condition:** `domain_wall(nx, ny, a, w)` in
    `src/simulator/initial_conditions.py` — a stripe domain
    crossing the track, with one Néel DW in the middle.
  - **Track geometry:** use the existing PBC lattice but tile so
    the DW is far from itself across the PBC.
  - Gaussian pulse: needed.
  - `dw_angle`: needed (same as S44).

### Figure S46 — RKKY and pulse-width sweep at fixed J

- **Parameter set:** Set A.
- **Drive:** Gaussian pulse, J = 8.9×10¹¹ A/m².
- **Panel A — v, D vs H_RKKY (fixed pulse width 0.5 ns)**
  - Sweep H_RKKY ∈ {200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 
    700, 750, 800, 850, 900, 950} mT (16 points).
- **Panel B — v, D vs pulse width (fixed H_RKKY = 205 mT)**
  - Sweep FWHM ∈ {100, 150, 200, 250, 300, 350, 400, 450, 500} ps (8 points).
- **Observables:** v_avg, D_avg at pulse peak.
- **Total:** 16 + 8 = **24 runs**.
- **Simulator status:** Gaussian pulse + H_RKKY override
  (already a parameter).

### Figure S47 — RKKY and pulse-width sweep over J

- **Parameter set:** Set A.
- **Drive:** Gaussian pulse, **three configurations:**
  - (i) FWHM = 500 ps, H_RKKY = 205 mT.
  - (ii) FWHM = 500 ps, H_RKKY = 410 mT.
  - (iii) FWHM = 100 ps, H_RKKY = 205 mT.
- **Sweep:** J ∈ {0.5, 1, 1.5, 2, 3, 4, 5, 6, 8, 8.9} ×10¹¹ A/m² (10 points per configuration).
- **Observables per (config, J):**
  - v_avg, ψ_bot (DW angle, bottom layer), D1, D2 at pulse max.
  - Snapshots of m_top at pulse max for (i), (ii), (iii) at the
    highest J (panels E, F, G).
- **Total:** 10 × 3 = **30 runs** + 3 snapshots.
- **Simulator status:** Gaussian pulse + ellipse fit + DW angle.

### Figure S48 — RKKY effect on inertial dynamics

- **Purpose:** rise/fall time of v(t) vs H_RKKY → Thiele inertia.
- **Parameter set:** **Set B** (α=0.216, γ=175.9 GHz/T,
  D=0.62 mJ/m²).
- **Drive:** **DC square pulse**, J = 1×10¹¹ A/m², duration 2 ns.
- **Sweep:** H_RKKY ∈ {100, 150, 200, 250, 300, 350, 
    400, 450, 500} mT (9 points).
- **Observables:**
  - Panels (a), (b): v(t) rise and fall transients (zoomed in).
  - Panel (c): inverse time constant 1/τ extracted by
    exponential fit, plotted vs H_RKKY (one mean + error bar per
    H_RKKY, averaged over 4 fits: top/bottom layers × rise/fall).
- **Total:** **9 runs** (DC pulse, two layers tracked, four fits
  each).
- **Simulator status:**
  - DC pulse with **rise/fall captured** (need finer `dump_every`).
  - **Exponential fit** of `v(t)` (post-processing, not
    simulator core).

### Figure S49 — Topological spin Hall torque

- **Purpose:** show that the topological spin Hall (TSH) torque
  is negligible for large skyrmions, significant when R/Δ ≈ 1.
- **Parameter set:** Set B.
- **Method:** **not a dynamics simulation** — direct numerical
  evaluation of the Thiele force integral
  $F_{tst,x} = -(M_s t/\gamma)\,b_j \lambda^2 \int\!\int
  ((\partial_x \mb{m}\times\partial_y\mb{m})\cdot\mb{m})^2
  \,dx\,dy$
  and the SOT Thiele force on an analytic Néel-skyrmion profile
  $\theta(r) = 2\,\text{atan}(\exp(-(r-R)/\Delta))$.
- **Sweep:**
  - $R/\Delta \in \{1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5\}$ (9 points).
  - Two λ² values: 3 nm² and 50 nm².
  - Δ fixed = 24.5 nm.
  - J = 8×10¹¹ A/m², P = 0.5.
- **Total:** **9 × 2 = 18 evaluations**, all analytic — no time
  integration.
- **Observables:** v_SOT, v_TSH(λ²=3), v_TSH(λ²=50) vs R/Δ.
- **Simulator status:**
  - **New module needed:** `src/analysis/topological_torque.py`
    that takes an analytic profile and integrates the TSH force.
  - **No LLGS time-stepping required** — independent of
    `integrator.py`. Pure post-processing math.


### Simulator extensions required (rough priority order)

1. **Gaussian pulse drive** in `integrator.py` (or a thin wrapper):
   `J(t) = J_0 exp(-(t-t_0)²/(2σ²))`, with σ from FWHM and a
   safe leading dead-time. Required by S42, S43, S44, S45, S46,
   S47. Touches `rk4_step` (recompute H_DL, H_FL each substep).
2. **Ellipse fit** for skyrmion shape (`skyrmion_ellipse`) in
   `analysis.py` — principal axes of the `mz<0` region. Required
   by S44, S47.
3. **DW magnetization angle** (`dw_angle`) in `analysis.py` — from
   the in-plane `m_xy` averaged along the `mz=0` contour.
   Required by S44, S45, S47.
4. **Topological-torque module** (`src/analysis/topological_torque.py`)
   — pure analytic-profile integration. Required by S49.

### Post-processing scripts needed

1. `scripts/sweep_J.py` — drives the J sweeps (S42, S43b, S44,
   S45, S47).
2. `scripts/sweep_pulse_width.py` — drives the FWHM sweeps (S43a,
   S46b).
3. `scripts/sweep_HRKKY.py` — drives the H_RKKY sweeps (S46a,
   S47ii, S48).
4. `scripts/fit_inertia.py` — exponential fit of v(t) for S48c.
5. `scripts/plot_S41.py` … `scripts/plot_S49.py` — one plotting
   script per figure (or a single dispatcher).

### Output layout (proposed)

```
output/
  figures_S41_S49/
    S41_v_t.png
    S42_v_t.png, S42_vmax_J.png
    S43a_v_FWHM.png, S43b_v_J.png
    S44a_d_t.png, S44b_v_d_J.png, S44c_psi_J.png
    S46a_v_HRKKY.png, S46b_v_FWHM.png
    S47_A.png, S47_B.png, S47_C.png, S47_D.png
    S47_E.png, S47_F.png, S47_G.png  (snapshots)
    S48a_v_t_rise.png, S48b_v_t_fall.png, S48c_inv_tau.png
    S49_v_topo.png
  sweeps_S41_S49/
    <one .npz / .csv per (figure, sweep_point)>
```

---

## Open questions for the user before planning code changes

(Edit this section with answers, then we can lock the plan.)

1. **Are figures S41 and S48 in scope?** They use the parameter
   set B (α=0.216, γ=175.9 GHz/T), different from the rest. The
   simulator can run them by overriding parameters, but the
   plotting machinery may need a second config path.

   Answer: Yes, they are in scope. 

2. **Demag:** the paper uses full FFT demag (MuMax3). Our main
   loop uses the local $K_{\text{eff}}$ approximation. Do we
   reproduce S41–S49 with local-demag or with full FFT demag
   (`src/phase_diagram/fields_demag.py`)? Full demag is more
   faithful but 10× slower.

   Answer: Full demag. 

3. **Pulse leading/trailing dead-time:** the paper's Gaussian
   pulses have a Gaussian time profile that decays smoothly. How
   long a "tail" to integrate (e.g. ±3σ) before counting v_avg?

   Answer: 3 sigma. 

4. **DW track geometry for S45:** keep the 256×256 lattice with
   PBC and put a single Néel DW running parallel to ŷ at the
   centre, or extend the box (e.g. 512×128) so the DW has more
   room to move along x̂?

  Answer: Do not plot Fig S45, disregard it. 
  
5. **Topological spin Hall torque (S49):** purely analytic
   evaluation, or also implement the torque in the LLGS RHS to
   verify the analytic prediction numerically?

  Answer: implement the torque in the LLGS RHS to 
  verify the analytic prediction numerically.

6. **Compute budget:** ~96 dynamics runs at 30 s–5 min each
   depending on lattice size and demag choice. Run locally
   serially (~few hours) or use the HPC array template
   (`scripts/submit_sweep_array.sh`)?

   Answer: both should be possible.

7. **Snapshots (S47 E–G):** any preference for Ovito-style
   `write_dump` followed by an Ovito script, or direct
   `matplotlib` heatmap of m_z at the snapshot time (faster
   pipeline, no third-party tool)?

   Answer: Direct matplotlib heatmap for now.
