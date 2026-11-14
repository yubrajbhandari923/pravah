# E16 generalization and sensitivity ablations: design note (pre-pilot)

Status: written 2026-11-14 on the E16 agent branch, **before any E16 computation**, and committed alone. Later changes go
into §10 (Deviations) with their reason. Items marked **[LEAD REVIEW]** are choices the plan leaves open; the pilot
proceeds with the stated default. Synthetic 2-D depth-averaged passive tracer; no statement here concerns real PM2.5.
The only measurement made before this note was a solver timing (s per simulated hour at 32²: upwind dt 300 s 0.18;
VanLeer dt 75 s 1.14; dt 60 s 1.40), used to size the pilot. No E16 metric was computed.

Inputs: plan-03 key `e16-generalization`; E07 journal and gate-07 (source_near beats dopt under the 24 h-window MAP;
dopt beats random); gate-09 (with all past data dopt beats source_near, 103/120 at 64²); E08 (k*(0.2) ≈ K for
location-aware designs, 8–16 for location-blind ones); E09/gate-09 (memory gain of the growing-window MAP over map24,
+0.58 to +0.74 at 32², 0.85–0.94 at 64²).

## 1. Question

Which of the existing placement / memory conclusions survive a **one-at-a-time** change of the physical model or the
source configuration, and which are conditional on the benchmark's nominal settings? The plan hypothesis is that
rankings are conditional; this study is built to find reversals, not to confirm stability.

## 2. Conclusions under test (each is a paired contrast with a fixed sign at nominal)

Primary endpoint everywhere: **held-out 1 h forecast RMSE** on the test segment (held-out wind regimes), white noise,
mean over R = 5 paired noise draws, exactly the E07/E08 scorer. Contrast ρ(A vs B) = (RMSE_B − RMSE_A)/RMSE_B per
scenario, > 0 favours A. `random` = mean RMSE over its 10 networks.

| id | conclusion (source) | estimator | contrast at k = 8 | expected sign |
|---|---|---|---|---|
| C1 | source_near beats dopt (E07) | map24 | source_near vs dopt | + |
| C2 | dopt beats random (E07 confirmatory) | map24 | dopt vs random | + |
| C3 | ordering reverses with all past data (gate-09) | map_grow | dopt vs source_near | + |
| C4 | memory gain (E09) | map24 → map_grow | M = (RMSE_map24 − RMSE_grow)/RMSE_map24, for dopt and for source_near | + |
| C5 | k*(0.2) ≈ K for location-aware designs, larger for location-blind ones (E08) | map24 (= E08 REF: 24 h window, hourly) | k* per strategy, nRMSE = RMSE / background-only RMSE | k*_sn, k*_dopt ≤ K; k*_random > K |

Also reported (secondary, not part of any verdict): the same contrasts at k = 4 and 16; coverage vs random; per-regime
forecast RMSE (southerly block); analysis and source RMSE.

Strategies: random (10 networks), coverage, source_near, dopt (greedy + swap), exactly the E07/E08 implementations
(`cross_objective.make_design`). The robust/forecast designs of E11/E15 are **not** included (E15 is running separately);
adding them belongs to the 64² run once E15 is integrated **[LEAD REVIEW]**.

Estimators: map24 (`cross_objective.estimate_series`, window 24 h) and map_grow (all hours 0..t, prior N(0, I) in a_ref
units; `forecast_design.estimate(window=None)`, ≡ E09 kf_aug with a perfect model).

## 3. Sensitivity axes (one at a time) and the declared interaction

Nominal = the E07/E08 32² pilot world: 32² upwind dt 300 s, spin-up 12 h, calendar 4/1/6 days (train/val/test),
training winds `mixed` (steady westerly, veering, weak variable), held-out winds steady southerly, stagnation then front,
diurnal rotation; K_phys = 50 m² s⁻¹, removal λ = 1/(12 h), source footprint σ_s = 400 m, source min separation 1500 m,
background c_bg = 5 µg m⁻³, mixing depth 500 m, a_ref = 1 g s⁻¹.

| cell | axis | change (units) | what it stresses |
|---|---|---|---|
| nominal | – | – | reference |
| `diff_hi` | effective diffusivity ↑ | K_phys = 1000 m² s⁻¹ (upwind K_eff ≈ 1.5–2.5 × 10³ → roughly +50 %) | broader, smoother plumes |
| `diff_lo` | effective diffusivity ↓ | VanLeer split, dt 75 s, K_phys 50 (K_eff drops from ~10³ numerical to ~10²; max Courant 0.84) | compact plumes; also the 32² scheme of the 64² world |
| `loss_hi` | first-order loss ↑ | λ = 1/(2 h) | shorter plume reach |
| `src_wide` | source shape | σ_s = 800 m | wider emission discs |
| `src_sparse` | source spacing | min separation 3000 m (same strengths; layouts change) | sources farther apart |
| `bg_hi` | inflow background | c_bg = 20 µg m⁻³ | under the protocol's σ = 0.05 × median truth rule this raises σ ~3–4× (an SNR shift; the estimator knows c_bg) |
| `diff_lo×loss_hi` | declared interaction | VanLeer dt 75 s and λ = 1/(2 h) | most compact plumes: the stress case for location-blind designs and for k* ≈ K |

Dimensionless groups at nominal (L = 10 km, U ≈ 2–3.5 m s⁻¹): Pe = UL/K_eff, Da = λL/U, footprint ratio σ_s/L,
spacing/L. Each axis changes one group by a stated factor; the effective (numerical + physical) diffusivity is reported
from the A2 modified-equation estimate, not assumed.

Notes and limits:
- At 32² upwind the numerical diffusivity (≈ K + |u|dx/2 + |u|²dt/2 ≈ 10³ m² s⁻¹) dominates the physical K = 50. A
  physical-K change therefore moves K_eff only modestly; the large diffusivity contrast comes from `diff_lo` (the scheme).
  `diff_lo` thus confounds scheme and diffusivity by construction; it is labelled "effective diffusivity", not "physics".
- **Resolution (32² vs 64²) is cluster-only** (heavy compute deferred). With the physical parameters held fixed (so the
  physical Pe, Da, σ_s/L are unchanged and only the grid Péclet changes), the 64² nominal cell is the E07/E08 sweep world
  (VanLeer dt 30 s), whose cached fields exist on the cluster. The resolution contrast is 32² `diff_lo` (VanLeer) vs
  64² nominal (VanLeer): same scheme, refined grid. See `docs/research/hpc/e16-generalization-run.md`.
- Axes the plan names but deferred here: sensor noise level and cadence (E08 already varies them), spatial cost mask,
  held-out wind direction regime (E07 follow-up and E12 cover it; per-regime RMSE is reported secondary).
  Model-vs-measurement mismatch is E12/E13; every cell here has a perfect model (truth and estimator share physics).
  **[LEAD REVIEW]**

## 4. Pairing and split isolation

- Scenario = (K, seed), K ∈ {2, 4}, seeds 0–4 → n = 10 independent clusters per cell (one training wind, so no
  veering/mixed sharing). Same seed ⇒ same truth stream, wind calendar, noise stream, placement seed and strengths in every
  cell. Source **layouts** are identical in every cell except `src_sparse` (same RNG permutation, different acceptance).
- Truth and estimator use the same (perturbed) physics within a cell: the axes are changes of the world, not mismatch.
- Designs see only training-hour responses (`DesignInput`), sigma, a_ref, source locations, candidates and the placement
  seed. Estimators see readings up to each issue time. Truth enters only the scorer. Noise draws are shared by all designs
  and both estimators within a scenario.

## 5. Ranking-stability metric and verdict rule (predefined)

For each conclusion C1–C4 and each cell c, with n = 10 paired scenarios:
1. **Contrast in the cell:** median ρ_c, distribution-free 95 % order-statistic CI for the median (`df_median_ci`; at
   n = 10 this is (x₍₂₎, x₍₉₎), coverage 0.979), wins/losses and sign-test p.
2. **Verdict:** `holds` if the CI lies on the nominal sign's side of 0; `reversed` if it lies on the other side;
   `weakened` if it straddles 0 with the median on the nominal side; `flipped (indicative)` if it straddles with the
   median on the other side.
3. **Axis effect:** Δρ = ρ_c − ρ_nominal paired by scenario, with the same CI. Reported, not a verdict.
4. **Ranking stability:** per scenario, Kendall τ between the 4-strategy ranking (by forecast RMSE at k = 8) in cell c and
   at nominal, for each estimator; report the median τ and the fraction of scenarios whose top strategy is unchanged.
5. **C5:** median k*(0.2) per strategy and K; fraction of scenarios with k* ≤ K for source_near and dopt, and > K for
   random. `holds` if ≥ 80 % of scenarios satisfy k* ≤ K for both location-aware designs and the median random k* > K;
   otherwise `conditional`. ks tested: 2, 4, 8, 16 (k* > 16 coded as 64).

A conclusion is called **robust (pilot)** if it `holds` in every cell, **conditional** if any cell gives `reversed` or
`flipped`, otherwise **weakened in some regimes**. With 7 non-nominal cells × 5 conclusions this is a screen of 35
verdicts without multiplicity correction; the pilot is exploratory and every verdict is indicative. At 64² the
sign-test p-values within each conclusion are Holm-adjusted across cells.

## 6. Stopping rule (no matrix expansion after borderline results)

- The pilot cells, levels, seeds and ks above are fixed now. No level, axis or seed is added locally after seeing results.
- The 64² plan may contain only cells listed here (plus the resolution cell). A cell whose pilot verdict is `weakened` or
  `flipped` is run at 64² once at the preregistered n (20 seeds); if it is still borderline there, it is reported as
  **unresolved** and no further levels, seeds or interactions are added for it.
- A pilot `reversed` result is not expanded into a dose-response series; one extra level would need a lead decision
  recorded before it is run.

## 7. Scales and cost

- Pilot (local, ≤ 6 workers): 8 cells × 10 scenarios; solves per scenario = 1 + K; 320 solves. Upwind ≈ 50 s/solve
  (276 simulated hours), VanLeer dt 75 ≈ 315 s/solve → ≈ 10 CPU-h, ≈ 2 h wall on 6 workers. Evaluation is seconds per
  scenario (cached Gram scorer).
- 64²: one job, ≤ 64 CPUs, VanLeer dt 30 s, full protocol calendar, 20 seeds, K ∈ {2, 4}; cost projected from the pilot
  and the E07/E08 sweep timings in the run document.

## 8. Implementation

- `src/pravah/experiments/generalization.py`: axis table, config builder with range/unit validation, per-scenario
  evaluation (designs + both estimators + k*), contrast/verdict/Kendall statistics. Reuses `cross_objective` (designs,
  map24, readings, CIs), `forecast_design.estimate` (map_grow), `sufficiency.GramScorer` and `breakpoint`.
- `scripts/e16_generalization.py`: stage 1 (field solves cached on disk with content hashes, as E07) and stage 2
  (evaluation), `--pilot`, `--sweep`, `--report`, `--workers`. Reuses the E07 driver's field cache (`scenario_fields`,
  `run_field`, `load_scenario`) by registering one E07 scale per cell. No new solver.
- `tests/test_generalization.py`: parameter ranges, units, paired seeds, split isolation, unique run ids.

## 9. What this cannot establish

A perfect-model 2-D passive tracer with uniform winds. Robustness here means robustness to the benchmark's own physical
parameters, not to real atmospheric variability, chemistry or vertical structure. Ten scenarios per cell detect only
large effects; a `holds` verdict at n = 10 needs ≥ 9/10 same-sign scenarios.

## 10. Deviations

1. **VanLeer dt 75 s → 60 s (2026-11-14, engineering, before any VanLeer-cell metric).** The pilot's 75 s step violates
   Courant ≤ 1 for seeds 2 and 4 (max |u| 4.41 and 4.6 m s⁻¹; Courant 1.06/1.10); the solver refused them. dt 60 s gives
   ≤ 0.88 for every pilot seed (`test_vanleer_courant_over_every_calendar`). The dt-75 caches of seeds 0, 1, 3 are unused.
2. **Pilot reported in two parts.** The six upwind cells (nominal, diff_hi, loss_hi, src_wide, src_sparse, bg_hi) are
   complete (n = 10 each) and are the tables committed as `e16-local-*`. The two VanLeer cells (`diff_lo`,
   `diff_lo×loss_hi`, 78 solves at dt 60 s, ≈ 1.5 h on 6 workers) were still solving at handoff; rerunning
   `scripts/e16_generalization.py --pilot --workers 6` (resumes from `.cache/e16/local`) regenerates all tables with 8 cells.
3. **Observation, no rule change:** `src_sparse` changes the source layout in only 4/10 scenarios (all K = 4); at K = 2 the
   nominal draws already lie ≥ 3000 m apart. Its effective n for the axis effect is 4; its other 6 scenarios equal nominal.
4. **Observation:** `bg_hi` makes the location-blind designs fail τ = 0.2 at every tested k in many scenarios (k* coded
   > 16), so C5's verdict there partly reflects the k-range ceiling, not only the designs.
