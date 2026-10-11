# Phase 3 findings, limitations and Phase 4 recommendations

Status at handoff to HPC. Everything here is **synthetic** (emissions, winds, populations) and a **research testbed result**, not a validated
forecast. Findings are `preliminary` in Research OS unless stated; each one below was independently reviewed by a non-author agent, and the reviews
corrected every first draft (see "What reviews changed"). Numbers come from `results/phase3-*` files; finding IDs point to the evidence.

Kind of claim, per the Phase 3 brief: **mathematical verification** (M), **numerical convergence** (N), **synthetic-model robustness** (S),
**empirical validation** (E: none yet), **deployment recommendation** (D: none).

## Supported by evidence so far

| Claim | Kind | Finding | Strength |
|---|---|---|---|
| The geodesign pipeline runs on an unlike synthetic region with config + asset changes; only the raw-data loader needed a role-name change | M/S | T-023, `docs/transfer.md` | tested on one synthetic region; no real second municipality |
| Population layer conserves totals under regrid; metrics match hand computation | M | `tests/test_population.py` | unit tests |
| `vanleer` scheme: closure 3e-13, positive, order ~1.8-2.4 periodic test, nonlinear, needs sub-stepping, legacy outputs bit-identical (21 files, 0 differ) | M/N | `results/phase3-scheme/` | one test family |
| At dt 25 s the 500 m exponential field has ~5-6% (lower bound) space-time error vs 125 m, observed order ~1.0-1.2; mesh error dominates dt error at 500 m; the external-patch fingerprint is worst resolved (~10% at 500 m) | N | F-025 | one case, one wind |
| Unknown wind direction error raises field error ~1.4-1.6x at 5 deg and 3.5-3.8x at 30 deg (fixed wrong wind); for >=10 deg it outweighs the sensor-count effect; chi2/dof detects it, identifiability flags (data-independent) cannot | S | F-029 | toy domain, 32 test cases, inverse crime |
| A wind-ensemble estimator recovers part of the loss only when the true wind is near an ensemble member; its intervals are overconfident otherwise | S | F-029 | same |
| Outer-domain half-width needed for <1% exposure ~ U T /2 for the reversal scenarios (40 km here), set by the return window, grows with T | S | F-031 | one source geometry |

## Preliminary or contested

* **Population-aware design (F-026, F-030).** With greedy selection no objective effect is established (EXP-017). With exact selection (EXP-020, effective n = 4
  source placements): at 2 sensors on concentrated populations population-weighted layouts cut test population RMSE 63-74% at 6-7x area RMSE; at 8 sensors they
  raise test population RMSE 9-12% while looking 24% better on the design cases (overfitting); mixed beta = 0.5 is the gentler compromise. No Belauri-scale result.
* **Greedy selection is far from optimal at 2-4 sensors on the design objective (F-030)**: median excess 6.7x / 8.9x / 3.0x at budgets 2/3/4. The test-error cost
  was not measured. Phase 2 small-budget greedy layouts and their rankings (F-020/F-021) should be read with this in mind.
* **Prescribed-wind scenarios (F-031).** Channel offset (-43% exposure) is a placement effect; the vortex is -10%; reversals raise exposure +8% to +42%.
  Returned mass is good to ~10% (mesh dependence).

## Real inputs now in the repo (descriptive only, not yet used in an experiment)

* `data/population/`: WorldPop R2025A 100 m counts onto the 500/250 m Belauri grids (184,030 people in the domain, about 54,400 in the study polygon; modelled estimate, nodata -99999 declared, CC-BY-4.0 tag).
* `data/era5/`: ERA5 hourly 10 m wind, Jan and Jul 2023, 9 x 9 cells at 0.25 deg (`src/pravah/era5.py`, `scripts/era5_summary.py`, `results/phase3-era5/`). At the Belauri centroid January winds come mostly from W/WSW/WNW
  (WSW bin 12.9% of hours, 26.5% of hours from 225-270 deg, constant-sector runs up to 19 h) and July winds from E/ENE/ESE (WSW 0.9%). Two months of one year: not a climatology; ERA5 cannot resolve local flow.
  F-018 found that the external patch reaches the municipality when the wind carries it from the WSW side, so the WSW share is the quantity to carry into any wind-uncertainty study on real winds.

## Not done (needs HPC or a decision)

* EXP-015 stage 2: do sensor-design decisions (displacement, regret) converge with mesh/dt relative to a resampling baseline? Code built, ~2 h on the laptop. **All
  Belauri-scale sensor-site conclusions are conditional on it.**
* EXP-016: Belauri-scale population-aware design (specified in `docs/hpc-runbook.md`, not implemented).
* S5a: wind-robust and population-robust placement (conditional, not started).
* Real population data (WorldPop/GHS-POP) and real meteorology (ERA5): feasibility notes only (`docs/population-data.md`, `docs/winds.md`); nothing downloaded.
* Review-requested controls: wind-error toy with an offset prior grid, alpha sensitivity for the known-wind baseline, model error at zero wind error, other base-wind
  directions; population toy with a design-split control, the uniform fixture, greedy layouts scored on test; winds toy with T and lambda varied independently.

## What reviews changed (so later agents trust the right versions)

F-022 -> F-025 (order 1.4-1.7 -> 1.0-1.2; stated errors are lower bounds). F-023 -> F-026 (greedy suboptimality confounded the objective effect; "dense core
worse" was a 1.3% near-tie). F-024 -> F-027 -> F-031 (unconverged reference, boundary artefact, vortex "ridge"; then a converged re-run; U/lambda explanation
refuted by a control). F-028 -> F-029 (sign mixing, prior is 15 members not a 3x3 grid, coverage wording). F-030 was written after review. A claim without
an independent review should be treated as a draft.

## Decisions and rationale

* VanLeer is a verification tool and a less-diffusive truth generator, not a design arm: it is nonlinear, so cached source-response bases are invalid for it.
* Toy (synthetic-domain) experiments were run on the laptop; Belauri-scale and long runs are deferred to HPC by the PI.
* Alpha is frozen from validation (area objective, or strategy B for the wind toy) and the same value is used across compared methods.
* No DEM-to-wind coupling was built: terrain is context only; every wind is prescribed and labelled so.

## Compute used (laptop, M1, 8 cores)

Stage-1 convergence 43 min; EXP-017 7 min; EXP-018 5 min; EXP-019 4 min (+ pilot 8 min); EXP-020 17 min (6 workers); EXP-021 18 min; full test suite 15.6 min (532 tests).
Estimates for the deferred work are in `docs/hpc-runbook.md`.

## Phase 4 recommendations (ranked)

1. Run EXP-015 stage 2 on HPC (plus the optional reference-mesh replicate baseline and a 125 m design stage); decide the mesh for design.
2. Re-evaluate Phase 2 small-budget layouts with exact or swap-search selection and score greedy on test; rerun geodesign/geodesign_robust with the G2 fixes.
3. Build and run the Belauri-scale EXP-016 with exact/swap selection, replicated population draws, design-split control, bootstrap over placements.
4. Belauri-scale wind-error study with the review controls; then S5a (robust placement) if wind error is still design-relevant at scale.
5. One real dataset each: WorldPop/GHS-POP population for Belauri; a bounded ERA5 extract entering only through `winds.segments_from_series`, with its coarseness disclosed.
