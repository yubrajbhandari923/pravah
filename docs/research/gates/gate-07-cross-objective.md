# Gate-07: independent audit of E07 (cross-objective placement, 64² sweep)

**Date:** 2026-11-13 · **Reviewer:** independent gate agent (no edits to src/, scripts/, state/, tables).
**Object:** job 204894, code dd2c51c (+ `confirmatory()` added in 22d2700 after the sweep), tables
`reports/quarto/tables/e07-sweep-*`, design note `docs/research/cross-objective-design.md` (3e7d4d2 + §12 deviations 1–6).
**Check scripts and outputs:** `docs/research/gates/gate-07/check_tables.py` (+ `.out`), `check_fields.py` (+ `.out`).
Field checks used the cached truth and unit-response fields of two scenarios, copied read-only from
`<hpc-host>:.../e07-sweep-dd2c51c/fields`: `veering:2:7` (where dopt does worst) and `mixed:4:9` (largest superposition
error). Cache names recomputed locally matched the cluster files, so the local config equals the run's config.

## Verdict: PASS WITH CONDITIONS

The numbers are reproducible and free of leakage. The confirmatory rule is implemented faithfully, and the main contrasts
are robust to the statistical choices I tried. The conditions concern **what the results may be said to show**: the
scope of the confirmatory claim, credit for the gain (source-location knowledge vs optimisation), and the precise
mechanism.

## What I verified (no issue found)

| Check | Evidence |
|---|---|
| Confirmatory table reproduced from rows with my own code | 8/8 cells identical to `e07-sweep-confirmatory.csv` (max abs. difference 0.0 in median and CI) |
| `confirmatory()` matches the committed rule | design note §7 + dev. 6: dopt vs mean of 10 random RMSEs, white + MAP, cells (wind, K, k ≥ 8), CI lower bound > 10 % in ≥ 70 % of cells; contradicted/unresolved branches correct |
| Forecast scores reproduced from fields by an independent MAP/forecast implementation (explicit per-window loop) | dopt/source_near k = 8: 0.16772/0.02512 (veering:2:7), 0.04697/0.03519 (mixed:4:9), equal to the committed rows to 5 digits |
| Leakage: designs see training hours only | dopt and source_near rebuilt with every non-training hour of G3 set to NaN give the committed sites (both scenarios). Truth is read only in `score` and `readings`. `tests/test_cross_objective.py` and `tests/test_e07_sweep.py`: 13 passed |
| Pairing and seeds | One set of R = 5 noise draws per arm, shared by every design (draw 0 rebuilt independently from `SeedSequence([2000+s, 0])`). Random networks use `rng([3000+s, j])`. Test winds depend only on the seed, so they are the same for both training-wind arms |
| Issue times | 425 per scenario (test hours 1014–1438). The forecast hour stays inside the test segment (declared in note §5) |
| VanLeer superposition (A2) does not bias the comparison | see issue 7: model-error floor ≤ 0.0021, and rescoring on a linear truth changes each RMSE by ≤ 1e-4 |
| `podqr` undefined handling | 12 undefined rows (4 per arm: 1 at k = 8, 3 at k = 16), scored −∞ / excluded from "best" as declared; it does not touch the confirmatory test |

## Issues, ranked

### 1. (Major, scope) The confirmatory verdict is valid only for the reduced cell set; the protocol's own rule calls the rest unresolved
Protocol §6/§12: N ≥ 8 cells span K ∈ {2, 4, 8} × N ∈ {8, 16, 32} with 20 seeds, and §12 says cells NOT RUN count as
"unresolved". Deviation 6 (a user-requested compute cut, decided before any 64² run) redefines the denominator as
2 winds × K ∈ {2, 4} × k ∈ {8, 16}, with 10 seeds. That is a change to the confirmatory claim. The approval decision says
such changes "still go to the user". Deviation 6 records that the user requested the scope cut, not that the user
approved redefining the verdict's denominator.
**Fix:** state the result as "supported in all 8 cells run (K ∈ {2, 4}, k ∈ {8, 16}, 10 seeds); K = 8 and N = 32 NOT RUN".
Ask the user to confirm that the reduced cell set is the confirmatory population. Otherwise report the protocol-level
verdict as "supported in the run cells, unresolved overall".

### 2. (Major, interpretation) "dopt beats random" must not be read as "optimisation helps"
The locations-only control `source_near` beats `dopt` on held-out forecast RMSE in 115/120 (white), 110/120 (ar1) and
118/120 (persite) paired (scenario, k) comparisons. The median relative advantage is 0.41 (IQR 0.23–0.60, white). It
holds in all 20 source layouts (median 0.41, bootstrap CI 0.28–0.56 over layouts). Most of dopt's gain over random comes
from knowing source locations, and the D-optimisation on training winds costs accuracy relative to simply sitting next
to the sources.
**Fix:** every statement of the confirmatory result carries this sentence (wording below).

### 3. (Moderate, mechanism) Lead's mechanism is right in direction; the specific failure is per-window rank loss under one unseen wind direction
- In both field-checked scenarios, dopt has **zero** rank-deficient 24 h windows over the training hours but 17.6 % in the
  test period. All 75 deficient windows lie inside the held-out **steady_southerly** blocks. Training winds come from
  270–315° (plus 90°/270° reversals in `mixed`); a wind from 180° is never seen. `source_near` has 0 % deficient windows
  in both periods.
- In veering:2:7 those windows carry 97.5 % of dopt's squared forecast error. In mixed:4:9 they carry 37 %.
- Sweep-wide, white arm: dopt has at least one rank-deficient test window in 32/30/26 of 40 scenarios at k = 4/8/16,
  against 20/7/6 for source_near.
- The aggregate test-period D criterion does **not** see this: D_test(source_near) > D_test(dopt) in only 35/120 cases,
  and selecting by D_test still costs a median forecast regret of 0.37. A whole-period information criterion, even one
  computed on the true test winds, misses short windows in which a source is unseen.
- So "dopt overfits training wind directions" is supported. The claim "sensors at sources stay informative whatever the
  wind" is supported for this geometry (issue 4). The evidence rests on one unseen direction (southerly), and the window
  analysis covers 2 scenarios in detail.

**Fix:** state the mechanism as below. Recording `frac_window_rank_lt_K` per test regime across the sweep would make it
sweep-wide; that needs only designs and evaluation on the cached fields, no new solves.

### 4. (Moderate, artefact risk) Near-source sensors sit inside the emitting area
Sources are Gaussian discs with σ = 400 m, truncated at 1200 m. The admissible mask only removes sites within 400 m.
source_near's nearest sites are 456–643 m from source centres, where emission is about half its peak, so they see local
emission under any wind. At k = 8 a median of 5.5 of 8 source_near sites (and 4 of 8 dopt sites) lie inside the 1200 m
disc.
**Control (check (h)):** I rebuilt both designs with every site within 1200 m of any source removed. source_near still
beats dopt:

| Scenario | k | source_near | dopt |
|---|---|---|---|
| veering:2:7 | 8 | 0.043 | 0.203 |
| veering:2:7 | 16 | 0.036 | 0.201 |
| mixed:4:9 | 8 | 0.115 | 0.229 |
| mixed:4:9 | 16 | 0.063 | 0.209 |

So the win is not only in-footprint sensing, but this control covers 2 scenarios.
**Fix:** declare the geometry (sources are 400 m area sources; sensors are allowed inside the emission disc) next to the
claim. Optionally run the >3σ exclusion control on the whole sweep from the cached fields.

### 5. (Moderate, statistics) The percentile bootstrap of a median at n = 10 is anti-conservative; the verdict survives
In the veering K = 2, k = 16 cell, the bootstrap lower bound is 0.228. The distribution-free order-statistic interval
(x(2), x(9), 97.9 % coverage) has lower bound −0.192, because dopt is worse than random in 2 of 10 seeds (worst −0.71).
With order-statistic CIs, 7/8 cells are above 10 % (87.5 % ≥ 70 %), so the verdict is still "supported". With a bootstrap
of the mean (protocol §10.1 says "averaged"), 8/8 cells are above. Pooled over winds (pilot-style cells), all 4 cells are
above, with lower bounds 0.42–0.72.
**Fix:** report these sensitivity variants next to the verdict.

### 6. (Minor, multiplicity and dependence) Cells and pairs are not independent
The two training-wind arms of a seed share source layout, strengths, test winds and noise seeds, and k = 8/16 reuse the
same scenarios. "8/8 cells" is therefore far fewer than 8 independent replications, and the 115/120 sign test
(p ≈ 3e-28) treats clustered pairs as independent.
**Fix:** quote cluster-level evidence (20/20 layouts; scenario level: source_near better at every k in 36/40 white, 31/40
ar1, 39/40 persite). Drop pooled-pair p-values. The source_near-vs-dopt contrast was not preregistered: label it
exploratory.

### 7. (Minor, A2 record) The superposition error is 27× larger over the full calendar than the A2 decision quotes; its effect on E07 is negligible
`linearity_max_abs` reaches 0.183 µg m⁻³ (0.73 σ) in mixed:4:9, against 6.7e-3 measured over 96 h in A2.

| Measure | veering:2:7 | mixed:4:9 |
|---|---|---|
| Forecast RMSE with the true strengths (model-error floor) | 0.00015 | 0.0021 |
| Max superposition residual during test hours | 0.017 | 0.112 |
| RMS residual at the chosen sites | ≤ 0.01 σ | ≤ 0.01 σ |

Re-scoring both designs with readings and truth from the linear superposition (same noise draws) changes the RMSEs by
≤ 1e-4 and keeps the ordering.
**Fix:** add the full-calendar figure and this impact check to the A2 decision or the E07 journal.

### 8. (Minor, interpretation) Endpoint agreement is close to built in
All four endpoints are functions of the same static estimate â under a perfect model and constant sources. Agreement
between them (forecast-best equals source-best in 116/120; off-diagonal median regret 0) is expected and is weak
evidence about the plan's "no single network wins" hypothesis. Likewise, the D_train selector's regret (0.70) largely
restates dopt's regret, because D_train picks dopt in 110/120 cases.
**Fix:** wording below.

### 9. (Minor) dopt is a local optimum of its criterion
- Hotspot has a higher D_train than dopt in 10/120 cases.
- In the 5×5 block check, greedy + swap is exact in 146/160 cases (max relative gap 0.11); greedy alone is exact in
  91/160.
- This does not threaten the comparison: source_near never has the highest D_train, and a better D_train optimum would
  only fit the training winds more closely.

**Fix:** none needed; report as is.

### 10. (Minor, records)
- Deviation 4 predicted that `podqr` would be "mostly undefined" at k = 16. In the sweep it is undefined in only
  1 (k = 8) and 3 (k = 16) of 40 scenarios per arm. Update the text.
- σ ≈ 0.25 ≈ 0.05·c_bg: the median candidate site barely sees a plume, so σ is in effect a fixed instrument noise, and
  near-source SNR is about 50–80. State this, since it explains why the persite arm, which raises σ near sources, changes
  little.
- Run provenance is adequate: `git_rev` is empty in the meta file but recorded in runs.jsonl. `confirmatory()` was
  committed after the results; I re-derived it independently.

## Defensible wording

**Confirmatory (protocol §12, scope per deviation 6, pending user sign-off on issue 1).**
> In the 64² synthetic sweep (K ∈ {2, 4}, k ∈ {8, 16}, training winds veering or mixed, 10 seeds per cell, white noise,
> MAP with a 24 h window), the D-optimal network reduced held-out one-step forecast RMSE relative to the mean of 10 random
> networks by a median of 42–84 % per cell. The 95 % bootstrap CI lay above 10 % in 8/8 cells, so the preregistered
> hypothesis is supported in the cells run. With distribution-free intervals, it is supported in 7/8. K = 8 and k = 32
> were not run. A locations-only control placing sensors next to the known sources did better still, so the gain over
> random is attributable to knowing source locations, not to D-optimisation.

**Exploratory.**
> Placing sensors at the nearest admissible sites to the known sources gave lower held-out forecast RMSE than the
> training-wind D-optimal design in 115/120 paired comparisons (110/120 with AR(1) noise, 118/120 with per-site σ). This
> held in all 20 source layouts, with a median relative improvement of about 0.4. In the scenarios examined, the
> D-optimal networks lost per-window identifiability of at least one source when the held-out wind blew from a direction
> absent from training (steady southerly), while the near-source networks did not. Whole-period information criteria,
> whether on training or test winds, did not predict this.

**Not warranted.**
- Any statement about real sources, real winds, point sources, or deployment.
- "Optimised placement is better than random" without the source_near caveat.
- "No criterion mismatch between endpoints" as a general finding: in this perfect-model, static-source setup it is
  structural.
- Any generalisation beyond one unseen wind direction, known source locations, a 400 m area-source footprint that
  sensors may sit inside, a known background, and the 24 h estimator window.

## Recommendations (cheap, no new solves)
1. Get user sign-off on the confirmatory scope (issue 1). Until then, label the verdict "supported in run cells".
2. Add the order-statistic, mean-bootstrap and pooled-wind sensitivity analyses to the confirmatory table (issue 5).
3. From the cached fields, compute rank-deficient-window fractions per test regime for every design and scenario, and run
   the >1200 m exclusion control sweep-wide (issues 3, 4).
4. Update the A2 note with the full-calendar superposition error and the impact check (issue 7).
