# E13 background, source-basis and observation-error mismatch: design note (pre-pilot)

Status: written 2026-11-13 on branch `experiment/e13-model-mismatch`, **before any E13 computation**, and committed alone.
Later changes go into §12 (Deviations) with their reason. Items marked **[LEAD REVIEW]** are choices the plan and
protocol leave open. Synthetic 2-D passive tracer, uniform hourly winds, known (nominal) source locations.
Template: `wind-mismatch-design.md` (E12). E12 mismatched the *model's wind*; E13 keeps the model fixed and makes the
*world* (truth emissions, background, sensors) violate the model.

Inputs: plan-03 key `e13-nonwind-mismatch`; `preregistered-protocol.md` §4 (background variant b ~ N(5, 1²); diurnal
source variant), §5 (white and AR(1) noise; bias reserved for plan 3); E10 (constant-source MAP is overconfident under
time-varying truth: coverage 0.09–0.25); E12 (fit stays good while strengths are wrong; persistence is the interpretable
reference because the oracle is near-perfect); gate-09 (ordering depends on estimator memory); gate-10/11 (cluster =
(K, seed)).

Scale facts measured before writing (32² pilot world, K = 2, seed 0; a single print, no E13 code): instrument
σ = 0.05 × median training site truth = 0.25 µg m⁻³ (the median is the 5 µg m⁻³ background); 90th percentile site
reading 6.0, max 15.6 µg m⁻³; domain-mean plume anomaly 0.29 µg m⁻³. So 1 µg m⁻³ = 4σ and is larger than the mean plume.

## 1. Question and hypotheses

Does the network stay reliable when actual emissions, background or sensors violate the inverse model? Which
violations are visible in the residuals, and which give a good fit with wrong, overconfident answers?

Hypotheses (stated before data; all **exploratory**, §9):
- **H1 (overconfidence).** Every non-oracle family lowers the 95 % coverage of the strengths below the oracle's, and in
  at least the background, common-mode, source-shift and missing-source families coverage falls below 0.5 at k = 8
  under map_full, while the sensor-fit ratio to the oracle stays < 1.3 (low residual, wrong answer).
- **H2 (omission vs unknown coefficient).** A background offset (or common-mode bias) is an *unknown coefficient*:
  adding a background column to the estimator (`*_bg`) removes most of the source bias. A missing source or a wrong
  footprint is an *omission*: no estimator in the set repairs it; the strengths near it absorb its signal.
- **H3 (unidentifiable pair).** At the sensors, a background offset δ and a common-mode sensor bias δ are the same data.
  The `_bg` estimator therefore helps the field forecast under the background offset and *hurts* it under the
  common-mode bias (it adds the bias to the map). Both give the same source estimates (bit-level up to noise).
- **H4 (time structure).** Correlated noise (AR(1)) leaves strengths roughly unbiased but under-covers (white-noise
  assumption overcounts data); dropout at random only widens intervals (coverage near nominal). Drift, the humidity-like
  gain and the drifting background bias map_full more than map24 (long memory averages a moving error). Under the
  diurnal truth E10's result reappears: map_full under-covers; map24 forecasts better.
- **H5 (diagnostics).** The whitened residual RMS and the lag-1 innovation autocorrelation warn (fit ratio > 1.1 or
  ac1 above the oracle's maximum over clusters) for the drift/diurnal/AR(1) families but **not** for a constant background
  offset or a common-mode bias. Two-sided: we report which families are silent.

## 2. Truth / inverse separation

- **Inverse/forecast model (fixed for every spec):** the E07 nominal model: unit responses F (K, hours, grid) of the K
  *nominal* source centres with footprint σ = 400 m, constant emission, the true wind, background known = 5 µg m⁻³,
  white noise with the protocol σ. Same F for every spec (built once; the oracle's F).
- **Truth:** per spec, a *world* = (truth field anomaly tf (hours, grid), noise-free site truth (sites, hours), reading
  corruption). Built by `src/pravah/model_mismatch.py` from (i) the unchanged E07 truth `benchmarks.generate_scenario`
  and (ii) extra *truth-side* unit solves (shifted / widened footprints, an extra source, diurnal emission) that the
  estimator never sees. The forecast is scored against the spec's truth only.
- **Oracle branch:** spec `oracle` ⇒ the world is the E07 truth with E07 white readings (bit-identical).
- **Designs:** the E07 designs from the nominal training responses (as E07/E12), so the same networks are scored
  under every spec (only the world changes; a designer cannot know about the mismatch).
- Tests (`tests/test_model_mismatch.py`): the oracle world equals the E07 truth and readings bit for bit; building any
  world never writes the scenario's arrays; the estimator inputs (A from nominal F) are identical across specs;
  shifted footprints integrate to 1 and give non-negative anomalies; noise processes have the declared marginal sd and
  lag-1 correlation; per-site bias is constant in time and common-mode bias equal across sites; dropout readings are
  NaN (never filled from the truth) and masked estimation equals estimation on the observed rows only.

## 3. Mismatch families (pre-declared magnitudes; µg m⁻³, σ = instrument sd ≈ 0.25 µg m⁻³)

All random draws use E13-only streams `default_rng([13000, family code, seed, K])`; magnitudes of one family share the
draws (scaled). Realism label: **R** = realistic simplification, **S** = synthetic stress test.

| spec | family | definition | label |
|---|---|---|---|
| `oracle` | — | E07 truth, white noise | — |
| `bgoff0.25`, `bgoff1` | background level | truth c_bg = 5 + δ everywhere, δ ∈ {0.25, 1} (1σ; 1 = protocol prior sd); inverse assumes 5 | R |
| `bgdrift0.25`, `bgdrift1` | low-rank drifting background | c_bg(t) = 5 + A[sin(2π(t − t₀)/24) + t/T_cal] (diurnal + linear trend; uniform in space; t₀ ~ U(0, 24)), A ∈ {0.25, 1} | R |
| `shift400`, `shift800` | source location error | each true centre displaced by d ∈ {400, 800} m (1σ, 2σ of the footprint) in a random direction (clipped 3σ inside the domain); inverse uses nominal centres | R |
| `wide800` | footprint shape error | true footprint σ = 800 m (inverse 400 m), same centres, same total rate | S |
| `missing1` | omitted source | one extra true source on the 12 × 12 lattice ≥ 1.5 km from all others, a ~ LogUniform(0.5, 5)·a_ref; inverse keeps the K nominal sources | R |
| `diurnal` | time-varying truth, constant-source inverse | protocol §4: g_k(t) = 1 + 0.5 sin(2π(t − t_k)/24), t_k ~ U(0, 24) (E10 link) | R |
| `ar1` | correlated noise | protocol §5 AR(1), ρ = 0.8, same marginal σ (`noise_field` kind `ar1`); inverse assumes white | R |
| `sitebias0.25`, `sitebias1` | per-site calibration bias | constant offset b_s ~ N(0, β²), β ∈ {0.25, 1} | R |
| `common1` | common-mode bias | all sensors +1 µg m⁻³ (truth field unchanged) — the sensor twin of `bgoff1` | S |
| `drift1` | sensor drift | b_s(t) = u_s · 1 · t/T_cal, u_s ~ U(0.5, 1.5) (all upward) | R |
| `humid0.3` | humidity-like gain | y = c·(1 + 0.3 h(t)), h(t) = ½ + ½ sin(2π(t − 4)/24), common to all sites (a diurnal RH proxy; **not** a Plantower calibration) | S |
| `drop0.2`, `drop0.5` | sensor outage | per-site two-state Markov chain, mean outage 12 h, stationary missing fraction f ∈ {0.2, 0.5}; missing readings are NaN and masked | R |
| `combined` | joint stress | shift400 + bgdrift0.25 + ar1 + sitebias0.25 + drop0.2 together | S |

Not included (scope): spatially structured background, multiplicative per-site gain, non-Gaussian outliers,
missing-not-at-random outages, wrong diffusivity, combined wind + non-wind errors (E12 × E13).

## 4. Estimators

Linear-Gaussian MAP, prior N(0, a_ref² I) on strengths, white noise with the nominal σ (E07/E12), on the nominal F:
- `map24`, `map_full` (E07 frozen 24 h window; all past hours = E09 kf_aug with q = 0);
- `map24_bg`, `map_full_bg`: plus one unknown uniform background offset b with prior N(0, 1²) µg m⁻³ (the protocol's
  background-uncertainty variant), constant over the estimator's window; forecast = nominal + b̂ + F â.

Missing readings: rows (site, hour) with NaN are dropped from the normal equations (A and z zeroed by an explicit mask);
no reading is ever imputed. Fit/innovation diagnostics use observed entries only.

## 5. Placements, scale and clusters

E07 32² pilot world (upwind, dt 300 s, 12 h spin-up, train days 1–4, validation 5, test 6–11; 264 h), training wind
`mixed`, K ∈ {2, 4}, seeds 0–2 ⇒ **6 clusters** (cluster = (K, seed)). k ∈ {2, 8} (plan). Designs: random (5 networks,
averaged per cluster), dopt (greedy + swap), source_near. 3 white/AR(1) noise draws (E07 R = 3 of the pilot's 5, as E12).

## 6. Endpoints (held-out test segment, E07 issue times, 6 h warm-up)

Truth-scored (against the spec's own truth): `forecast_rmse_h1` (main), `forecast_rmse_h6`, `field_rmse` (analysis RMSE
at the issue hour), `source_rmse` (a_ref units), `source_bias` (mean signed relative error of a_k over issue times,
sources, draws), `cover95_a` (nominal 95 % interval coverage of a_k), `false_conf` = fraction of (issue, source, draw)
whose 95 % interval excludes a_k **and** whose half-width is < 25 % of |â| (a confident wrong answer).
Observable: `sensor_fit` (whitened RMS residual at the issue hour), `innov_rms`, `innov_ac1` (lag-1 autocorrelation of
the per-sensor-centred whitened 1 h innovations; the plan's residual_whiteness).
For the missing source, `source_rmse`/`source_bias`/coverage are over the K modelled sources (the omitted one has no
estimate; its signal is in the field metrics).

References (pre-declared, learned from E12): persistence c(t) → c(t+h) and background-only, both on the spec's own truth
and the same issue times; skill `vs_persistence_h1` = (persistence − model)/persistence per cluster. Ratios to the oracle
D = metric(spec)/metric(oracle) are reported beside, because the oracle is near-perfect (E12 §12.2).

## 7. Statistics

Per cluster: draws and random networks averaged first. Summaries per (spec, k, design, estimator): median over 6
clusters, distribution-free order-statistic CI (= (min, max) at n = 6, 96.9 %), sign test. No pooling over k or specs.
6/6 gives p = 0.031; nothing is multiplicity-corrected; the pilot is descriptive.

## 8. Counterexamples / failure taxonomy (pre-declared search)

(a) deceptive fit: sensor_fit ratio < 1.1 and forecast ratio > 1.5 or coverage < 0.5; (b) silent families: median
innov_ac1 not above the oracle's max **and** fit ratio < 1.1 while coverage < 0.5; (c) `_bg` making things worse
(H3); (d) memory reversal: map_full worse than map24 under the spec but better under the oracle; (e) cells where no
estimator beats persistence at h = 1 ("no reliable algorithm").

## 9. Pre-declared vs exploratory

Everything in E13 is **exploratory** (protocol §8: model-error studies do not redefine the endpoint). Pre-declared here:
families, magnitudes, seeds, estimators, designs, endpoints, references, cluster rule, counterexample rules, 64² subset.
Anything added after seeing pilot output is labelled post hoc in §12.

## 10. Compute

Pilot (local, 4 workers, OMP_NUM_THREADS=1, fork pools): 24 nominal solves (6 truths + 18 unit responses) + truth-side
solves 18 (shift400) + 18 (shift800) + 18 (wide800) + 6 (missing1) + 18 (diurnal) = 102 solves × ≈ 48 CPU-s (E12
32² timing) ≈ 82 CPU-min ≈ 21 min wall; stage 2 is linear algebra (minutes).

**64² plan (for later; not submitted by this branch).** E07 sweep world (A2: VanLeer, dt 30 s, 1440 h, 48 h spin-up),
`mixed`, K ∈ {2, 4}, seeds 0–9 ⇒ 20 clusters; k ∈ {4, 8, 16}. Truth and nominal responses reused read-only from the
E07 sweep cache (content-hashed names, `--fields`; the dry run must show truth 0, src 0). New truth-side solves per
scenario: shift800 (K) + missing1 (1) + diurnal (K) = 2K + 1 ⇒ 10 × (5 + 9) = **140 solves × 4.4 CPU-h ≈ 616 CPU-h**
(E07 measured 4.4 CPU-h per 64² unit solve); 140/64 ⇒ 3 rounds × 4.4 h ≈ 13 h wall on 64 CPUs (≈ 18 h on 32 CPUs:
5 rounds × 4.4 h ≈ 22 h; prefer 64). Solve-free families (bg*, ar1, sitebias*, common1, drift1, humid0.3, drop*) are
all kept; shift400 and wide800 are dropped (pre-declared reduction; each would add 60 solves ≈ 264 CPU-h); `combined`
uses shift800 in place of shift400 at 64². Stage 2: ≈ 20 specs × E12's 230 CPU-s per scenario × 20 ≈ 25 CPU-h; stage-2
memory ≈ (K + 2K + 1) fields × 1440 × 64² × 8 B ≈ 0.5 GB per K = 4 task. Request: 64 CPU, 128 GB, 24 h. The exact
command is in §12/the handoff once the script exists.

## 11. Leakage and validity checks

Oracle bit-identity test; estimator-input identity across specs; per-scenario superposition check of the nominal truth
|c_bg + Σ a_k F_k − truth| (linearity; E07/E12); for truth-side superposition worlds the same check is the build itself
(recorded); estimators read no row after the issue time (map_series); no NaN reaches the normal equations.

## 12. Deviations

1. 2026-11-14 (after the pilot; does not change pilot numbers): `truth_solves` looked up `combined` in the pilot spec
   table, so a 64² dry run listed 200 truth-side solves (shift400 for `combined`) instead of 140. Fixed (the spec table
   is now passed in) and pinned by `test_sweep_solve_keys`. The pilot's specs, names and fields are unchanged (pilot dry
   run after the fix: 0 to solve).
2. **Post hoc observation** (no rule changed): the pre-declared `silent_failure` rule (§8 b) requires coverage < 0.5, so
   it misses the clearest silent case in the pilot: `common1` with a `_bg` estimator has perfect fit, white innovations
   and nominal strength coverage, while the field forecast is far worse than persistence (the bias is added to the map).
   Reported in the handoff as a field-level silent failure; the 64² run keeps the rule as declared and adds no new one.
3. The persistence row for `combined` uses the combined truth (shift400 + drifting background), as declared in §6.

### 64² submission (for later; not submitted). Snapshot of the committed revision; `$R` = PRAVAH_HPC_ROOT from
`settings.local.yaml`; `$H` = the host from the same file.

```bash
REV=$(git rev-parse --short HEAD)       # a commit containing this note and scripts/e13_model_mismatch.py
git archive --format=tar "$REV" | ssh "$H" "mkdir -p $R/code/snapshots/$REV && tar -x -C $R/code/snapshots/$REV"
ssh "$H" "cd $R/code/snapshots/$REV && UV_PROJECT_ENVIRONMENT=$R/envs/main ~/.local/bin/uv sync --frozen"
# 1) reuse check (1 CPU, minutes): must print {'truth': 0, 'src': 0, 'e13': 140}
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-mm -c 1 --mem=4G -t 00:20:00 --export=ALL,PRAVAH_HPC_ROOT=$R \
  -o $R/logs/%x-%j.out scripts/hpc/run.sbatch $R/envs/main/bin/python scripts/e13_model_mismatch.py --sweep \
  --fields $R/results/e07-sweep-dd2c51c/fields --out $R/results/e13-sweep-$REV --dry-run"
# 2) the run (check squeue first; ≈ 616 CPU-h solves + ≈ 25 CPU-h stage 2; ≈ 13 h wall on 64 CPUs)
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-mm -c 64 --mem=128G -t 24:00:00 --export=ALL,PRAVAH_HPC_ROOT=$R \
  -o $R/logs/%x-%j.out scripts/hpc/run.sbatch $R/envs/main/bin/python scripts/e13_model_mismatch.py --sweep \
  --fields $R/results/e07-sweep-dd2c51c/fields --out $R/results/e13-sweep-$REV --workers 64 --stage2-workers 32"
# 3) report locally after copying OUT/e13-sweep-*.csv/json into reports/quarto/tables:
uv run python scripts/e13_model_mismatch.py --sweep --report
```
The E07 sweep cache path is the one in `reports/runs/runs.jsonl` (E07 sweep, `results/e07-sweep-dd2c51c`). Reruns skip
finished solves and scenarios (atomic caches), so a time-out can be resumed by resubmitting the same command. If the
dry run shows truth or src > 0 the cache was not found: stop (≈ 78 extra solves ≈ 340 CPU-h otherwise).
On 32 CPUs instead: `-c 32 --workers 32 --stage2-workers 16 -t 30:00:00` (5 rounds × 4.4 h ≈ 22 h).

### Lead decisions (2026-11-14)
- The 64² run adds a **field-level silent-failure rule**, reported beside the pre-declared one and labelled post hoc:
  sensor fit ratio < 1.1, |lag-1| < 0.1, and forecast worse than persistence. It is prompted by `common1` with the
  background term.
- The background term is reported as a two-edged remedy that cannot be chosen from the sensors alone. Whether a
  colocated reference or an upwind background site fixes it is a question for E15 and E14.
- The 64² run is deferred (user decision 2026-11-13). The plan is in §12.
