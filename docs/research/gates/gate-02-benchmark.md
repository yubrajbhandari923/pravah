# Gate 02 — Independent review and freeze conditions for the synthetic benchmark (T-012)

**Verdict: CONDITIONALLY CLEARED as engineering for the smoke-scale pipeline (three tiny scenarios). NOT CLEARED to freeze `benchmark-v1` as a scientific benchmark.** Four items need a human decision first (§8). This is an engineering audit, not a scientific approval, and not a review of finding F-002 (that still needs a promotion reviewer on a different agent profile). I changed no code, frozen test, metric or claim. Nothing was frozen by me: freezing is a human act.

**Reviewer:** claude-balanced / claude-sonnet-5-5. The code under review (T-009 spec, T-010 code, T-011 run) was written by earlier tasks of the same model family. Independence is by task and re-reading only. The mandatory independent promotion reviewer is still outstanding.

**Scope read:** `docs/research/truth-design.md`, `src/pravah/benchmarks.py`, `observations.py`, `experiment_config.py`, `tests/test_benchmarks.py`, `tests/test_e02_examples.py` (names and the manifest/determinism tests), `scripts/e02_examples.py` (header, examples, mass budget), `reports/quarto/tables/e02-checks.csv` and `e02-splits.csv`, `configs/benchmark-v1.yaml` (head only), `preregistered-protocol.md` (budget/split lines), gate-01, and the ledger records of EXP-003 (RUN-0007…RUN-0010). I did not open the PNG, did not render the Quarto chapter, and did not read all 1566 lines of the YAML.

**Provenance.** I ran no new computation and registered no run (EXP-003 has 4 unsynthesized runs and the budget is 5). All numbers below come from the RUN-0010 records and `e02-checks.csv`. RUN-0010 ran at commit `6cbdc68`, clean tree, backend local. A re-run of the three benchmark test files with the project venv gave 41 passed (§9).

## 1. Truth / operator consistency

- **Operator.** `site_truth` is `bilinear_sample` of the stored hourly field at each lattice site. Readings are `site_truth + noise` (`test_readings_equal_truth_plus_noise_at_site`). The sampling operator is exact at cell centres. Lattice sites sit at cell index `2+4i` on 64² (centre of a 4×4 block), which is *not* a cell centre on the 4×4 block grid but is a cell centre of the fine grid, so sampling there is exact. On the 32² smoke grid the offset is `1+2i`, also a fine-grid centre. No mismatch found.
- **Truth vs forcing.** The forcing used by the solver is the same `source_basis` sum used by the mass check; the integrated rate equals `sum(a)` to 1e-16 in all three examples (`source_rate_err`). Units: a in ug s⁻¹, φ in m⁻², S = Σaφ/H in ug m⁻³ s⁻¹, consistent with gate-01 §3.
- **Anomaly formulation.** The generator solves the anomaly d = c − c_bg with c_bg = 0 and adds 5 ug m⁻³ afterwards. This is exact for the affine equation with Dirichlet inflow at c_bg, and it was chosen because FiPy's LU residual tolerance silently dropped sources at c_bg = 5 (per the code comment and the T-010/T-011 notes; I did not reproduce that). **Weakness:** this is a real solver sensitivity. The same absolute tolerance may also truncate the anomaly itself, since the anomaly is ~1e-3 (§3). The solver was verified (gate-01) only with generic O(1–5) concentrations, not at the 1e-4 level. See §3.
- **Mass budget.** Outflow is a *residual* (emitted − removed − stored change), 0.89–0.92 of emitted. It is a diagnostic, not a conservation test: with the `study` boundary there is no independent outflow flux to compare it with (gate-01 §6.1). The removal term uses an hourly trapezoid, so it is approximate.
- **Perfect-model only.** Truth and the `view.model` dict use the same parameters, so "truth/operator consistency" is by construction. No IMPERFECT variants exist in code.

## 2. Split assignment and independent sources of randomness

Streams, read from the code:

| Quantity | Seed | Notes |
|---|---|---|
| sources | `SeedSequence([1000+s, K])` | independent per K |
| noise | `SeedSequence(2000+s)`, children 0 (white) and 1 (AR1) | **not keyed by K** |
| winds | `default_rng(5000 + 100·segment + s)` | not keyed by K; segments 0/1/2 give 5000/5100/5200 |
| placement | `3000+s` | not used in `benchmarks.py` yet |

- **Leakage guard.** `Scenario.view` calls `check_no_leak`; tests and the 65 recorded checks show that test-period data is refused for placement and tuning, validation is refused for evaluation, held-out wind regimes are refused for fitting, and the view arrays are read-only copies with no truth fields. In the three examples all 65 checks pass, 0 fail. Hours of the three splits are disjoint.
- **Weakness W1 (noise shared across K).** Noise depends on `2000+s` only, so for one replicate the K = 2, 4 and 8 scenarios use the *same* standard-normal array (scaled by their own sigma). Results across K within a replicate are therefore not independent in their noise. This is fine for a paired comparison but must not be treated as independent replication across K when computing uncertainty.
- **Weakness W2 (winds shared across K).** Same reasoning: winds depend on the replicate and segment, not on K. The three examples also use replicates 0, 1, 2, so their wind streams differ, but the real design shares them across K.
- **Weakness W3 (sources shared across splits).** Already stated in truth-design §3 and open question O-1: train and test differ in calendar and wind regime, not in source scenario. The benchmark tests generalisation across winds only. I confirm this from the code: one `draw_sources` call feeds all splits.
- **Weakness W4 (validation = same regimes as train).** `make_winds` for the validation segment uses the *training* regimes with a different seed. Validation therefore measures nothing about regime shift. If tuning on validation is intended to predict test performance under held-out regimes, it will be optimistic. This is a design limitation, not a bug.
- **Weakness W5 (spin-up winds).** The spin-up replays the first spin-up hours of the training stream. It is train-side information only, so no leakage, but it is an unreviewed [decision].
- **Weakness W6 (noise sigma from truth).** `noise_sigma` is 5 % of the median *train-period truth* over all 256 candidate sites, and the view exposes it. This uses hidden truth for the instrument-noise level, and since the median is dominated by the 5 ug m⁻³ background it is a near-constant 0.25 ug m⁻³. Harmless for leakage, but the spec calls the noise "5 % of background" and the code calls it "5 % of median truth"; they coincide only because the plume is negligible (§3).
- **Weakness W7 (smoke-scale only).** The split checks ran on a 4/1/3-day calendar (days 1–4, 5, 6–8) and 32² grid. At the full 30/12/18-day calendar with 48 h blocks, only the 8-day smoke calendar was exercised. The first validation block is `steady_westerly` only. Regime coverage of the full calendar (e.g. all three held-out regimes appearing in days 43–60, which has 18 days = 9 blocks) was **not run**.
- `check_no_leak` raises `LeakError` for days that straddle segments only through `segment_of` on each day; the day list comes from the hours the view requests, so the guard is as strong as the day arithmetic in `split_hours` (inclusive 1-based ranges, `(lo-1)*24 … hi*24`). I traced it by hand for train = (1,30): hours 0…719. Correct.

## 3. Known weaknesses of the benchmark as defined (record, do not hide)

1. **No detectable signal at the preregistered scale (negative result, F-002 preliminary).** Peak anomaly 5.3e-4, 6.5e-4, 8.1e-4 ug m⁻³ against noise sigma 0.25 ug m⁻³; SNR 2.1e-3, 2.6e-3, 3.3e-3 (RUN-0010 metrics `*.snr_peak`). At that SNR a source-recovery or placement comparison would measure noise. This is the gate-00 B1 concern, and it is now empirical. A sanity bound agrees in order of magnitude: 100 ug s⁻¹ spread over a depth of 500 m and a wind of a few m s⁻¹ and a plume width of order km gives ~1e-4 ug m⁻³. So it is plausibly physics, not a bug. I did not run an independent oracle for it.
2. **Solver precision at the plume level is unverified.** Because of §1, the solver's absolute linear-solve tolerance is of unknown size relative to a 1e-4–1e-3 anomaly. The anomaly trick makes the background irrelevant but the plume is still tiny. A test that the field changes linearly with the source amplitude (double a, double d) at the benchmark scale would settle it. Not run.
3. **Numerical truth, not PDE truth.** Gate-01 §5: upwind numerical diffusion at 4–5 m s⁻¹ on 64² is far above K = 50 m² s⁻¹. The truth is "the output of this scheme". Convergence ≠ accuracy at 64².
4. **Wind jumps and face reclassification.** Wind changes hourly and at 48 h block edges; gate-01 §6.2 says the face-switch path is untested. The benchmark exercises it unverified. `weak_variable` flips direction by 180° every 6 h.
5. **Mass budget is a residual only** (§1). Boundary behaviour has no independent oracle (gate-01 §6.1).
6. **Truth for only three hand-placed examples exists** (`fixed_sources` overrides `draw_sources`; the override does not enter `truth_id`, so the example truth IDs describe a config that did not generate them for source placement). The 60 production scenarios were never generated; only their source draws exist in the YAML (RUN-0007, 1 s).
7. **Realism.** Uniform winds, one fixed depth, no PM2.5 chemistry. No claim about real exposure.

## 4. Immutable test design (proposal for the human freeze)

I propose, for human approval, freezing the following as they stand. Freezing records hashes; it does not approve the science.

- `tests/test_benchmarks.py` (operator, noise, truth completeness, source basis normalisation, split enforcement, view isolation, truth-id link, manifest match).
- `tests/test_e02_examples.py` (examples, mass budget, leak checks, determinism, shared truth across designs).
- `tests/test_experiment_config.py` (config and leakage guard).
- The previously frozen forward tests from gate-01 stay as they are.

Hidden-test conditions to record at the freeze:

- Test split = days 43–60 only, held-out regimes only (`steady_southerly`, `stagnation_then_front`, `diurnal_rotation`). The 426-issue pre-assimilation endpoint window is protocol §10 and cannot be replaced.
- Test truth, test winds and test readings are reachable only through `view(split="test", purpose="evaluation")`. Evaluation computed once per frozen configuration.
- Truth object (`Scenario.truth`) is public in Python. The separation is by convention plus `view`. An estimator that imports `Scenario` can read truth. The guard is a code-review rule, not a technical barrier. **Add to the freeze:** estimators must receive only `InferenceView` objects, enforced by a frozen import test (protocol §11 already requires this; no such test exists yet in this branch).

## 5. Supported source counts, window lengths, budgets

- **Supported by code and config:** K in {2, 4, 8} (`Design.n_sources`; `truth_id` refuses others); sensor budgets {2, 4, 8, 16, 32}; noise kinds white and ar1 (rho = 0.8); candidate lattice 16×16 with 196 static-admissible sites, ≥187 admissible in the worst of 60 scenarios (RUN-0007); minimum separation 1.25 km.
- **Window lengths snapshot / 6 h / 24 h / 7 d are declared in `Design.windows` but nothing in `benchmarks.py` implements windowing.** `view` returns a whole split. Window selection (and the rule that a 7-day window ends at the forecast issue time, protocol §6) is not implemented or tested. **Not supported yet.** The smoke calendar (8 days) is too short for a 7 d window in the test split plus warm-up.
- **Scale actually exercised:** 32², 12 h spin-up, 8-day calendar, K in {2, 4}. The preregistered scale (64², 48 h spin-up, 60 days, K up to 8) was never generated. K = 8 has never been run at all.
- **Cost.** RUN-0010 took 88 s for three scenarios at 32² with 12 h spin-up and 8 days. Extrapolation (not measured): the full scenario has about 7× the hours and ~4× the cells, so very roughly 30× the work per scenario, i.e. tens of minutes each, and 60 scenarios times two noise kinds (noise is cheap) would be of order tens of core-hours serial. The protocol cap is 48 core-hours and local-only. **This is an order-of-magnitude guess; run one full-scale scenario under a registered run before committing to the 60.**

## 6. Reproducibility and versioning gaps (why a version bump rule is needed)

`truth_id` hashes model, sources, winds, split, truth seed and K. It does **not** cover, and `config_hash` does not cover either, the following module constants and in-code definitions, so changing them would silently change the data while the ID stays the same:

- `SIGMA_REL` (0.05), `BLOCK_H` (48), `WIND_BASE` (5000), noise `rho` (0.8), the numeric parameters inside every wind regime (speeds, angles, U(0.8,1.2) scaling, N(0,10°) perturbation), clipping range [0.2, 5], source truncation (3σ), the spin-up rule and the lattice size `lattice_n` = 16 (a function argument).
- The solver revision and numpy version (the YAML records numpy 2.5.4).
- Noise settings other than the seed (`noise` is in `Design`, which is excluded from `truth_id` by intent, but `SIGMA_REL` is not in the config at all).

Also: the only golden data check is `test_scenario_matches_frozen_manifest`, which pins the *source draw* of `s00_K2`. There is no golden hash of a generated field, truth or noise array at the frozen scale. `test_deterministic` only compares two runs on one machine. The generator that wrote `benchmark-v1.yaml` is not in the repository (truth-design §8), so the YAML cannot be regenerated from the repo. The e02 example run is recorded as `git_dirty False` at `6cbdc68`, which is good.

**Requirements before any freeze (version-bump rule).**
1. Move the constants above into the config (or into a `BENCHMARK_VERSION` string that is part of `truth_id` and of the manifest).
2. Add a golden hash of the full-scale truth field and noise arrays for at least one scenario.
3. Any later change to a data definition (truth, winds, noise, sources, lattice, split, windowing, solver numerics) requires an approved version bump (`benchmark-v2`), a new `truth_id` family, and a re-run of all affected tests. Frozen tests are never edited to pass; a changed test needs a recorded human-approved correction.
4. The current `schema_version = 1` is the config schema, not a benchmark data version.

## 7. Immutable hidden-test conditions summary

| Condition | Status |
|---|---|
| Test days 43–60 only; held-out regimes only | Enforced in code; tested on the 8-day smoke calendar only |
| No test data for placement, tuning, model selection | Enforced by `check_no_leak`; 65/65 checks pass |
| Evaluate once per frozen config | A process rule; nothing enforces it |
| Estimator sees `InferenceView` only | A convention; no import test yet |
| Sensor placement from train information only | Enforced via `view` purposes |
| Noise identical across designs at shared sites | Tested (`same_truth` and shared-noise check pass) |
| Full-scale truth exists and is hash-pinned | **No** |

## 8. Decisions the human must make before the freeze (I made none)

1. **SNR.** Peak anomaly is ~1e-3 of the noise sigma at the preregistered source scale. Options are a protocol amendment (source rate, depth, noise definition) or accepting a benchmark that is uninformative. This is a scientific change; the preregistered values were left untouched. Until decided, no E03 or placement result is interpretable.
2. **Open questions O-1, O-2 from truth-design** (separate scenario seeds for tuning vs evaluation; mismatch magnitudes, 48 h blocks, 1σ site exclusion), and **W4** (validation uses train regimes).
3. **Gate-01 items still open**: the convergence-rule deviation, the scheme choice, the `study`-BC oracle and the wind-reversal test, since truth generation depends on them.
4. **Windowing** (snapshot/6 h/24 h/7 d) must be specified and implemented, and the **version constants** moved into the hashed configuration, before the freeze.

## 9. Frozen-test and re-run status

I did not freeze anything. The 36 forward tests were reported as passing in gate-01. For this gate I re-ran `test_benchmarks.py`, `test_e02_examples.py` and `test_experiment_config.py` with the project venv (`PYTHONPATH=src`, Python 3.14): **41 passed, 0 failed, 2 deprecation warnings (FiPy's `numpy.core` import), 195.7 s**. It is not registered as a run (EXP-003 has 4 of 5 runs used and nothing new was computed). These tests were not run under `research orchestrate verify` by me.

## 10. Handoff for downstream experiments

- **Commits on this branch:** `3bbf14b` (T-009 spec and YAML), `6cbdc68` (T-010 truth, measurements, split manifest), `3204de6` (T-011 tiny benchmark run and tables). "Verified" here means the checks pass and the code matches the specification at smoke scale. It does not mean the science is validated.
- **Downstream needs:** (a) the SNR decision above; (b) window selection code and tests; (c) the version constants in config plus a golden full-scale hash; (d) one registered full-scale run for timing before the sweep; (e) the paired-noise caveat W1 in every uncertainty calculation; (f) a frozen import test keeping estimators away from `Truth`; (g) synthesis of EXP-003 (4 of 5 runs used) and an independent review of F-002; (h) the solver linearity check at plume scale (§3.2).
- **Not done:** full-scale generation, K = 8, windowing, IMPERFECT variants, Quarto render, solver mutation tests, an independent promotion review.
