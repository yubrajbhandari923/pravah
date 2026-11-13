# PRAVAH E04 — source–receptor footprints, Fisher information and identifiability

**Task:** T-023 · **Experiment:** EXP-005 · **Author:** claude-balanced / claude-sonnet-5-5 · **Date:** 2026-11-10
**Status:** DRAFT derivation plus a smoke-scale check. Not frozen, not human-approved, not independently reviewed. It reports no supported finding.
Synthetic 2-D passive tracer, known wind, perfect model, known source locations. Nothing here is evidence about ambient PM2.5.

## 0. Prerequisite handoff (checked before writing)

| Item | What was checked | Result |
|---|---|---|
| Prior plan | `research plan status PLAN-001` (key `pravah-p1-foundations`, rev 9) | completed, 22/22 tasks verified |
| Promotions | `research promote list` | "no promotion candidates": none pending or in flight |
| Main checkout | `git status --untracked-files=all` in main | clean; HEAD `7678e81` ("docs: track literature references and PDF ignore policy"), the human commit that fixed the earlier dirty `docs/papers/` |
| Gate file and code | `docs/research/gates/gate-03-inverse.md` (added in `1ffb740`, T-016) and `src/pravah/{sensitivity,inverse,inversion_study}.py`, `tests/test_source_inversion.py`, `tests/test_inversion_study.py` | present at base `514b1a4` of this worktree, which is the commit before `7678e81`; `7678e81` only adds `docs/papers/` |
| Reviewer report | T-016 task record (output `docs/research/gates/gate-03-inverse.md`) | verdict **NOT CLEARED** to freeze the inversion baseline. Reviewer was claude-balanced / claude-sonnet-5-5, the same model family that wrote the code (the report says so), so independence is by task only; finding F-003 still awaits a promotion reviewer on a different profile |

The gate-03 verdict is a valid negative outcome, not a contradiction of the physical assumptions or the acceptance criteria, so this task proceeds. Its limits carry over: at smoke scale no fit was full rank, more than half of the 648 fits were worse than predicting zero, and the rank tolerance is not noise-referenced. I did not verify that main's `7678e81` content equals what this worktree sees beyond the file presence above.

**Sources not read.** Green (2006) is a scanned PDF with no extractable text; Bhardwaj & Subramanian (2026) is an unreviewed preprint; Schlutow et al. (2026) was not opened. Everything attributed to them below comes from `green-2006-reading-notes.md` and `literature-map.md`, is unverified against the originals, and every extension is marked **[PRAVAH adaptation]**. No equation here is claimed to be reproduced from those papers.

## 1. Causal impulse response and lagged sensitivities

From `inverse-derivation.md` §2 (same solver, same winds): the anomaly e = c − c_bg is linear in the source, with hourly state map Φ_h and source map Ψ_h (backward Euler, 12 sub-steps, wind u_h). Reading at site i, hour h:

```
y_{i,h} = c_bg + Σ_k a_k G_{(i,h),k} + ε_{i,h},   G_{(i,h),k} = H_i Σ_{τ ≤ h} [Φ_h ⋯ Φ_{τ+1}] Ψ_τ φ_k / H
```

- **Causal impulse response.** A release of unit rate during hour τ contributes to hour h only for h ≥ τ (the product is empty for h = τ and absent for h < τ). G is therefore lower-triangular in release time, and a reading never depends on future emissions.
- **Time-lagged sensitivity.** The sensitivity of reading (i,h) to a release in hour τ is the single term `H_i [Φ_h⋯Φ_{τ+1}] Ψ_τ φ_k / H`. With constant-in-time strengths (the only temporal factor implemented) the lags are summed into one column per source; with per-hour strengths there would be one column per (k, τ) and G would be the full causal block-triangular operator.
- **Units.** G is µg m⁻³ per µg s⁻¹. A row of G is the *footprint* of that reading: sensitivity to emissions, **not** a pollutant field and not a measured quantity. Columns are sensitivities of all readings to one source; they are not concentration maps either, although for a single source they coincide numerically with that source's anomaly sampled at the sensors.
- **Transient vs steady state.** If winds and sources were constant and the spin-up long, Φ^n → 0 and G collapses to the steady-state footprint (the Schlutow-type concentration footprint) **[unverified against the paper]**. Here winds change hourly, so rows at different hours are different footprints; a steady footprint would discard exactly the information that a time history adds, and would be wrong during the start-up transient (slowest mode ~ e^{−λT_s}) and after wind shifts. Conversely, the steady footprint is cheaper and is a valid limit only for stationary winds longer than the transport time. The repo implements the transient G (`sensitivity.scenario_response_matrix`); no steady footprint is implemented, so the two were not compared numerically.

## 2. Fisher information, criteria and local covariance

Stack all selected readings into y = G a + ε, ε ~ N(0, R) with R independent of a. The log-likelihood has score Gᵀ R⁻¹ (y − G a), so

```
M = Gᵀ R⁻¹ G         (Fisher information of a; K×K, symmetric, PSD)
M_post = M + P0      (Gaussian prior a ~ N(·, P0⁻¹); P0 = prior precision, needed when M is singular)
```

- **Additivity.** For independent time steps, M = Σ_h M_h; with correlated noise it is not a sum over hours but is still `(W G)ᵀ(W G)` with W = R^{-1/2} (Cholesky whitening, as in `inverse._whiten`). Independent sensors add when R is block-diagonal across sensors.
- **Spectrum link.** If W G = U S Vᵀ then M = V S² Vᵀ. Eigenvectors of M with small eigenvalues are the poorly informed **source combinations** (null or near-null modes); the singular spectrum of the stacked sensor-time response is the object the hypothesis of EXP-005 is about.
- **Criteria** (finite only for nonsingular M): D: maximise log det M (volume of the local confidence ellipsoid); A: minimise tr M⁻¹ (mean variance); E: maximise λ_min(M) (worst direction). When M is singular D is −∞, A is +∞, E is 0, and the criteria cannot rank designs: use the prior-augmented M + P0, or the rank and the partial spectrum **[PRAVAH adaptation of Green's D-opt idea, which assumes nonsingular M]**.
- **Local asymptotic covariance.** For the unconstrained linear-Gaussian model the maximum-likelihood estimate has covariance exactly M⁻¹ (Cramér–Rao attained), not just asymptotically; with a prior the posterior covariance is (M + P0)⁻¹. Limits: (i) a ≥ 0 and NNLS: at the boundary (a_k near 0) the estimator is not Gaussian and M⁻¹ over- or under-states spread, so confidence ellipses are valid only for a_k comfortably above 0; (ii) R misspecified, wind or location error: the covariance is conditional on the model being right (perfect-model benchmark only); (iii) M⁻¹ is meaningless when cond(M) exceeds 1/ε_machine.
- **Correlated noise.** Temporal correlation (AR(1), ρ = 0.8 per sensor in the check) lowers the information in repeated readings; ignoring it (using σ²I) over-states M. Cross-sensor correlation is not implemented in the benchmark (`noise_field` kinds were not exercised here).

## 3. Observability rank versus practical conditioning

Structural (generic) observability: rank G = K. Practical: how many singular values stay above both numerical precision and the noise floor. Two separate failures:

1. **Rank deficiency** (exact or numerical null modes): some combination of sources produces no reading change; no data, however many, fixes it. More sensors or longer history raise rank only if the added rows have components in the null space.
2. **Ill-conditioning**: rank is nominally K but κ(G) = s_1/s_K is huge, so the near-null combination has variance ∝ 1/s_K² ≫ the amplitude of interest. `rank_info` thresholds at 1e-8 relative to s_1 (not noise-referenced, as gate-03 noted); a noise-referenced rule would compare s_K / σ with the amplitude scale, which the CRB column below does.

Sensor count n and rank are different quantities: rank ≤ min(n·T, K), and the sensors in the check are fixed by a spread rule, not optimised.

## 4. Background nuisance projection

Add a uniform unknown background b with design column 1 (the E03 benchmark has a known background; here it is treated as nuisance, **[PRAVAH adaptation]** of the background projection attributed to Bhardwaj & Subramanian, unverified). With θ = (a, b), the joint information is block `[[M_aa, M_ab],[M_ba, M_bb]]` and the information on a with b profiled out is the Schur complement

```
M_a|b = M_aa − M_ab M_bb⁻¹ M_ba = (W G)ᵀ P⊥ (W G),   P⊥ = I − n nᵀ/(nᵀ n),  n = W 1
```

i.e. G with its component along the whitened background direction projected out. Consequences: (i) M_a|b ⪯ M_aa, so unknown background can only reduce information; (ii) any source combination whose footprint is constant over sensors and times (nearly uniform in the window) becomes unidentifiable from b; (iii) a persistent plume signal that barely changes across the window is exactly what a constant offset absorbs, so the loss is largest for slowly varying footprints.

## 5. Numerical check (one run, smoke scale)

`scripts/e05_footprint_fim.py`, 32² grid, 24 h window, known wind, perfect model, anomaly form, sensors chosen by the same deterministic spread rule as E03 (up to 8), σ from the scenario's noise (0.25 µg m⁻³), reference rate 100 µg s⁻¹. Examples `separated2` (K = 2) and `changing4` (K = 4) from E02. Truth strengths are not used except to scale the CRB. Metric names below are those logged by the run (values read from the run record).

**Identities (validity checks), both examples.** `*_eig_vs_sv2_rel_err` ≈ 5e-16 (M's eigenvalues equal squared singular values of the whitened G); `*_time_additivity_rel_err` ≈ 6e-16 to 9e-16 (M = Σ_h M_h); `*_corr_whitened_vs_direct_rel_err` ≈ 5e-16 (whitened vs direct R⁻¹ FIM with AR(1) noise); `*_schur_vs_projection_rel_err` ≈ 5e-16 to 2e-15 (Schur complement = projected form). These test the algebra of §2 and §4 only; they do not test the PDE model (that is E01/E03) and are not independent of the code being checked beyond two algebraically different routes.

**Rank and conditioning (tol 1e-8 on singular values), snapshot (last hour) vs 24 h window, n sensors:**

| example (K) | n = 1 | n = 2 | n = 4 | n = 8 |
|---|---|---|---|---|
| `separated2` (2): snapshot / window rank | 1 / 1 | 1 / 1 | 1 / 1 | 2 / 2, cond 2.74e7 (both) |
| `changing4` (4): snapshot / window rank | 1 / 1 | 2 / 2 | 2 / 2 | 2 / 2 |

`cond` is `inf` whenever rank < K (the `rank_info` convention of T-022).

Observations (descriptive, not tested hypotheses):
- No configuration reaches full rank for K = 4; adding sensors from 2 to 8 and moving from a snapshot to 24 hours leaves the rank at 2. Sensor count and time history did **not** raise rank here.
- For K = 2 the rank stays at 1 up to 4 sensors and becomes 2 only at 8, where κ is 2.74e7: full rank in the exact-arithmetic sense, practically ill-conditioned. With `n8_win24_lambda_min_over_max` = 1.33e-15, the D value (`logdet`) is −92.1, `trace_inv` is 2.7e27 and the largest CRB standard deviation is 5.2e11 times the 100 µg s⁻¹ reference: the local covariance is meaningless at this sensor set even though the rank test passes.
- `changing4` with 8 sensors over 24 h: M is numerically singular (`singular_M` = 1; the smallest eigenvalue ratio is −1.5e-18, i.e. rounding noise, so D/A/E are undefined and not reported).
- Background nuisance: for `separated2`, `lambda_min_ratio_nuisance_over_known` = 0.655, so profiling out a uniform background cuts the weakest eigenvalue to about two thirds. For `changing4` the ratio is −1.0e-4, a ratio of two numerically zero eigenvalues: **not interpretable**. Both `known_bg` and `nuisance_bg` FIM ranks are 2 for both examples, so background did not change the rank at this scale.
- Prior: with one reading (rank-1 M), M + P0 with P0 = I/(100 µg s⁻¹)² has condition number 1.0 (`snapshot1_posterior_cond`) in both examples. That is because the footprint information is ~1e-15 of the prior precision at the preregistered source scale (cf. F-002, SNR ~2e-3): the prior, not the data, determines the posterior. This is a regularisation artefact, not identifiability.
- Correlated noise: for `separated2` the AR(1) ρ = 0.8 FIM has log det lower than the white-noise one by 3.82 (`corr_over_white_logdet_diff`), i.e. assuming white noise over-states D-information at this ρ. Not computed for `changing4` (M singular).

## 6. What this does and does not say about the EXP-005 hypothesis

Hypothesis: the singular spectrum of the stacked sensor-time response predicts recoverable source combinations better than sensor count or coverage.

- **Not tested.** No comparison of spectrum-based prediction against a count or coverage predictor was run, no recovery error (NNLS vs prediction) was computed in this run, and the 1-D Green D-optimality comparator was not reproduced (the PDF is unreadable by text tools). Two sensor sets per example on one seed is not a controlled comparison.
- **Directionally consistent, nothing more.** For `changing4` rank is 2 for every sensor count from 2 to 8, so n ≥ K does not imply identifiability and the spectrum shows why (rank 2 < 4). For `separated2` rank rises only at n = 8. These are consistent with the hypothesis but the examples were also rank-deficient in E03 (F-003); a design with full rank (amplified sources or optimised sites) is needed before the spectrum can be compared with recovery error. Status: **unresolved**.
- **Sensitivity of conclusions to the rank rule.** The conclusion "rank 2" depends on the 1e-8 relative tolerance; `separated2` n8 sits at κ 2.7e7, (s_2/s_1 ≈ 3.6e-8), so a relative tolerance of 1e-7 would count it rank 1. Only the 1e-8 tolerance was run.

## 7. Limitations

Single scale and seed; two scenarios; fixed non-optimised sensors; perfect model and known wind; uniform background only; constant-in-time strengths (no per-hour release columns, so the "time-lagged" sensitivities are summed, not resolved); NNLS boundary effects not treated in the local covariance; steady-state (Schlutow-type) footprints not implemented or compared; Green and Bhardwaj not verified against the originals; unit of the ranking criteria depends on the scale of G, so D/A values are not comparable across scenarios without normalisation (only κ and ratios are scale-free).

## 8. Provenance

- Run: RUN-0016 of EXP-005, command `python scripts/e05_footprint_fim.py` (project venv Python), local backend, host RAD-88300596, 85 s, exit 0, git commit `514b1a4` (the script was untracked at run time and is captured in the run's snapshot ref `refs/research/snapshots/RUN-0016`), no random draws in the script (scenario seeds are those of the E02 examples).
- Code reused unchanged: `sensitivity.scenario_response_matrix`, `inverse._whiten`, `inverse.rank_info`, E02 `build_examples`.
- Plan handoff: PLAN-001 (`pravah-p1-foundations`), main `7678e81`, gate-03 report `docs/research/gates/gate-03-inverse.md` from T-016 (see §0).
- Numbers are quoted from the RUN-0016 metrics; no finding has been created from them and none is claimed supported.
