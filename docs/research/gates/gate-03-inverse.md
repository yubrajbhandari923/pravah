# Gate 03 — Independent audit of the static inverse baseline (T-016)

**Verdict: NOT CLEARED to freeze the inversion baseline for fair placement comparisons.** The code is engineering-sound at the unit level (23 tests pass), but at the only scale run (smoke, fixed sensors, 24 h window) no fit is full-rank, so the baseline has no demonstrated recovery regime. Freezing is a human decision (§8). This is an engineering and claim audit, not a scientific approval, and not a review of finding F-003 (that needs a promotion reviewer on a different agent profile).

**Reviewer:** claude-balanced / claude-sonnet-5-5. The code under review (T-013 to T-015) was written by earlier tasks of the same model family; independence is by task and re-reading only. I changed no code, frozen test, metric or claim, and registered no run (EXP-004 is at 5/5 runs, synthesis pending).

**Scope read:** `src/pravah/inverse.py`, `sensitivity.py`, `inversion_study.py`, `scripts/e03_inversion.py`, `tests/test_source_inversion.py` and `test_inversion_study.py` (grep only), `e03-inversion.csv` (all 648 rows), `e03-nonidentifiable.json`, `e03-validity.md`, `e03-compare.md`, gate-02, and the T-015 task result. I did not read the RUN-0015 metric records line by line, open the PNG, render the chapter, or re-run the E03 grid.

**Provenance.** Tests: `PYTHONPATH=src` with the project venv, `tests/test_source_inversion.py` + `tests/test_inversion_study.py`: 23 passed. The CSV tallies below are my own unregistered read-only computations over `e03-inversion.csv` (committed in f61c356), not run records. Quote them as "audit tally of the committed table", not as run metrics. I did not verify that the CSV equals the RUN-0015 metrics.

## 1. Truth separation

- **Strengths:** used only for scoring (`source_scores`, `field_rmse`, `crb_fraction`, the non-identifiable construction). The estimator receives `Gs`, `y`, `sigma`, `nuisance`. Confirmed in `e03_inversion.py`.
- **Not separated, by design (perfect model):** G is built from `sc.truth.source_xy` (known locations), `sc.truth.winds` (known wind) and the truth's own solver and spin-up. The estimator also gets the exact noise sigma (heteroscedastic levels are the generating sigma) and the exact background in the "known" mode. So all errors below are **optimistic floors** for a correctly specified model, not expected performance.
- **Noise reference uses truth:** `ref` is the rms true anomaly at all admissible sites, and the "prereg" sigma comes from `view.noise_sigma`, itself 5 % of median truth (gate-02 W6). Harmless for leakage; it means the `r=` levels are scenario-relative, not absolute.
- **Sensors and truth_id:** sensor sites come from the e02 helper `_design`; I did not check that it uses no truth.
- **Validation:** `G,F reproduce hidden truth anomaly` rel. error 3.8e-6 to 8.7e-6 over 10 scenarios (e03-validity). This confirms G is consistent with the truth field to about 1e-5, at the plume level. It is a consistency check against the same solver, not an independent oracle.

## 2. Nonnegative prior

- `scipy.optimize.nnls` on source coefficients; nuisance split `d = d+ − d−` with both ≥ 0 and unpenalised, so the background is sign-free. Correct by inspection and by the T-014 tests.
- Ridge: `lam = ridge · ‖WG‖₂²`, ridge = 1e-3 fixed a priori. It biases estimates toward 0 and is relative to the largest singular value, so it is negligible for columns far below that scale (the near-null directions that matter here).
- **Consequence of the prior:** 130 of 648 fits return all-zero coefficients (`n_zero_coef == K`). NNLS can sit on the boundary and report a "success" with an uninformative solution. `failure_mode` only reports `all_zero_estimate` when rank ≥ K, which never occurred, so those 130 are all labelled `non_identifiable`.

## 3. Optimisation convergence

- `solve_nnls` returns `residual_norm`, `rank`, `cond`. It returns **no convergence status, iteration count or KKT / dual-feasibility check**. scipy raises on `maxiter` in current versions; I did not confirm the installed version's behaviour, so "no non-convergence occurred" is **not established**, only "none raised".
- Observed `measurement_fit` (rms weighted residual, expected about 1 for a correct fit): range 0.77 to 1.69. 36 fits are below 0.9 (over-fit by near-null columns, expected when rank-deficient). 2 fits exceed 1.3: both `weighted_ridge`, r=0.01, K=4, 8 sensors, 192 rows (1.58 and 1.69). Ridge-induced bias is the likely cause; I did not test it. These are the only high-SNR, near-informative cells and they are the ones with a possibly under-converged or over-regularised fit. Flag for the human.
- The 1e27 errors are consistent with NNLS exploiting near-null columns (G column scales ~1e-5 against sigma). I did not verify the mechanism.

## 4. Relative metrics when the true source is zero

- `source_scores` raises `ValueError` if the true vector is all zero (checked: `[0,0]` raises). So `source_relative_rmse`, `parameter_bias` (÷ mean a) and the field ratio are **undefined, not zero**, for a no-source truth. Behaviour is documented in the test contract and safe, but there is **no scoring rule for a zero-strength truth or a zero-strength source `a_k = 0`** among K. With a mix (`[0,1]`) the scores are finite. Placement and detection tasks that include null scenarios need a decision: absolute metric (`source_rmse_ref`, which stays defined) as primary, or a reported "null scenario" label.
- `solve_nnls(G, 0)` returns zeros (checked), so a zero truth with zero noise is handled by the solver.
- The zero-estimate reference exists (`field_rmse_zero_estimate`). Against it, an audit tally of the 648 committed rows: source relative RMSE ≤ 1 (at least as good as predicting all zeros) in 279; worse than all-zero in 369. Field RMSE not worse than all-zero in 260 of 648. **More than half of the fits are worse than doing nothing.**

## 5. Error floors and failure modes (audit tally; weighted-ridge, background known, 27 cells per level)

| noise level | best source rel. RMSE | median | cells ≤ 1 |
|---|---|---|---|
| r=0.01 | 0.132 | 0.94 | 21 / 27 |
| r=0.1 | 0.133 | 0.94 | 22 / 27 |
| r=1 | 0.197 | 1.0 | 19 / 27 |
| prereg | 1.0 | 146 | 12 / 27 |

- Best fit in the whole table: K=4, 8 sensors, r=0.01, ridge, rel. RMSE 0.132 with rank 3 < 4. Even there the problem is rank-deficient; the floor is about 0.13 and is not attained under identifiability.
- Rank: `rank < K` in all 648. K=2 gives rank 1 in all 216 rows, including 8 sensors × 24 h at SNR 27, which is surprising (columns practically collinear, or most sensors seeing exact zero; the table has `snr_rms` = 0.0 rows). I did not probe why; this is the most important thing to understand before using E03 for placement.
- **Rank tolerance is relative to the largest singular value (1e-8), not to the noise.** A column 1e-9 times the leader may be informative at high SNR yet counted as null, and vice versa. Both rank and `crb_fraction` (0.0 when deficient) inherit this. Not validated against a noise-referenced criterion.
- Prereg level: SNR (rms signal / sigma) ≤ 2.7e-4 in every row. At the preregistered noise the data carry essentially no plume (consistent with F-002). Errors there measure noise amplification, not inversion quality.
- Non-identifiable case (K=4, 2 sensors, one snapshot, noiseless): rank 2; a nonnegative alternative (min coefficient 33 ug/s) has relative data misfit 1.6e-15 and relative field difference 2.6e4. The non-uniqueness half of the hypothesis is demonstrated. Noiseless NNLS picks a different member with relative error 0.81.
- Weighted vs unweighted vs ridge, and background known vs estimated: **0 paired full-rank cells**, so no comparison is supported (e03-compare). Background estimation error ranged −0.145 to +0.027 ug m⁻³ (tally).

## 6. Claims checked against records

| Claim (F-003 / T-015 result) | Status |
|---|---|
| All 648 fits rank-deficient at 1e-8 | Confirmed from CSV (648 non_identifiable). |
| 0 full-rank paired cells | Confirmed (rank < K everywhere). |
| Errors up to ~1e27 | Confirmed: max source relative RMSE 9.8e27. |
| Non-identifiable alternative fits to ~1e-15 | Confirmed from `e03-nonidentifiable.json` (1.56e-15). |
| Identifiable recovery "unresolved" | Consistent with the data. I would say "not demonstrated at this configuration", not "unresolved". |
| Run-record agreement (RUN-0015 metrics) | **Not checked.** |

## 7. Limitations

Synthetic, perfect-model, known wind, known locations, known noise, smoke scale (32², one 24 h window), fixed non-optimised sensors, 3 replicate seeds (noise shared across K, gate-02 W1), RIDGE fixed at 1e-3 with no tuning. Nothing here says anything about real PM2.5.

## 8. Decisions needed (human) and handoff

1. **Do not freeze this baseline yet.** For fair placement comparisons the estimator needs a regime where rank = K is achievable (longer window, optimised sensors, or a larger source scale). That is a change of design, so I did not make it.
2. Decide the zero-strength scoring rule (§4) and a noise-referenced rank criterion (§5) before they are used as gates.
3. Decide whether the ridge strength may be tuned on validation only.
4. Add convergence diagnostics (KKT residual, status) to `solve_nnls` and re-check the two `measurement_fit` > 1.3 cells.
5. Understand why K=2 has rank 1 with 8 sensors (§5) before running placement.
6. Independent promotion reviewer on a different agent profile is still required.

**Commits in scope:** cb2102c (T-013), db08ccc (T-014), f61c356 (T-015); base d053ff7. Downstream experiments need: `inverse.solve_nnls` and `rank_info`, `sensitivity.scenario_response_matrix`, `inversion_study.*`, plus the changes in item 1 to 5 above.
