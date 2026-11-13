# Gate 04 — Independent audit of the uncertainty and identifiability claims (T-026, EXP-005)

**Verdict: PARTLY CLEARED, NOT a scientific approval.** The Fisher/CRB machinery (`pravah.information`) is correct for the one case where it should be exact (linear Gaussian, interior truth, correctly specified white noise) and is conservative under the NNLS boundary in the cases I ran. It is **badly miscalibrated when the noise model is wrong**, it **cannot be evaluated at all for the three of four E04 designs that matter for the hypothesis** (singular or near-singular M), and the E04 smoke result (F-004) says nothing about the EXP-005 hypothesis because it is one replicate at a noise level where nothing is above noise. The EXP-005 hypothesis (spectrum predicts recoverable source combinations better than sensor count or coverage) remains **unresolved**. Freezing any of this for placement work is a human decision (§8).

**Reviewer:** claude-balanced / claude-sonnet-5-5. The code under review (T-023 to T-025) was written by earlier tasks of the same model family, so independence is by task and re-reading only; an independent promotion reviewer on a different agent profile is still required and is not replaced by this document. I changed no repository code, frozen test, metric definition or claim. I registered one run (RUN-0018) with a script kept **outside** the repo (§1).

## 1. Provenance

| Item | Value |
|---|---|
| Run | RUN-0018 of EXP-005, exit 0, 51 s, local backend, host RAD-88300596 |
| Command | `env PRAVAH_ROOT=<worktree> python3 /tmp/audit/e04_audit_mc.py` (via `research run exec`) |
| Git | branch `exp/EXP-005-footprints-fisher-information-and-identi`, commit `c7610a1`, clean tree at run time |
| Seed / draws | seed 20261110, 2000 noise draws per cell |
| Scenario / dataset | E02 `changing4` layout, `steady_westerly`, replicate 2, 32², 24 h, perfect model (same truth as the E04 meta: `TRUTH-c6b40a19efc2`); informative sensor set from `e04-meta.json` (4 sensors, 96 rows, K = 4) |
| Noise | `sigma_info_scale` = 2.577e-6 ug m^-3 from the run record: the E04 *non-preregistered* scale (0.1 x rms reference signal), times multipliers 1, 100, 1000 |
| Artifact | `audit_mc.json` (A-0025), all metrics in the run record under `A.*`, `B.*`, `C.*` |
| Caveat | The script is **not in the repo** (the task's permitted output is this file only), so RUN-0018 is reproducible from the run's snapshot ref plus the script text, not from the commit alone. If this audit is kept, the script should be committed by a task that is allowed to. |

All numbers below are read from the RUN-0018 metric records / `audit_mc.json`. Numbers marked *(tally)* are my read-only reading of the committed E04 CSV (`e04-identifiability.csv`), not run metrics. Coverage values carry binomial noise: with 2000 draws a true 95 % has a 95 % interval of about 94.0 to 96.0 %.

## 2. What I read

`src/pravah/information.py`, `identifiability.py`, `inverse.py`, `scripts/e04_footprints.py`, `docs/research/footprints-identifiability.md`, `e04-meta.json`, `e04-identifiability.csv`, gate-03. I did not open the PNG, render the chapter, read RUN-0016/RUN-0017 records line by line, re-run `e04_footprints.py`, or run the unit tests in this audit. I did not read `tests/test_information.py` or `tests/test_identifiability.py` beyond their existence.

## 3. FIM local Gaussian vs empirical repeated-noise coverage (RUN-0018)

Question: does the 95 % ellipsoid `(a_hat - a)^T M (a_hat - a) <= chi2_0.95(K)` (and the 2-parameter marginal for sources 0 and 1, used in E04) contain the truth in about 95 % of repeated noise draws? Informative design, M = G^T R^-1 G with white sigma. Truth a is the E02 vector (smallest component 80 ug/s); at the base noise the CRB standard deviation is at most 1.03 ug/s (`crb_sd_max`), about 1.3 % of the smallest source.

**Unconstrained least squares (the case where the CRB is exact).** In every one of the 9 truth x noise cells, 4-parameter coverage was 0.9395 to 0.958 and 2-parameter coverage 0.942 to 0.958, all consistent with 95 %. The empirical-to-CRB variance ratio was 0.92 to 1.06. This is a validity check of the algebra and of `fisher`, not new science. It holds even when the truth sits on the boundary, because plain least squares ignores the constraint.

**NNLS (the project's actual estimator).** NNLS estimates were produced with `pravah.inverse.solve_nnls` with the correct sigma, no ridge, background known.

| noise multiplier | truth | fraction of estimates exactly 0 | 4-param coverage | 2-param coverage | variance ratio (min to max) |
|---|---|---|---|---|---|
| 1 | interior | 0.000 | 0.9465 | 0.9545 | 0.96 to 1.03 |
| 1 | source 0 at 1 CRB-sd | 0.042 | 0.958 | 0.967 | 0.74 to 1.00 |
| 1 | source 0 at 0 | 0.125 | 0.9645 | 0.9705 | 0.33 to 0.96 |
| 100 | interior | 0.098 | 0.9755 | 0.965 | 0.58 to 1.01 |
| 100 | source 0 at 1 CRB-sd | 0.214 | 0.982 | 0.9745 | 0.28 to 0.98 |
| 100 | source 0 at 0 | 0.210 | 0.9865 | 0.9835 | 0.27 to 0.93 |
| 1000 | interior | 0.498 | 0.9905 | 0.9925 | 0.24 to 0.35 |
| 1000 | source 0 at 1 CRB-sd | 0.508 | 0.9895 | 0.9915 | 0.23 to 0.35 |
| 1000 | source 0 at 0 | 0.508 | 0.9925 | 0.9945 | 0.21 to 0.34 |

Reading: the FIM ellipse is **exact in the interior regime and conservative (over-covers, 96 to 99 %) once the constraint is active**, because NNLS truncates the estimates and its empirical variance falls to 21 to 35 % of the CRB in the worst cells. So in these runs the ellipse never under-covered under the boundary; it reports more uncertainty than is present. That is the benign direction. It is not a proof: one truth vector, one design, only source 0 placed on the boundary, a well-conditioned M (cond 3.17). A design in which several components are near zero, or in which the constraint and a weak mode interact, was not tested. The ellipse is also **not a calibrated confidence region for NNLS**: at multiplier 1000 about half of all coefficients are exactly 0 and the nominal 95 % ellipse covers 99 %, so it cannot be inverted into a statement about a_k > 0.

## 4. Misspecified covariance (the failure case)

Estimator and FIM assume white noise with sigma; the data are drawn differently. Informative design, interior truth, unconstrained least squares.

| generating noise | 4-param coverage | 2-param coverage | empirical / assumed variance |
|---|---|---|---|
| white, 0.5 x assumed sigma | 1.000 | 1.000 | 0.24 to 0.27 |
| white, 2 x assumed sigma | 0.329 | 0.540 | 3.95 to 4.04 |
| AR(1) in time, rho = 0.8, assumed sigma | 0.1395 | 0.3315 | 7.07 to 7.50 |

A factor-of-two error in sigma drops nominal 95 % coverage to 33 %, and ignoring a plausible temporal correlation (rho = 0.8 per sensor, as in T-023) drops it to 14 %, with the variance understated about 7 times. The estimate itself is only mildly degraded (relative parameter RMSE 0.016 vs 0.0058 white), which is exactly the danger: the point estimate looks fine and the stated uncertainty is wrong by a large factor. Any place the project reports a CRB ellipse as uncertainty has to carry the noise-model assumption with it; the preregistered sigma is a point value (5 % of the median truth, F-002/gate-02), not a validated covariance. Cross-sensor correlation was not tested at all. T-023 §5 already showed that assuming white noise over-states log det information by 3.82 for AR(1) at rho = 0.8 on the `separated2` example; the audit confirms the same direction in coverage terms and quantifies it.

## 5. Parameter information versus field skill

The Fisher matrix and ellipses are statements about **source strengths a**. The project's primary endpoint and several secondary ones are about **fields**. In RUN-0018 I scored the field error as the rms of `G_all (a_hat - a)` over all admissible sites and all 24 hours, divided by the rms true anomaly.

- **For the well-conditioned design the two move together.** Relative parameter RMSE vs relative field RMSE (unconstrained LS, interior truth): 0.0058 vs 0.0043 at multiplier 1, 0.584 vs 0.438 at 100, 5.94 vs 4.40 at 1000. Field error is about 0.74 x the parameter error in all three. This is expected when cond(M) = 3.2 (no weak mode exists), so it demonstrates nothing about the cases that motivate the hypothesis.
- **NNLS reduces error without changing the ellipse.** At multiplier 1000, NNLS relative parameter RMSE is 3.37 and field RMSE 2.77 against 5.94 and 4.40 for LS, yet the nominal ellipse is the same. Parameter-space coverage (99 %) and field skill (error 2.8 times the signal) are different quantities: an estimator can have a "valid" covering ellipse while the field it implies is worse than predicting zero (relative field error > 1). At multiplier 1000 the field error exceeds the signal for both estimators.
- **Not tested here, and the crux of the hypothesis:** the near-null direction. E04 shows a source vector moved 50 % of ||a|| along the weakest singular mode changes the 4-sensor data by < 1 sigma in 10 of 12 cases *(tally from the CSV)*. Parameter information is then near zero, but the **field** change in unobserved sites and hours may be large or small; I did not compute it. A weakly identified source combination is not automatically a bad field reconstruction, and the reverse can also hold (a field can be accurate at the sensors and wrong between them). The E04 numbers are parameter-space diagnostics and should not be read as field-skill predictions without that computation.

## 6. Numerical degeneracy

1. **The 1e-8 rule is applied on different scales in two places.** `rank_info` and `rank_coherence` threshold singular values of G at `1e-8 s_max`. `criteria` thresholds eigenvalues of M = G^T G at `1e-8 w_max`, which corresponds to a singular-value ratio of 1e-4. The same matrix is therefore called full rank by one function and singular by the other. Evidence in RUN-0018: for the E04 "coverage" and "redundant" designs `criteria` reported rank 2 (`C.coverage.m1.float_rank` = 2, `C.redundant.m*.float_rank` = 2) and `defined = False`, while the E04 CSV records `float_rank` 4 and 3 for those designs *(tally; singular-value rule)*. The coverage design has cond(M) = 4.66e9 (singular-value ratio about 1.5e-5): passes `rank_info`, fails `criteria`. This is a real inconsistency, not a numerical accident, and it determines which designs can be ranked by D/A/E. It is harmless where the code always adds a prior (E04 does), but any later placement code calling `criteria` on a raw M inherits it.
2. **Only one of the three E04 designs has a computable CRB at the non-preregistered noise level.** Informative: cond(M) 3.17. Coverage: 4.66e9. Redundant: 1.9e300 at one noise level and 2.4e16 at another (the 1e300 is a division by a clamp in my script, i.e. "singular", not a measured number). Coverage experiments for those designs are therefore **not possible without a prior**; the ellipses in the E04 figure for them are prior-dominated (the dashed curves lie on the prior circle, as the figure title itself says).
3. **A prior hides the degeneracy.** E04 reports `posterior_logdet` and `ellipse_area_over_prior` from M + P0 with prior sd = A_ref. At the preregistered sigma, `ellipse_area_over_prior` is 0.99999 or higher in all six cases *(tally)*: the data change nothing. The posterior is defined and well conditioned, but is not evidence of identifiability. Posterior log det values near -36.84 at the preregistered sigma are the prior alone.
4. **Practical rank is 0 of 4 at the preregistered sigma in all six wind/design cases** *(tally from the CSV, consistent with F-004)*. At that noise there is no regime in which the FIM-vs-coverage comparison above means anything for the real benchmark: I used the non-preregistered scale, with 100x and 1000x multipliers, to get any signal. Results in §3 and §4 are therefore **method checks at a SNR the benchmark does not provide**, not performance statements. At multiplier 1000 the ratio of CRB sd to the smallest source is about 13, i.e. the preregistered regime (sigma about 1e5 times the base) would be far deeper into the constrained regime than anything I ran.
5. **Rank claims depend on the tolerance.** T-023 already notes `separated2` n = 8 sits at cond 2.7e7, so a tolerance of 1e-7 gives rank 1. The practical-rank rule `s * A_ref / sigma >= 1` is noise-referenced and cures this for rank, but A_ref (100 ug/s) is a free choice that sets the threshold; it is not a preregistered quantity that I could find.

## 7. Claims checked against records

| Claim | Status |
|---|---|
| FIM ellipse is exact for unconstrained LS, interior truth, correct white noise | Supported by RUN-0018 (coverage 0.94 to 0.96, variance ratio 0.92 to 1.06). Algebraic check, one design. |
| FIM ellipse is valid for NNLS | **Not supported as a nominal 95 % region.** Observed 96 to 99 % (conservative) in the cases run; no under-coverage seen; not shown in general. |
| FIM ellipse is robust to noise-model error | **Contradicted** (coverage 14 % and 33 % under rho = 0.8 AR(1) and 2x sigma). |
| F-004: practical rank 0/4 at the preregistered sigma, float rank 3 to 4 | Consistent with the CSV *(tally)*; but `criteria` gives rank 2 for two of those designs (§6.1). The "float rank" in F-004 is the singular-value rule only. |
| F-004: weak-mode counterexample moves data < 1 sigma in 10 of 12 cases | Consistent with the CSV *(tally)*; one replicate, one scenario; says nothing about field error. |
| EXP-005 hypothesis (spectrum beats sensor count or coverage) | **Unresolved.** No recovery error was computed against spectral prediction in E04 or here; designs are three hand-built sets on one replicate. |
| Green D-optimality 1-D comparator reproduced | Not reproduced (T-023: the PDF has no extractable text). A closed-form exponential-profile Jacobian is only a pedagogical stand-in. |

## 8. Required human decisions and recommendations (none made by me)

1. **Noise model.** Decide whether the benchmark's noise covariance is an assumption or a quantity to estimate; if CRB ellipses are to be reported, add a misspecification sensitivity (sigma x0.5/x2, AR(1)) next to every ellipse. The numbers in §4 show why.
2. **Consistent rank tolerance.** Decide whether `criteria` should threshold on singular-value scale like `rank_info`, or whether both should be noise-referenced. Changing it alters tested behaviour, so it needs a task that may touch `information.py` and its frozen tests; I did not.
3. **Coverage claims for NNLS** should be stated as "conservative in the tested cases" until a profile-likelihood or bootstrap region is compared on boundary-heavy truths and on multi-source boundaries. Do not describe the CRB ellipse as a confidence region for the constrained estimate.
4. **Field skill.** Before E06 to E08 use the singular spectrum to choose sensors for a *field* objective, compute field error along the weakest modes (the E04 counterexample vectors are ready for this) and report parameter information and field skill separately.
5. **Replication.** E04 and this audit are one scenario and one replicate. A paired-seed repeat at the preregistered scale, or an agreed amplified source scale that is itself preregistered, is needed for any test of the hypothesis. A positive effect is not required for the experiment to pass.
6. **Script placement.** The Monte Carlo script lives in `/tmp/audit`; a task with file permission should commit it (or reject the finding) so RUN-0018 is reproducible from the repository.

## 9. Handoff for downstream experiments

- **Verified commits relevant to this experiment:** T-023 `bae5b4e` (derivation and smoke check), T-024 `82ed85f` (information.py), T-025 `c7610a1` (identifiability.py, e04 figure/table/chapter), on base `514b1a4` (T-022 rank_info). "Verified" means the task checks passed, not that the science was reviewed; F-004 is preliminary and no finding was reviewed.
- **Changes downstream work needs (not done here):** unify the rank tolerances (§6.1); add an explicit prior or a "singular" return in `criteria` for the near-singular case; commit the audit script or a reviewed equivalent; add a noise-misspecification option to any reported ellipse; synthesize the unsynthesized runs (EXP-005 now has 3, the project limit is 5 before synthesis).
- **Not promoted by me.** Promotion is by Research OS at the experiment boundary.
