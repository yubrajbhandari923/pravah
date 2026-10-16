# PRAVAH — Final Scientific Review, Independent Audit, and Research Closeout

**Date:** 2026-10-15  
**Checkpoint:** CP-016  
**Status:** Complete Research Closeout & Scientific Audit Record  

---

## 1. Executive Research Verdict

**PRAVAH** is a mathematically rigorous, computationally verified finite-volume regional atmospheric inversion and sensor-network design testbed. Its core forward solver, response-cache superposition, and nonnegative total-variation (ADMM/BVLS) inversion solvers are verified to machine precision ($10^{-15}$) on synthetic test cases.

However, **PRAVAH is strictly an idealized synthetic computational testbed, not an empirical model of air quality in Belauri, Nepal.** All emissions, concentration fields, sensor observations, and transport episodes are synthetic. The real geographic boundary and WorldPop raster serve as spatial inputs only; no empirical concentration measurements exist in the repository, and no physical validation has taken place.

Crucially, the project's own controls, independent reviews, and sensitivity audits **repeatedly overturned or narrowed initial intuitive hypotheses**:
1. **The value of measurement history is an emission-change tracking effect, not atmospheric plume carry-over:** The causal time-varying estimator beats the causal static fit (by 54–79%) because source emissions change over time; atmospheric plume carry-over was never isolated or tested. Furthermore, the steady-wind negative control would fire again and had to be retired.
2. **Temporal regularisation (TV) provides modest, wind-concentrated gains that reverse under model error:** TV regularization yields a 16–20% RMSE reduction over independent bins at 4 sensors, but this effect fails to generalize across the wind population (one wind, `rotation_a`, supplies 48–68% of the benefit). The gain vanishes at 9 sensors and *reverses* into a penalty under 10–20° of wind-direction error at Belauri scale.
3. **Surrogate-designed networks beat random placement, but fail to beat static designs or simple heuristics:** While surrogate-optimized layouts beat random sensor placement across scales (23–84%), they show **no resolved advantage over a static-model optimal design** at Belauri scale, and fail to beat a simple "go where people live" (`high_population`) heuristic in the toy setting at budgets $k=3, 5$.
4. **Local vs. transboundary attribution is limited by non-detection, not misattribution:** Where a change is detected, group attribution is accurate (pooled error $\le 2\%$ at distinguishability $d=16$). The real limitation of a 4-sensor network is *non-detection* under unfavorable wind-sensor geometries.
5. **The 500 m / dt 100 mesh remains provisional:** While ranking correlation is preserved ($\rho \approx 0.94\text{--}0.96$), the registered percentile-bootstrap test labels 500 m INADEQUATE ($E_{\text{lo95}} > 0.05$). While this verdict is borderline and interval-dependent (basic and BCa intervals indicate INCONCLUSIVE), 500 m shape error sits around 4–12%—an error magnitude comparable to the sensor-design effect sizes themselves.
6. **Population-weighted design at Belauri scale is statistically unresolved:** In EXP-026, population-weighted design yielded 0 beneficial cells for exact/best-found layouts and 24 for greedy layouts; with 12 test cases, statistical power detects only ~33–55% of baseline RMSE, rendering the result inconclusive.

The repository stands as a clean, reproducible, and candidly documented methodological case study on the limits of proxy-driven sensor design and regional source inversion.

---

## 2. Finding Review Ledger

This ledger records the formal auditor review across all 14 corrected dynamic findings (F-051 through F-077) and the earlier Phase 3 findings (F-029 through F-035).

```
========================================================================================================================
FINDING AUDIT LEDGER · PRAVAH FINAL SCIENTIFIC CLOSEOUT
========================================================================================================================
```

### [F-051] DYN-00 Numerical Verification
- **Experiment / Run:** EXP-028 / RUN-0091 (`scripts/dyn00_verify.py`).
- **Review Outcome:** **SUPPORTED (High Confidence).** (Independently reviewed by `scientific-reviewer` in predecessor pass; verified here).
- **Audit Verification:** All 25 pre-set checks pass. Relative mass closure residuals are $\le 4.4 \times 10^{-15}$ across 5 dynamic cases. Linear superposition of bin-expanded response functions matches direct time-varying simulation to $1.2 \times 10^{-15}$. Temporal convergence order on closed box is 0.997 (consistent with implicit Euler). Inflow-boundary face reclassification switches exactly to zero supply upon wind reversal ($3.36 \times 10^{-2}$ to $0.0$).
- **Corrections Applied / Status:** Earlier vacuous check (`west_no_import_after_reversal`) was replaced with the non-vacuous `basis_face_reclassification`.
- **Final Scientific Conclusion:** The numerical transport core and dynamic linear superposition machinery are verified and mathematically sound on the toy box.
- **Remaining Qualifications:** Checks are restricted to a $16 \times 12$ grid; spatial order is checked via monotone convergence only; Van Leer limiter and Belauri-scale dynamics are not covered.

---

### [F-055] DYN-02 v3 Historical Source Inversion (M0, M1, M2)
- **Experiment / Run:** EXP-030 / RUN-0087 (`pravah.studies.toy.dynamic_study`, test stage).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Numbers reproduce from `.research/runs/RUN-0087/episodes.csv`. At 4 sensors, paired Bonferroni 98.75% CIs for $\sqrt{J_{\text{pop}}}$ show M2 beats fair baseline M1 by $-15.7\%$ (2% noise: $-0.00051$ [$-0.00107, -0.00009$] $\mu\text{g}/\text{m}^3$) and $-20.1\%$ (10% noise: $-0.00067$ [$-0.00122, -0.00022$]). At 9 sensors, gain drops to $0.7\%\text{--}2.4\%$ (effect criterion not met).
- **Corrections Applied / Status:** Replaced F-054. The headline claim of "robustness across winds" was overturned: wind `rotation_a` supplies 68% (2% noise) and 48% (10% noise) of the entire pooled effect. A $t$-interval across the 6 wind means spans zero ($p = 0.195$, [$-0.00180, +0.00079$]).
- **Final Scientific Conclusion:** TV regularization provides a modest benefit when sensors are scarce (4 sensors) under rotating/oblique winds, but **fails to generalize across the population of winds**. At 9 sensors, M1 is already within 9–17% of the projection floor, leaving negligible margin for regularisation.
- **Remaining Qualifications:** 9 of 16 hyperparameter selections sit on flat grid plateaus; 11 non-converged validation fits occurred on suboptimal candidate points (test fits all converged); synthetic matched model.

---

### [F-056] DYN-07 Robustness to Model Error
- **Experiment / Run:** EXP-035 / RUN-0088 (`pravah.studies.toy.dynamic_robustness`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Wind rotation of 5°/10°/20° multiplies 4-sensor M1 error by $\times 1.84 / 2.59 / 3.74$ and 9-sensor M1 error by $\times 2.72 / 4.51 / 7.33$. The M2 advantage over M1 at 4 sensors erodes from $-20\%$ (nominal) to $-3\%$ (at 20°). M2 beats static M0 in 91–96% (4 sensors) and 61–99% (9 sensors) of episodes.
- **Corrections Applied / Status:** Replaced F-049. Withdrew the claim that "extra sensors tighten the fit to a wrong model." The unmodeled source at $(6000, 4500)\,\text{m}$ was located just 354 m from the center sensor of the 9-sensor layout (within its 600 m Gaussian plume), creating a severe local geometric artifact. In absolute error, 9 sensors outperform 4 sensors up to 10° of wind error.
- **Final Scientific Conclusion:** Wind and diffusion errors increase reconstruction error by factors of 1.7–9. Beyond 15–20° of wind error, additional sensors no longer provide an absolute advantage.
- **Remaining Qualifications:** Global rotations/scalings only; hyperparameters were deliberately not retuned per condition.

---

### [F-057] EXP-027 Mesh Decision Stability on Padded Belauri Domain
- **Experiment / Run:** EXP-027 / RUN-0081 (`scripts/hpc_exp027.sh`).
- **Review Outcome:** **PRELIMINARY (Audited, Confirmed VOID).**
- **Audit Verification:** In all 4 cells, the deliberately coarse 2 km falsification control passed the stability test ($p = 0.69\text{--}1.00$). Median layout displacements: 500 m vs. 250 m reference was 1118–2738 m; 2 km control was 2262–3234 m; independent replicate pairs varied by 3147–3996 m. Control regret medians were 10–25% of pair medians.
- **Corrections Applied / Status:** Replaced F-048. Endorsed the registered VOID verdict. Corrected the compute cost model for an adequate re-run (190–375 core-hours, not 90–190).
- **Final Scientific Conclusion:** **Experiment is VOID by its registered stop rule.** Layouts from 6-case design draws vary by 3–4 km purely from case sampling noise, which completely swamps mesh degradation (2x or 8x). The test had zero power to detect mesh inadequacy.
- **Remaining Qualifications:** 500 m mesh remained only provisionally admissible; necessitated EXP-027b.

---

### [F-061] DYN-04 Dynamic Sensor Placement vs. Baselines
- **Experiment / Run:** EXP-032 / RUN-0090 (`pravah.studies.toy.dynamic_design_study`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Evaluated 40 contrasts. Under the strict family ($\alpha = 0.05/40$), 23 robust wins, 1 robust loss, and 11 ties are observed. Dynamic-optimal beats random by $-32\%\text{--}-84\%$ across all 8 cells, and beats `spread` at $k \ge 3$ by $-64\%\text{--}-72\%$. At $k=1$ (10% noise), spread beats dynamic-optimal by $+20\%$ (robust loss).
- **Corrections Applied / Status:** Replaced F-059. Resolved ambiguity between lenient ($0.05/4$) and strict ($0.05/40$) multiplicity families. Corrected greedy-vs-optimum search gap description (greedy is $3.1\times$ exhaustive optimum at $k=3$, and $5.2\times$ best swap at $k=5$).
- **Final Scientific Conclusion:** Registered hypothesis H3 is supported against random layouts and against spread layouts at $k \ge 3$, but **is contradicted against the `high_population` heuristic** (which ties at $k=3$ and is marginally better at $k=5$).
- **Remaining Qualifications:** Hyperparameters were tuned on the comparator (`spread`), introducing potential tuning bias; surrogate ranks poorly at $k=1$ (Spearman $\rho = 0.43$, $p = 0.11$).

---

### [F-062] DYN-05 Design-Ensemble Sufficiency
- **Experiment / Run:** EXP-033 / RUN-0078, RUN-0080 (`pravah.studies.toy.dynamic_sufficiency`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** For the closed-form surrogate, 90th percentile relative regret $\le 5\%$ is met from $N=24$ (noise 0.10, $k=2$) and $N=48$ (noise 0.02, $k=3$), but **fails to hold up to $N=192$** in the other two cells ($p_{90}$ reaches 23–47%). For the real constrained estimator, two-stage CIs comparing $N=6$ vs. $N=192$ span zero in all 4 cells.
- **Corrections Applied / Status:** Replaced F-058. Disclosed that reference layout $S^*_A$ was noisy and suboptimal (+31% worse than the best layout at noise 0.10, $k=3$).
- **Final Scientific Conclusion:** Hypothesis H4 is contradicted in 2 of 4 cells for the surrogate. For the real constrained estimator, more design episodes are **not shown to help**. An unconstrained Gaussian surrogate cannot certify statistical sufficiency for the constrained TV estimator.
- **Remaining Qualifications:** Wind sampling was unstratified; reference layout noise inflates regret estimates.

---

### [F-063] DYN-06 v4 Local vs. Transboundary Attribution
- **Experiment / Run:** EXP-034 / RUN-0093 (`pravah.studies.toy.dynamic_separability_threshold`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Under null-calibrated detection (one-sided binomial $p < 0.01$ vs. 0.10 false alarm), pooled wrong-group attribution at distinguishability $d=16$ with 4 sensors is $\le 2\%$ ($0.01$ for S1, $0.02$ for S2, $0.00$ for S4, $0.01$ for S5, $0.01$ for S6). However, mean detection at $d=16$ with 4 sensors is only $0.58\text{--}0.94$ (S1 detected in only 4 of 6 winds). P3 (misattribution $\ge 25\%$ at $d=2$) is contradicted where evaluable (0.05–0.11).
- **Corrections Applied / Status:** Replaced F-060. Overturned the previous headline claim of 17–26% misattribution, which was caused by unweighted averaging over winds with detection at the false-alarm floor. Disclosed that the flip of P2 to "holds" resulted from a post-hoc change in scoring metric.
- **Final Scientific Conclusion:** Where emission changes are detected, local vs. boundary attribution is reliable ($\le 2\%$ error at $d=16$). The primary limitation of a sparse network is **non-detection**, not misattribution.
- **Remaining Qualifications:** Structural non-identifiability occurs when boundary faces become outflow faces (S5 signature is identically zero in 48 cells); no confidence intervals were stored on confusion rates.

---

### [F-065] DYN-03 v2 Value of Measurement History
- **Experiment / Run:** EXP-031 / RUN-0094 (`pravah.studies.toy.dynamic_history`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** CAUSAL outperforms CT by $-54\%\text{--}-57\%$ (4 sensors) and $-77\%\text{--}-79\%$ (9 sensors). Missed change detection drops from 61% (CT) to 9.7% (CAUSAL) at 4 sensors. On constant-emission episodes, CAUSAL gain is zero ($-0.00002$), and CAUSAL is worse than CT on activation episodes ($+0.00124$).
- **Corrections Applied / Status:** Replaced F-064. Proved mathematically and numerically that at tuned $\alpha \ge 10^4\text{--}10^9$, **CT equals the causal running static fit**. Thus, CAUSAL vs. CT evaluates time-varying vs. time-invariant models, not "history per se." Atmospheric plume carry-over was never tested. The steady-wind negative control would fire again (all 8 intervals exclude zero) and was retired under D-024.
- **Final Scientific Conclusion:** A causal time-varying estimator substantially outperforms a causal running static fit because emission rates vary over time. The mechanism of gain is emission tracking, not plume persistence.
- **Remaining Qualifications:** The original registered negative control would fail; wind-variation relative gain prediction is undetermined (intervals include zero); matched model.

---

### [F-069] DYN-08 v1 Belauri Synthetic Deployment (RUN-0099)
- **Experiment / Run:** EXP-036 / RUN-0099 (`scripts/hpc_dyn08.sh`).
- **Review Outcome:** **PRELIMINARY (Audited, Confirmed STOPPED).**
- **Audit Verification:** RUN-0099 terminated at registered stop rules 4.5(a) and 4.5(b). Control (a) (`lowpop` losing to `spread`) failed in 3 of 4 cells. Control (b) (fixed random groups showing no difference) failed at $k=5$ ($-20\%$ and $-17.5\%$). Independent review proved both controls were flawed by construction: `lowpop` was not poor under the surrogate, and fixed-layout null controls with episode-only bootstrap fail by construction. Bin 0 in CT/CAUSAL was left unestimated (zero).
- **Corrections Applied / Status:** Preserved as a registered negative outcome; descriptive results only.
- **Final Scientific Conclusion:** No confirmatory H8 verdict was issued. Led to post-hoc-amended protocol D-029 and rerun RUN-0101.
- **Remaining Qualifications:** Results are purely descriptive.

---

### [F-070, F-071, F-072] EXP-026 Population-Weighted Design at Belauri Scale
- **Experiment / Run:** EXP-016v / RUN-0098 (`scripts/hpc_exp026.sh`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:**
  - **F-070 (H1 Null & Optimiser Dependence):** Exact/best-found layouts show 0 beneficial cells and 5 harmful cells out of 200 (representing only 2 distinct layout comparisons against a single shared area reference layout ranking 2nd–3rd of 50). Greedy vs. greedy yields 24 beneficial cells. 12 test cases provide low statistical power (detectable effect is ~33–55% of baseline RMSE).
  - **F-071 (WorldPop Transfer):** WorldPop fixture is a joint null (0 beneficial cells; 1 harmful cell sharing the same area reference layout). Cross-fixture ratios show synthetic layouts on WorldPop are 1.02–1.31x baseline.
  - **F-072 (Greedy Gaps & Optimism):** Greedy design objectives sit 28–79% above best-found layouts. Half-split design optimism reaches 175–531%. All draws shared the exact same 6 design cases (seed 101), so replicate spread across design draws was never measured.
- **Corrections Applied / Status:** Replaced F-066, F-067, F-068. Overturned the claim that "population weighting does not work"; the true outcome is **absence of evidence due to low power and shared reference layouts**.
- **Final Scientific Conclusion:** Population-weighted design benefits at Belauri scale are unresolved and optimiser-dependent. The null outcome is explained by low test power and greedy sub-optimality.
- **Remaining Qualifications:** 12 test cases; 6 shared design cases; Phase 2 regression check was skipped under D-028.

---

### [F-075] EXP-027b Common-Case Mesh Adequacy
- **Experiment / Run:** EXP-037 / RUN-0096 (stage 1), RUN-0100 (stage 2) (`scripts/hpc_exp027b.sh`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** In stage 2 (72 cases across 3 draws), the 2 km control passed its falsification gate in 4 of 4 cells ($E_{\text{lo95}} = 0.18\text{--}0.27 > 0.05$). For the 500 m mesh, the registered percentile bootstrap interval yields INADEQUATE in 2 of 4 cells (noise 0.05, $k=3$: $E = 0.0873$ [$0.0595, 0.1881$]; noise 0.05, $k=5$: $E = 0.0919$ [$0.0516, 0.1948$]).
- **Corrections Applied / Status:** Replaced F-073. Identified that the percentile interval on a median-absolute statistic is upwardly biased. Cell 0.05/5 cleared the 0.05 threshold by only 0.0016 (within bootstrap Monte Carlo noise). Layout rankings are largely preserved ($\rho = 0.94\text{--}0.96$).
- **Final Scientific Conclusion:** Under the registered rule, the 500 m mesh is **INADEQUATE (borderline, rule-based)**. Shape error is approximately 5–10%. Decision equivalence is neither confirmed nor excluded. The 500 m / dt 100 mesh remains provisional.
- **Remaining Qualifications:** Evaluated on scale-free shape error, not direct decision regret; menu selection was in-sample.

---

### [F-076] DYN-08 v2 Belauri Synthetic Deployment Rerun
- **Experiment / Run:** EXP-036 / RUN-0101 (`scripts/hpc_dyn08.sh all 16 configs/dyn08_v2.json`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Protocol D-029 controls passed: control (a') (spread vs. worst-surrogate layout) met criteria in 4 of 4 cells ($-27\%\text{--}-63\%$); control (b') (two-level bootstrap on fresh random layouts) showed no effect in any cell. In primary contrasts, dynamic-optimal beats random layouts in 4 of 4 cells ($-23.3\%\text{--}-31.5\%$). However, against static-optimal layouts, dynamic-optimal is **completely indistinguishable** ($-0.8\%, -5.9\%, +0.1\%, -0.4\%$, all non-significant). Under 10° and 20° wind error, M2 error increases $\times 3.6\text{--}13$, and M2's advantage over M1 *reverses* in 4/8 and 6/8 rows.
- **Corrections Applied / Status:** Replaced F-074. Clarified that v2 is a post-hoc-amended rerun reusing winds and layouts from v1; H8 is supported under the D-026 counting rule only with post-hoc amendments. Overturned the claim of dynamic-specific design benefits.
- **Final Scientific Conclusion:** Surrogate-designed layouts outperform random placement at Belauri scale, but **provide no advantage over a static-model design**. Wind-direction error destroys regularisation advantages.
- **Remaining Qualifications:** Not an independent replication of v1; mesh250 error inflation ($\times 1.26\text{--}2.54$) is comparable to design effect sizes (24–32%); synthetic meteorology and emissions.

---

### [F-077] EXP-027b Post-Hoc Sensitivity Analysis
- **Experiment / Run:** EXP-037 / RUN-0102 (`scripts/exp027b_sensitivity.py`).
- **Review Outcome:** **PRELIMINARY (Audited, Wording Verified).**
- **Audit Verification:** Across 20 bootstrap seeds, the percentile interval consistently calls INADEQUATE. However, basic (reflected) and BCa intervals at $B=10000$ yield **INCONCLUSIVE** for both noise-0.05 cells (lower bounds $-0.0125$ to $+0.0408$). Out-of-sample cross-fit menus yield mean $E = 0.0997\text{--}0.1161$, while noise-debiased split-half estimates are $0.038\text{--}0.044$ (at or below the 0.05 threshold).
- **Corrections Applied / Status:** Authored as an exploratory post-hoc audit to qualify F-075.
- **Final Scientific Conclusion:** The INADEQUATE label is interval-dependent and driven by percentile interval bias on skewed statistics. True 500 m shape error sits between 4% (debiased) and 10–12% (out-of-sample). 500 m / dt 100 stays provisional.
- **Remaining Qualifications:** Exploratory post-hoc analysis; multiplicity unadjusted.

---

### Earlier Phase 3 Findings (F-029 through F-035)
- **F-029 (Toy Wind Error):** Wind direction error of 10–30° increases reconstruction error $3\text{--}4\times$, exceeding the sensor-count effect. *(Preliminary, unreviewed).*
- **F-030 & F-032 (Greedy Optimization Gaps):** Greedy forward selection is far from optimal at 2–4 sensors; greedy objective gaps reach $28\text{--}79\%$. *(Preliminary, unreviewed).*
- **F-031 (Domain Convergence):** For $T = 12\,\text{h}$, an outer half-width buffer of $\sim |\mathbf{u}|T/2$ is required for boundary convergence. *(Preliminary, unreviewed).*
- **F-034 (Wind Error Controls):** Direction error response survives alpha variation and off-member winds. *(Preliminary, unreviewed).*
- **F-035 (EXP-015 Stage 2):** H1 was untested due to incommensurable baselines between mesh comparisons (same draws) and resampling baselines (different draws). 500 m was provisionally admissible on narrow grounds. *(Reviewed by `scientific-reviewer` with verdict `needs_work`; superseded by EXP-027 and EXP-027b).*

---

## 3. Report and Artifact Audit

### Audited Documents
1. **`docs/FINAL_RESEARCH_REPORT.md`**
   - **Review Status Note Updated:** Line 3 updated to reflect the closeout audit (CP-016), noting that while first-round reviews were incorporated and all outputs reproduced, the closeout is a skeptical self-audit, and findings retain *preliminary* status.
   - **Discrepancy Ledger (§9.6):** Item 1 updated to document the reconciliation of `figure_index.json` (adding the 3 missing EXP-026 figures and removing obsolete pending placeholders).
   - **K_eff/K Clarification:** Re-verified that the 8.3 ratio at dt 100 s in F-025 stems from peak/diagonal wind conditions ($u \approx 2.2\text{--}2.5\,\text{m}/\text{s}$), whereas the nominal 1.5 m/s wind yields 4.9.
2. **`docs/PEER_PRESENTATION.md`**
   - **Review Status Note Updated:** Line 178 updated to record the CP-016 closeout audit and explicit retention of preliminary status.
   - **Slide 11 & Slide 12:** Audited for alignment with corrected findings (DYN-03 CT running static fit, DYN-06 pooled scoring, EXP-027b interval sensitivity, and DYN-08 v2 post-hoc labeling).
3. **`docs/figures/final/figure_index.json`**
   - **Reconciled:** Added all 3 missing EXP-026 figure entries (`exp026_fig1_pop_rmse_effect`, `exp026_fig2_tradeoff_plane`, `exp026_fig3_greedy_gap_optimism`).
   - **Cleaned:** Removed obsolete `"pending"` placeholders for `"DYN-08"`, `"EXP-027b"`, and `"Q-013"`.
   - **Verified:** All 20 referenced PNG files exist on disk, are properly labeled, and correspond exactly to the registered runs.

---

## 4. Mathematical and Computational Validity

```
+-----------------------------------------------------------------------------------------------------------------------+
| COMPONENT                  | STATUS                  | SCIENTIFIC ASSESSMENT                                          |
+-----------------------------------------------------------------------------------------------------------------------+
| Governing Transport PDE    | Verified Numerically    | 2D depth-averaged linear transport with lumped decay. Mass     |
|                            |                         | budget closes to 4.4e-15; superposition verified to 1.6e-15.   |
|                            |                         | Physical mixing height H and K are constant.                   |
+-----------------------------------------------------------------------------------------------------------------------+
| Discretization Schemes     | Verified (Baseline);    | Exponential convection + implicit Euler verified (order 0.997).|
|                            | Unverified (Van Leer)   | Van Leer flux limiter is unverified in dynamic inversion.      |
|                            |                         | Artificial diffusion (K_eff/K ~ 2.4-9.4) dominates nominal K.  |
+-----------------------------------------------------------------------------------------------------------------------+
| Nonnegative Inverse Solvers| Verified                | BVLS (beta=0) exact; ADMM (beta>0) verified against trust-QP   |
| (M0, M1, M2)               |                         | to 1.3e-6. Strict convergence tolerance (1e-7) prevents false |
|                            |                         | convergence.                                                   |
+-----------------------------------------------------------------------------------------------------------------------+
| History Estimators         | Verified (Causal);      | In v2, CT equals running static fit. CAUSAL filter is causal.  |
| (CT, CAUSAL, RETRO)        | Mechanism Re-interpreted| Gain comes from emission tracking; plume carry-over untested.  |
+-----------------------------------------------------------------------------------------------------------------------+
| Gaussian Bayes Surrogate   | Exercised / Limited     | Exact closed form for unconstrained linear Bayes, but ignores  |
|                            |                         | nonnegativity and TV; cannot certify sufficiency for M2.       |
+-----------------------------------------------------------------------------------------------------------------------+
| Physical Realism (Belauri) | Physically Unvalidated  | Flat terrain, spatially uniform ERA5 winds, synthetic constant |
|                            |                         | emission cycles, no empirical air quality measurements.        |
+-----------------------------------------------------------------------------------------------------------------------+
```

---

## 5. Hypothesis Ledger

- **Supported (Confirmatory / Numerically Verified):**
  - **DYN-00 (EXP-028):** Forward dynamic numerical conservation, superposition, and inflow face switching verified ($25/25$ checks pass; F-051).
  - **Phase 1-2 Baselines:** Bit-identical solver routing (F-009); 3-source static recovery under matched conditions (F-002); greedy layouts beating spread at $k \ge 3$ in static toy (F-006).
- **Supported with Post-Hoc Amendments:**
  - **H8 (DYN-08 v2, EXP-036):** Supported under the D-026 counting rule with post-hoc amendments of D-029 (F-076). Surrogate-designed layouts beat random layouts ($-23\%\text{--}-32\%$), M2 beats M1, and error grows with wind perturbation. Confidence is low due to post-hoc control construction and wind/layout reuse.
- **Contradicted / Falsified:**
  - **DYN-03 Negative Control:** Steady-wind control would fire again (gain on steady winds is non-zero and overlaps variable winds; F-065).
  - **DYN-06 P3:** Misattribution at $d=2$ is $\ge 25\%$ contradicted where evaluable (measured at 5–11%; F-063).
  - **DYN-05 H4:** Monotonic stabilization of design objective with episode count fails in 2 of 4 cells for the surrogate; no stabilization detected for the real estimator (F-062).
  - **DYN-04 H3 vs. High-Population Heuristic:** Dynamic-optimal layouts fail to beat the `high_population` heuristic at $k=3, 5$ in the toy domain (F-061).
  - **DYN-08 v2 Sub-prediction:** Margin over high-population heuristic was predicted to shrink at Belauri scale relative to the toy; instead, it expanded (F-076).
  - **DYN-08 v2 Model Error Claim:** The prediction that M2 retains an advantage over M1 under model error failed; M2's advantage reverses into a penalty under 10–20° wind error (F-076).
- **Inconclusive / Unresolved:**
  - **EXP-026 H1:** Population-weighted design reducing population-weighted RMSE at Belauri scale is unresolved and optimiser-dependent (0 beneficial cells for exact/best-found, 24 for greedy; low power at 12 cases; F-070).
  - **Q-013:** Identifiability of the sensor-design objective with 6 cases remains unclosed. Replicate-to-replicate spread across independent design draws at Belauri scale was never measured.
  - **DYN-02 Wind Generalizability:** TV regularisation gain fails to generalize across the population of winds ($t$-interval spans zero; F-055).
  - **EXP-027b Decision Equivalence:** Mesh shape error of 5–10% is detectable, but decision regret and layout ranking preservation ($\rho \approx 0.94\text{--}0.96$) neither prove nor rule out decision-relevant differences (F-075, F-077).
- **Void / Invalid:**
  - **EXP-027:** Falsification control passed because design-sample noise (3–4 km) swamped mesh differences; experiment declared void (F-057).
  - **DYN-08 v1:** Stopped under registered stop rules due to flawed control construction (F-069).
- **Untested:**
  - Atmospheric plume carry-over / memory mechanism in dynamic history.
  - Empirical validity for real-world Belauri air pollution.

---

## 6. Scientific Contributions

### Conservative Assessment of Novelty
None of the individual algorithmic components in PRAVAH are mathematically novel:
- Fused lasso / total variation regularisation with nonnegativity is established (Tibshirani et al. 2005; standard ADMM).
- Goal-oriented optimal experimental design (OED) with linear quantities of interest is established (Attia et al. 2018; Wu, Chen & Ghattas 2023).
- Population-weighted sensor design on steady transport was treated by Henriquez et al. (2015).
- Response-operator superposition and caching are standard linear system techniques.

### Defensible Scientific Contributions
1. **Separability Limits in Regional Source Inversion:** A rigorous, structure-aware demonstration that local vs. transboundary emission attribution fails through **non-detection**, not misattribution. When wind-sensor geometry captures an event, attribution error is $\le 2\%$. The failure modes (e.g. boundary influx becoming invisible when wind reverses) are rigorously cataloged (F-063).
2. **Failure of Unconstrained Proxies to Certify Constrained Design Sufficiency:** An empirical demonstration that an unconstrained Gaussian Bayes surrogate cannot certify sample-size sufficiency for a constrained, regularized (BVLS/ADMM) inverse estimator (F-062).
3. **Severe Vulnerability of Dynamic Regularization to Wind Error:** Evidence that the performance advantage of temporal TV regularisation is fragile: while beneficial under matched models, it erodes rapidly and **reverses into a performance penalty** under modest (10–20°) wind-direction errors (F-056, F-076).
4. **Methodological Case Study in Experimental Falsification:** A transparent research record demonstrating that registered falsification controls and independent reviews actively caught flawed baselines (DYN-03), invalid controls (EXP-027, DYN-08 v1), false-alarm-dominated metrics (DYN-06), and underpowered nulls (EXP-026).

---

## 7. Research OS Closeout

### Tasks and Questions Status
- **Tasks Closed as Blocked (Legitimate Scientific Blockers):**
  - `T-032` (PLAN-003): Formally marked **blocked**. Phase 3 findings F-029..F-034 never received external non-author review, and F-035 found H1 untested. The task acceptance criterion (*reviews recorded by non-author*) cannot be legitimately satisfied without external review.
  - `T-038` (PLAN-004): Formally marked **blocked**. The single Slurm bundle workflow (`phase3.sbatch`) was superseded by per-experiment cluster submissions. Final closeout synthesis is recorded in `docs/FINAL_RESEARCH_REPORT.md` and checkpoint `CP-016`.
- **Open Questions:**
  - `Q-013` (*Identifiability of design objective with 6 cases*): Retained explicitly as **open**; documented as partially addressed by F-062 and F-072, but unclosed at Belauri scale.
  - `Q-011`, `Q-021`, `Q-022`, `Q-024`: Documented as open / preliminary.
- **Syntheses Completed:**
  - `EXP-036` (DYN-08): Synthesized (`EXP-036/S2`) and marked completed.
  - `EXP-037` (EXP-027b): Synthesized (`EXP-037/S1`) and marked completed.
- **Checkpoints:**
  - `CP-016` created: *"CP-016 final scientific review, independent audit, and research closeout"*.

### Git Status
```
Changes staged/ready for commit:
	modified:   .research/context/current.md
	modified:   .research/events.jsonl
	modified:   .research/experiments/EXP-036.md
	modified:   .research/experiments/EXP-037.md
	modified:   .research/syntheses.jsonl
	modified:   .research/tasks/T-032.md
	modified:   .research/tasks/T-038.md
	modified:   docs/FINAL_RESEARCH_REPORT.md
	modified:   docs/PEER_PRESENTATION.md
	modified:   docs/figures/final/figure_index.json
	new file:   docs/SCIENTIFIC_CLOSEOUT_AUDIT.md
	new file:   .research/checkpoints/CP-016.md
```
*Zero changes to core solver physics (`pravah/forward.py`, `pravah/dynamic.py`), zero changes to regression baselines, zero unapproved pushes.*

---

## 8. Prioritized Future Experiments

If research resumes, the top three prioritized investigations are:

1. **Resolution of Q-013 via Stratified Design Ensembles:**
   - *Objective:* Directly measure the replicate-to-replicate sampling variance of the Belauri greedy sensor placement objective.
   - *Design:* Generate 16 independent design draws stratified across ERA5 wind sectors (rather than sharing seed-101 design cases), evaluate layout stability at budgets $k \in \{3, 5, 8\}$, and quantify objective variance $\sigma_J / J_{\text{ref}}$.
2. **250 m Belauri Inversion with 1 km Control:**
   - *Objective:* Resolve whether the 500 m / dt 100 mesh introduces material decision error in dynamic sensor placement.
   - *Design:* Execute dynamic inversion (M1/M2) directly on the 250 m reference mesh for a representative subset of episodes, and compare placement decisions against 500 m and 1 km designs under identical wind forcings.
3. **Empirical Air Quality Ingestion & Terrain-Coupled Transport:**
   - *Objective:* Break the synthetic inverse crime and evaluate real-world transfer.
   - *Design:* Ingest real monitoring station records (DHM/ICIMOD/US Embassy) and low-cost sensor time-series; replace spatially uniform wind fields with terrain-following diagnostic winds driven by the local DEM.

---

## 9. Reproducibility Reference Table

| Finding ID | Experiment | Slurm / Run ID | Frozen Config / Hparam Hash | Seeds | Primary Artifact / Results Location |
|---|---|---|---|---|---|
| **F-051** | EXP-028 | RUN-0091 | 25 pre-set checks | N/A | `.research/runs/RUN-0091/dyn00_checks.json` |
| **F-055** | EXP-030 | RUN-0087 | `results/dyn02v3/frozen_hparams.json` (`200ef2e894a62851`) | Test 7300–7395 | `.research/runs/RUN-0087/episodes.csv` |
| **F-056** | EXP-035 | RUN-0088 | `frozen_hparams.json` (`200ef2e894a62851`) | Test 8500–8595 | `.research/runs/RUN-0088/summary.json` |
| **F-057** | EXP-027 | RUN-0081 (Job 201981) | `configs/belauri_padded_geography` | Draws 1101–1108 | `results/exp027/paired.csv`, `paired_verdict.csv` |
| **F-061** | EXP-032 | RUN-0090 | Layouts `8bdef74d7c30fe2e`, Hparams `103b54317bac1be5` | Test 7800–7895 | `.research/runs/RUN-0090/summary.json` |
| **F-062** | EXP-033 | RUN-0078, RUN-0080 | `results/dyn05/design.json` (`e3968dd3e90b6902`) | Check 8400–8495 | `results/dyn05/`, `docs/figures/dynamic/tables/` |
| **F-063** | EXP-034 | RUN-0093 | Protocol D-025, `frozen_hparams.json` | Null 9500–9599, Sig 9600–9699 | `.research/runs/RUN-0093/summary.json` |
| **F-065** | EXP-031 | RUN-0094 | `results/dyn03v2/frozen_hparams.json` (`3c86b5e668181a3d`) | Test 8600–8695 | `.research/runs/RUN-0094/summary.json` |
| **F-069** | EXP-036 | RUN-0099 (Job 202529) | Layouts `a54267ac1fcf370b`, Hparams `7b179f28f24f32e1` | Test 43000–43191 | `results/dyn08-hpc/dyn08_all/summary.json` |
| **F-070..72**| EXP-026 | RUN-0098 (Job 202528) | `configs/population_belauri_hpc.json` (D-028) | Draws 6501–6506 | `results/exp026-hpc/cases.csv`, `layouts.csv` |
| **F-075** | EXP-037 | RUN-0096, RUN-0100 | `configs/convergence_paired_padded_exp027b.json` | Draws 1201–1203 | `results/exp027b-hpc/stage2/exp027b/` |
| **F-076** | EXP-036 | RUN-0101 (Job 202548) | Layouts `2bb1afbaecc2ce1a`, Hparams `123c6d12f5e12eaf` | Test 44000–44191 | `results/dyn08-v2-hpc/dyn08_all/summary.json` |
| **F-077** | EXP-037 | RUN-0102 | `scripts/exp027b_sensitivity.py` | 72 cases (RUN-0100) | `.research/runs/RUN-0102/sens_summary.json` |
