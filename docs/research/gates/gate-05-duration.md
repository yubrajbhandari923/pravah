# Gate 05: independent audit of E05 (temporal observability: sensor count versus observation window)

**Verdict: NOT CLEARED for interpretation as written. The computation is valid; the chapter is not.** The E05 driver reproduces bit-for-bit from HEAD, the design matrices, look-ahead guard, nesting and seed pairing are correct, and practical rank ranks designs consistently with estimator error. But (i) the estimator blows up (relative source error up to 1e23) in the 5 % of cells with no signal, and every **mean**-based table and figure in `05-time.qmd` (matched budget, fixed-N window, Branch B, all three `e05-*.png`) is dominated by those blow-ups and is uninterpretable; (ii) of the lead's four readings, only the qualitative "a longer window raises practical rank" survives, and a spot check shows **that gain is entirely caused by the wind direction changing between 48 h blocks**: with the per-block direction offset removed, a 7-day "constant" window adds exactly zero practical rank over 24 h (24/24 paired cases). "Window length" and "wind diversity" are therefore the same factor in this design; (iii) the wind-diversity table's "contradicted" labels overstate a null, and a better-posed contrast (window gain under veering minus window gain under constant) favours the hypothesis (14 of 17 non-tied seeds, sign p = 0.013), post hoc. Everything at a_ref = 1 g/s is **exploratory** (the amendment and the tie rule were both chosen after seeing data). The E05 hypothesis is **unresolved, with a positive exploratory signal** for "wind change, not elapsed time, adds rank".

**Reviewer:** Opus 5.5 subagent, plan-02 `e05-duration` step 4. Same model family as the author, so independence is by task and re-derivation only. I changed no repository file.

## 1. Provenance

| Item | Value |
|---|---|
| Repository | `/home/yubraj/research/pravah-new`, main; code read at HEAD `8a47a27` (E05 data from `0569822`) |
| Runs audited | `2026-11-12-e05-A`, `2026-11-12-e05-B` in `reports/runs/runs.jsonl` (git `f204841`, 20 seeds, local, uv lock `8ddcaa8b…`) |
| Data | `reports/quarto/tables/e05-duration.csv` (10 080 rows), `e05-duration-{A,B}.csv`, oracles CSVs |
| Audit scripts (scratchpad, not in repo) | `audit_csv.py` (re-analysis of the CSV, output `audit_csv.out`); `audit_jitter.py` (recomputes branch-A scenarios via `e05_duration.run_scenario`; `AUDIT_PATCH=1` removes the N(0, 10°) direction offset of `steady_westerly` while keeping the rng stream and speed factor) → `audit_repro.csv`, `audit_nojitter.csv`; plus inline pandas one-offs whose numbers are quoted below |
| Commands | `cd /home/yubraj/research/pravah-new && uv run python <scratchpad>/audit_csv.py`; `AUDIT_SEEDS=0,1 uv run python <scratchpad>/audit_jitter.py`; `AUDIT_PATCH=1 AUDIT_SEEDS=0,…,7 uv run python <scratchpad>/audit_jitter.py` (OMP_NUM_THREADS=1, 8 workers, ~3 min) |
| Reproduction | constant wind, K = 4, seeds 0 and 1, 168 rows: rank, practical rank, s_max, src_rmse and field_rmse identical to the committed CSV (max relative difference 0.0) |
| Caveat | The audit scripts must be committed beside this file if the lead wants these numbers reproducible from the repo. |

## 2. What I read

`AGENTS.md` §2; plan-02 `e05-duration`; `docs/research/time-observability-design.md`; `scripts/e05_duration.py`, `scripts/e05_report.py`; `src/pravah/{temporal_basis,identifiability,sensitivity,benchmarks}.py`; relevant parts of `experiment_config.py`; `reports/quarto/chapters/05-time.qmd`; `state/experiments/e05.md`; `state/decisions/2026-11-12-source-scale.md`; `docs/research/gates/gate-04-identifiability.md` (style); the last runs in `runs.jsonl`. Not opened: the PNGs (judged from the code that draws them), `tests/test_time_observability.py`, `e02_examples.py` beyond `make_cfg`.

## 3. Findings

Severity: **High** = changes a stated conclusion or makes a reported number meaningless; **Medium** = design/confound that limits a claim; **Low** = presentation or latent bug.

### F1 (High). The window effect is a wind-direction effect

`make_winds` adds one direction offset N(0, 10°) per 48 h block, also to `steady_westerly`. Over the 168 h calendar the four block directions of "constant" span 1.6° to 36° (median ≈ 20°) across the 20 seeds. Spot check (`audit_nojitter.csv`, K = 4, seeds 0 to 7, N ∈ {2, 4, 8}, preregistered σ):

| m | gain in practical rank, W = 24 → 168 h, with the block offset (committed data) | same, offset removed |
|---|---|---|
| 4 | +0.167 mean, 5 up / 1 down / 18 tied | **0.000, 0 / 0 / 24** |
| 16 | +0.250 mean, 6 / 1 / 17 | **0.000, 0 / 0 / 24** |

Without direction change, practical rank is identical at W = 6, 24 and 168 h for fixed m. Speed-factor changes between blocks add nothing. So "longer windows raise rank" in E05 means "longer windows sample more wind directions"; elapsed time per se contributes nothing in this model (consistent with the design note's prediction §3.3.2). Any statement framed as duration compensating for sensors must be reworded as wind-direction coverage compensating for sensors. In the full data, the per-seed 24 → 168 h gain under "constant" correlates only weakly with that seed's drift range (Spearman 0.22–0.28, p > 0.2), so the dependence on drift magnitude is not established; the binary on/off result above is.

### F2 (High). Estimator blow-ups make every mean-based table and figure meaningless

`ridge_ls` sets λ = 1e-3·‖G/σ‖², i.e. relative to the design's own top singular value, not to a prior scale. When a design sees nothing (s_max·a_ref/σ < 1e-6 in 4.9 % of branch-A cells, mostly N = 2 sensors upwind of every source), λ is also ~0 in absolute terms and the error is ~noise/(√λ) → relative source error up to 1.8e23 (median 6.8e8 in that bin). The top 1 % of cells carry 100.0 % of the mean. Consequences:
- The chapter's matched-budget, fixed-N, Branch-B and B-vs-A tables report seed **means** (values 1e5 to 1e27 µg/s): uninterpretable. The "B constant component is half of A's error" ratio (0.50–0.66) is an artefact; the paired median ratio is 1.00 at preregistered σ and 1.17 at 0.1× (B better in 51 % and 23 % of pairs).
- All three figures plot means on log axes; same problem.
- Medians (used by the lead's journal and by one chapter sentence) are not distorted: rel. error median 0.97 (σ) and 0.51 (0.1× σ).
- 47 % of branch-A cells at preregistered σ have relative error > 1 (worse than reporting zero), 11 % > 10.

Fix: prior-referenced ridge, λ = (σ/a_ref)² in data units (equivalently 1/a_ref² after whitening). That bounds the error near the prior scale in null cells and matches the practical-rank definition. Report medians and quantiles, never means of src_rmse.

### F3 (High). Claims (2) and (3) do not hold as stated

- **(2) "K = 4: N = 4 over 168 h beats N = 8 snapshot."** The comparison is N = 4, m = 16 (B = 64 readings) against N = 8, m = 1 (B = 8). At matched B = 8 (N = 4, m = 2, W = 168 h) practical rank is 1.50 vs 1.38 (11 wins / 7 losses / 22 ties, n.s.) and median rel. error 1.03 vs 1.26, **both ≥ 1**. So the gain is mostly reading count plus wind change (F1). Valid narrower statement: 4 sensors read twice a week apart (two different block directions) match 8 sensors read once; neither estimates the sources usefully at preregistered σ. Note also N = 8 read twice, a week apart, beats the N = 8 snapshot in 19/40 pairs with 0 losses: two readings in different wind directions are worth more than a long window.
- **(3) "At matched budget more sensors beat more readings per sensor."** Only N = 2 is reliably worse: at B = 16, N = 8 beats N = 2 in 27–31 of 40 paired cases (sign p 0.0007–0.04), but N = 8 vs N = 4 is 19–24 of 40 (p 0.27–0.87). At B = 32, K = 4, W = 168, N = 4 has the lower median error (0.71 vs 0.82). Supported: "two sensors are too few"; unresolved: anything between 4 and 8.

### F4 (Medium). Claim (1) holds for rank (weakly), not for error

- Practical rank: at fixed m = 4, seed-mean rank never decreases with W, but per seed 83 % of (wind, K, N) pairs are **tied** between 24 and 168 h (13 % up, 4 % down). With "best m" (as in the chapter's rank table) the comparison also changes the reading count: W = 1 allows only m = 1, W = 6 allows m ≤ 4, W ≥ 24 allows m ≤ 16.
- Source error: at fixed m = 4 the median is **not** monotone in W (e.g. constant, K = 4, N = 4: 0.85 → 0.88 → 0.98 for W = 6, 24, 168; per seed, error got worse from 24 to 168 h in 54 % of pairs). "Source error lowers monotonically for every K, N" is contradicted at fixed m.

### F5 (Medium). Wind-diversity contrast: mis-posed, and the label "contradicted" overstates a null

- Cell-level changing − constant compares two winds that point in **different directions at the issue hour** (veering ends 45° north of the westerly in every block: e.g. 301° vs 256° for seed 0). At W = 1 and 6 h neither wind changes within the window, so the contrast measures geometry, not diversity: practical rank is *lower* under veering there (mean −0.27 and −0.19) and higher at 24 and 168 h (+0.18, +0.14). Averaging cells cancels these.
- A post-hoc difference-in-differences that removes the geometry, [rank(veering, 168 h) − rank(veering, 6 h)] − [same for constant], m = 4, averaged over K and N per seed: mean +0.26 practical rank, 14 of 17 non-tied seeds positive, sign p = 0.013 at preregistered σ; +0.18, 13/18, p = 0.10 at 0.1× σ. **Exploratory** (devised after seeing the data), but it is the contrast the hypothesis actually asks for.
- The sign-test rule calls "no detectable difference" in ≥ 70 % of cells "contradicted". With 20 pairs, 66 % ties and 84 strongly dependent cells (nested N, shared seeds), that is absence of evidence, not evidence of absence. The design note's P1 asked for a paired bootstrap CI; it was replaced by a cell-wise sign test. The tie-exclusion part of the revision is reasonable (ties at 0 or at full rank carry no information) and **did not bind** at 1 g/s (84/84 cells informative); the "contradicted" threshold is the problem. Seed, not cell, is the unit of replication; the seed-level test gives no detectable main effect (K = 2: 8 up/10 down; K = 4: 10/10).
- The lead's explanation (constant drifts 15–30°, veering only 45° in the first 12 h) is correct in substance (F1 quantifies it) and should be stated as the reason the main contrast cannot answer the question.

### F6 (Medium). Practical rank: defensible as an ordinal measure, lenient as a threshold

Median relative error falls with practical rank in every stratum (K = 4, σ: 5.2, 1.00, 0.78, 0.53, 0.17 for prank 0–4; K = 2: 2.7, 0.63, 0.22), so it is consistent with estimator performance. But s·a_ref/σ ≥ 1 means CRB sd ≈ a_ref along that mode, i.e. ~100 % error at reference strength (truth is 0.5–5 a_ref): a mode can "count" and still be estimated badly (prank 1 of 4: median error 1.0). The a_ref×10/×100 columns are signal-scale sensitivity, not a stricter criterion; the design note asked for ×0.1, ×1, ×10. For E08 add a threshold of 3 (or report CRB relative to the true strengths) alongside 1.

### F7 (Medium). No held-out times and no forecast

`field_rmse_issue` is computed at the issue hour, which is itself a reading hour, over all 256 candidate sites (sensor sites included). It is an in-sample spatial reconstruction, not held-out-time skill, and it is not normalised (median 0.26 µg m⁻³ at σ = 0.25, i.e. ~1σ). The primary endpoint (one-step forecast) is not evaluated, as the design note stated. Look-ahead is correctly prevented (`stack_window` rejects hours > issue; no reading after 167; the calendar is a single train segment, so no held-out regime is used).

### F8 (Low). `reading_hours(·, W, m = 1)` reads the *start* of the window

`np.linspace(a, b, 1)` returns `[a]`, so for m = 1 the single reading is at hour 167 − W + 1 (W = 168 → hour 0), contrary to the docstring ("last = issue"). m = 1, W > 1 cells are therefore a single reading 6, 24 or 168 h before issue; they enter the "best m" maximum and the B = N tables. Practical effect small (rank is shift-invariant under steady wind), but those cells are not what the chapter describes. Fix: special-case m = 1 to `[issue]`, or exclude m = 1, W > 1.

### F9 (Low). Chapter text errors

- "20 seeds … This is *fewer* than the 20 seeds in the design note" (stale template sentence).
- "σ = 0.25": correct to 1e-8 for most scenarios (σ = 5 % of median site truth, dominated by the 5 µg m⁻³ background) but ranges to 0.267 (changing) / 0.260 (constant). Harmless; σ is essentially fixed, so it does not confound the wind contrast.
- Wind table headed "contradicted" for all rows (F5); Branch-B ratio (F2).
- "Best cell 0.04 and 0.004": minima over 3 360 cells, selection-biased; drop or label as such.

### Checked and found sound

- **Pairing.** Sources and strengths (`SeedSequence([truth, K])`), sites (placement stream, admissible set depends only on source positions) and wind speed/offset draws (`WIND_BASE + replicate`, same rng consumption for both regimes) are identical between winds. Noise `default_rng([4000, seed, K, len(wind)])`: `len("constant") == len("changing") == 8`, so both winds get the same draws — intended common random numbers, but by accident of string length; replace with an explicit constant. Draws are indexed by (site, hour), so they are shared across N (nested), W, m, branch A/B (K = 2) and noise scale (rescaled). Good.
- **Nested random sites** are a fair paired comparison across N (N = 2 ⊂ 4 ⊂ 8, same permutation); they differ between K because the admissible set depends on sources. They measure random-network behaviour only.
- **Unknowns** stated correctly (p = K, p = 3K); rank ≤ min(rows, p) never violated.
- O4 duplication oracle is an algebraic identity; O1 is a wiring check (same solve), as the journal says.

### Branch B: what it says

With a constant truth, the diurnal over-parameterisation (p = 6) reaches float rank/p 0.44 (snapshot) → 0.71–0.75 (24 h) and no further at 168 h, as predicted for a 24 h-periodic basis (median full rank 6 at N = 8, m = 16, W = 168). Paired with branch A, the constant-component error is unchanged at preregistered σ (median ratio 1.00) and 17 % worse at 0.1× σ. Exploratory reading: the extra temporal unknowns cost little here because noise dominates; it does not test recovery of a real diurnal source (truth has no diurnal part, K = 2 only, unconstrained). It should not be cited as evidence about temporal source fingerprints.

### Post-hoc changes: what must be labelled exploratory

1. **a_ref = 1 g/s (A1).** Reasonable: the old 100 µg/s was physically tiny and gave zero information everywhere, and the decision record is explicit that it was chosen from outcome data. Every E05 number at 1 g/s is exploratory; that is correctly stated in the chapter and decision.
2. **Tie-exclusion wind rule.** The exclusion is principled and did not change any cell at 1 g/s; the 70 % "contradicted" bucket (inherited) is not. Both the rule and this audit's difference-in-differences are exploratory.
3. **0.1× noise level**, ridge 1e-3 and a_ref×10/×100: not preregistered; sensitivity only.

## 4. Claims versus evidence

| Claim (lead / chapter) | Status |
|---|---|
| (1) Longer windows raise practical rank for every K, N | **Partly supported**: seed means non-decreasing at fixed m; 83 % of per-seed 24→168 h pairs tied; gain vanishes with no direction change (F1). Reword as wind-direction coverage. |
| (1) …and lower source error monotonically | **Contradicted** at fixed m (F4). |
| (2) Longer window partly compensates for fewer sensors (K = 4, N = 4 × 168 h vs N = 8 snapshot) | **Overstated**: confounded with 8× more readings; budget-matched difference n.s. and both errors ≥ 1 (F3). |
| (3) At matched B, more sensors beat more readings | **Supported only for N = 2 vs ≥ 4**; N = 4 vs 8 unresolved (F3). |
| (4) Wind diversity: no detectable rank gain in most cells; contrast weak | **"No detectable" correct at cell and seed level, "contradicted" wrong**; contrast is geometry-confounded; DiD gives an exploratory positive signal (F5). |
| Repeated snapshots add no rank (hypothesis part 2) | Algebraic identity (O4), correctly labelled. |
| Branch B constant error ≈ 0.5 × branch A | **Artefact of means**; median ratio 1.0–1.17 (F2). |
| Median rel. error 0.97 (σ) / 0.51 (0.1×) | Reproduced. |
| Practical-rank-0 fractions 0.32/0.31; full-rank 0.074/0.08 | Taken from code; consistent with my re-analysis (not re-derived line by line). |

## 5. Recommendations

**Before the chapter is cited (E05 cleanup):**
1. Replace all src_rmse/field_rmse means by medians and quantiles in tables and figures; recompute the B-vs-A ratio as a paired median (F2).
2. Re-label the wind table "no detectable difference / unresolved", and add the window × wind difference-in-differences as an explicitly post-hoc analysis (F5).
3. Rewrite claims (1)–(3) per §4; state that the 7-day gain disappears when the block direction offset is removed (F1, cite this audit's spot check).
4. Fix the stale "fewer than 20 seeds" sentence and the m = 1 reading-hour bug or its description (F8, F9).
5. Commit the audit scripts with this file.

**Does E05 need a rerun?** Not to retain its exploratory findings. A rerun is worthwhile only together with (a) the prior-referenced ridge and (b) an exactly-constant wind control, which together turn F1 and F2 into designed evidence; both are cheap (the full sweep is ~45 min).

**Which durations merit the larger replicated experiment (E08):**
- **Drop W = 1 h vs 6 h as a separate factor**: identical rank in every no-jitter case and nearly so in the data; the snapshot (W = 1, m = 1 at issue) is enough as baseline.
- **Keep 24 h and 7 d**, but define the factor as **number of distinct wind directions sampled** (e.g. 1, 2, 4 blocks) crossed with window, so duration and direction diversity can be separated. 24 h within one block is the "same direction, more time" arm; 7 d with an exactly constant wind is the null control.
- **Keep N ∈ {2, 4, 8}** (and add 16 if E06 placements are used), since the N = 4 vs 8 question is unresolved; prefer optimised and random sites as separate arms.
- **Reading budget**: B ∈ {8, 16, 32} with sparse cadence (readings spread across blocks) versus contiguous.

**Preregister for E08:**
1. Unit of analysis = scenario seed; ≥ 20 seeds (power: the DiD effect of ~0.26 rank with 66 % ties needs more than 20 for a cell-level claim); paired bootstrap CI over seeds, not per-cell sign tests.
2. Primary contrast for the wind hypothesis: window × wind interaction (DiD) on practical rank, with an exactly-constant wind control (direction offset off, speed factor matched); decision rule with a pre-set minimum meaningful effect (e.g. 0.25 rank) so "contradicted" requires the CI upper bound below it.
3. Estimator: prior-referenced ridge (λ = σ²/a_ref²) or NNLS, fixed in advance; summarise errors by medians/quantiles only.
4. Practical-rank thresholds 1 and 3 (σ units at a_ref), plus CRB sd relative to the true strengths; a_ref = 1 g/s fixed (A1) with ×0.1/×10 sensitivity as the design note asked.
5. Matched-budget comparisons at equal B only; wall-clock comparisons labelled as such.
6. Held-out evaluation: field error at hours after the issue (forecast) and at non-sensor sites, normalised by the true anomaly rms.
7. AR(1) noise (ρ = 0.8) as a planned arm: a contiguous 7-day window carries ≈ 1/9 of the white-noise information (design note §6), which could reverse the duration conclusions.
8. Explicit noise-seed constants per arm (not `len(wind)`).

## 6. Not checked

- Figures were not opened; their defect is inferred from the plotting code (means).
- The no-jitter spot check covers constant wind, K = 4, seeds 0–7, branch A only; I did not rerun veering without jitter, K = 2, or branch B.
- No AR(1) analysis; no test of `tests/test_time_observability.py`; no independent check of `generate_scenario` beyond reproducing two scenarios.
- Physical plausibility of the 1 g/s scale and of upwind numerical diffusion (gate-00 B3) is out of scope; both affect absolute ranks.
- I did not check the chapter's practical-rank-0 and full-rank fractions independently line by line.
