# PRAVAH E05 — temporal observability: history window, measurement budget and temporal emissions basis (design)

**Task:** T-027 · **Experiment:** EXP-006 · **Author:** claude-balanced / claude-sonnet-5-5
**Status:** DESIGN SPECIFICATION. Not frozen, not human-approved, not reviewed. It reports **no result** about EXP-006. The only computation behind it is one analytic check (§7, RUN-0019).
Synthetic 2-D passive tracer, known wind, perfect model, known source locations. Nothing here concerns ambient PM2.5.

## 0. Prerequisite handoff

- **Upstream code.** `src/pravah/{sensitivity,information,identifiability,observations,benchmarks}.py` are present at this worktree's base (`40f9b43`): `scenario_response_matrix`, `stack_jacobian`, `fisher`, `rank_coherence`, `practical_rank`, `noise_field` (white, AR(1)). I did not check that they equal main beyond that file presence, and I did not merge or copy anything.
- **Preceding evidence.** EXP-005 has F-004 (preliminary, awaiting review) and the T-026 audit (`gates/gate-04-identifiability.md`, "PARTLY CLEARED, not an approval"). No finding is reviewed. Those results are valid negative/unresolved outcomes and do not contradict the physical assumptions or the preregistered criteria, so this design proceeds. They carry these limits into EXP-006:
  1. At the preregistered noise σ, practical rank was 0 of 4 in every E04 smoke case (F-004), and the E02 plume is ~1e-3 ug m⁻³ against σ = 0.25 (F-002). A temporal-rank study at the preregistered scale may therefore report "nothing above noise" for every window. That is a valid outcome and must be reported as such (§8, decision D1).
  2. `information.criteria` thresholds M at 1e-8 (singular-value ratio 1e-4) while `rank_coherence` thresholds G at 1e-8; they disagree (gate-04 §6.1). This design never calls `criteria` for a rank claim (§4).
  3. A CRB ellipse is conditional on the noise model; under AR(1) ρ = 0.8 ignored, coverage of a nominal 95 % ellipse was 14 % (gate-04 §4). §6 builds the correlated-noise tests on that.

## 1. Questions and hypothesis (restated, not changed)

EXP-006 hypothesis: *changing wind directions can increase effective rank for constant/low-dimensional source coefficients; repeated redundant snapshots reduce noise without separating otherwise collinear sources.* Q-006: when does a longer history compensate for fewer sensors? Q-009: how do k, source complexity, noise, cadence and duration interact?

Every statement in §5 is a **prediction to test**, not a premise. In particular **we do not assume more times raise rank**: §3.3 gives cases where they provably do not.

## 2. Notation and the object under study

Sensor set S (|S| = N), reading hours T ⊂ {0..H_w−1} (|T| = m per sensor), K sources at known locations. Anomaly readings are linear in the coefficient vector θ ∈ R^p:

```
y = c_bg + G θ + ε,   G ∈ R^{(N·m) × p},   ε ~ N(0, R)
```

- **Branch A (constant amplitude).** g_k(t) = 1, θ = a ∈ R^K (ug s⁻¹), **p = K**. This is the existing `scenario_response_matrix` column space.
- **Branch B (low-order temporal basis).** g_k(t) = Σ_{j<J} β_{kj} b_j(t), θ = (β_kj), **p = K·J**, column (k, j) is the response to source k driven by b_j(t)·φ_k. Two bases, both **periodic with period 24 h** (so they repeat across a 7-day window and carry no extra information in time beyond one day):
  - B-diurnal: b = {1, sin(2πt/24 h), cos(2πt/24 h)}, **J = 3**. Equivalent to the preregistered variant 1 + 0.5 sin(2π(t − t_k)/24) with phase t_k absorbed linearly (β_k1 = 0.5 cos, β_k2 = −0.5 sin).
  - B-piecewise: four 6 h-of-day indicator functions, **J = 4**.
  The coefficient dimension p is printed in every table; nothing is compared across branches at equal rank without equal p, and rank is also reported as the fraction rank/p.
- **Positivity.** g_k(t) ≥ 0 is not implied by free β; the diurnal basis needs β_k0 ≥ |(β_k1, β_k2)|, a cone constraint NNLS does not impose. Identifiability here is a **linear, unconstrained** property of G (as in §9 of the protocol); constrained-estimator error is a secondary diagnostic only and the constraint choice is decision D3.
- **Implementation gap (not done here).** `_anomaly_fields` accepts one constant source field S. Branch B needs the same solver driven by S·b_j(t) per hour (b_j piecewise constant within an hour). That is new code in a task permitted to change `src/`; this task cannot. Branch A needs none.

Units: G in ug m⁻³ per (ug s⁻¹); background c_bg is outside G (known in base). Time hourly, 12 solver steps per hour, spin-up 48 h discarded.

## 3. Windows, budget and wind factors

### 3.1 Windows and wall-clock separation

Windows end at the forecast issue time (protocol §7: "7-day window ends at, and does not extend past, the issue time"). Window length W ∈ {snapshot, 6 h, 24 h, 7 d = 168 h}; the snapshot is one reading time (the issue hour). Wall-clock duration W and the number of readings are **two different factors** and are varied separately:

- **Matched-budget comparison (reading count fixed).** Total readings B = N·m fixed; each sensor takes m readings evenly spaced over W, the last at the issue hour. Cadence Δ = W/(m−1) (m ≥ 2), rounded to a whole hour; if a window cannot hold m distinct hourly readings the cell is **infeasible** and reported as such, not padded. Planned budgets B ∈ {32, 64} (both factor over the N levels); B = 32:

  | N | m | snapshot | 6 h | 24 h | 7 d |
  |---|---|---|---|---|---|
  | 32 | 1 | yes | n/a | n/a | n/a |
  | 16 | 2 | no (m = 1 only) | yes | yes | yes |
  | 8 | 4 | no | yes (4 of 6) | yes | yes |
  | 4 | 8 | no | infeasible (6 hourly times) | yes | yes |
  | 2 | 16 | no | infeasible | yes | yes |

  (The snapshot has m = 1 only, so only N = B matches it.)
- **Fixed-count, varying duration.** Same N and m, different W (e.g. N = 8, m = 4 at 6 h, 24 h, 7 d): isolates *spacing in time* (a longer span sampling more of the wind/transient history) from count.
- **Fixed-N, growing history (unmatched).** All hourly readings in W (m = W): budget grows with W. Reported separately and labelled "unmatched"; it must never be used to claim duration beats count.

Sensor sites are **random, nested** (the N = 2 set ⊂ N = 4 ⊂ …) from the 16 × 16 candidate lattice with the 1.25 km minimum separation, drawn by `placement_seed = 3000 + s`. No optimisation, no use of test truth: optimised designs are E06–E08, and this experiment must not use information the placement experiments have to earn.

### 3.2 Wind diversity

Winds are the preregistered **training** regimes only (`steady_westerly`, `veering`, `weak_variable`). The held-out regimes (steady southerly, stagnation-then-front, diurnal rotation) are never used (protocol §8); `diurnal_rotation` in `make_winds` is one of them and is excluded.

| Wind factor | Definition | Direction change within W |
|---|---|---|
| constant | `steady_westerly` | none by design |
| changing | `veering` (270°→315° over 12 h, repeated per 48 h block) | 45° per block |
| strongly changing | `weak_variable` (180° reversals each 6 h, speed 0.5–1.5) | 180° |

Two caveats that the analysis must carry. (i) `make_winds` perturbs each 48 h block with speed ×U(0.8,1.2) and direction +N(0,10°); so "constant" is constant only within a block, and a 7-day constant window has ≥ 4 blocks with ~10° jitter. If an exactly constant control is wanted it needs a wind option with the perturbation switched off (decision D2; the 10° jitter is itself a small wind change that can add rank, so without the control the contrast is confounded). (ii) The regimes differ in mean speed (3 vs 3 vs ~1), so "changing vs constant" is confounded with speed and with advection length unless speeds are matched; the design therefore reports each regime separately rather than pooling, and states the speed difference beside every table.

### 3.3 Cases where more times cannot add rank (to be stated as oracles, not discovered)

1. **Row duplication.** Appending rows already in the row space leaves rank unchanged: rank(G) ≤ min(N·m, p) and equals the rank of the distinct rows. *Repeated readings of the same time* (m copies of the snapshot with independent noise) add information in observed directions (information scales as m for white noise) and add exactly zero rank. This is the "repeated redundant snapshot" in the hypothesis and is exact, not an empirical claim.
2. **Stationary wind after transients.** For constant wind, constant sources and spin-up long relative to the slowest mode (e^{−λt}, 1/λ = 12 h), the anomaly at reading hour h converges; rows at hours h and h+1 become the same footprint up to e^{−λ·Δh}·(prefactor). Later hours then add rows in an already spanned space, so rank saturates. Where the rows are near-duplicates, *numerical* rank can still grow in the 1e-8 sense, so the practical (noise-referenced) rank is the primary tool, not the floating-point rank. This is a prediction about the constant branch only.
3. **Periodic basis.** In Branch B with period-24 h bases, readings more than 24 h apart under repeated winds add no new temporal phase; a 7-day window helps only through wind change, not through extra phase coverage.
4. **Rank ceiling.** A snapshot of N sensors gives rank ≤ N; a 6 h window with J = 3 covers 1/4 of the period and is expected to be nearly collinear between the 1 and the sin/cos columns (a *prediction*, §5).

## 4. Metrics (descriptive; none enters endpoint ranking — protocol §9)

Per cell, per scenario seed, per branch:

1. **Float rank** of the whitened G at relative singular-value tolerance 1e-8 (`rank_coherence`/`rank_info` rule) and the 2-norm condition number; rank/p.
2. **Practical rank** `practical_rank(G, σ, a_ref)` = #{s·a_ref/σ ≥ 1} with a_ref = 100 ug s⁻¹ and σ the preregistered per-family value. For Branch B the same rule is applied per coefficient *and* the amplitude scale is stated (a_ref multiplies b_j of order 1, so it is comparable to Branch A for the constant component only; for sin/cos it is the stated convention). Gate-04 §6.5 notes a_ref is a free choice; D1 asks for a sensitivity analysis rather than a tuned value.
3. **Spectrum.** Full singular-value vector s/σ (stored), the weakest mode and the null counterexample (`null_counterexample`: data change in σ units for a 50 % ‖a‖ move) — one source combination named per cell.
4. **CRB fraction.** Fraction of θ components with CRB sd ≤ 50 % of the true value (protocol §9), reported only where M is nonsingular by the singular-value rule; otherwise "undefined", never imputed. `criteria` is not used for this (§0 item 2).
5. **Collinearity of the specific pair** (hypothesis part 2): the cosine between whitened columns of the two closest sources, and the effective sd of their *difference* from M⁻¹ (undefined if singular). "Collinear" = cosine ≥ 0.99 at the snapshot design (a design-time label, not a tuned threshold).
6. **Secondary, exploratory:** source RMSE of the NNLS/WLS inverse (`inverse.solve_nnls`, ridge fixed in advance on the train segment), and the *field* error along the weakest mode (gate-04 §8.4 asked for it; reported separately from parameter information).

No new endpoint is introduced and none of the preregistered ones is changed. Primary endpoint (one-step forecast RMSE) is **not** evaluated here.

## 5. Predictions and decision rules (written before data)

Unit of analysis: scenario seed s = 0..19 (`truth_seed = 1000+s`, `noise_seed = 2000+s`, `placement_seed = 3000+s`), paired across windows, winds, branches. 95 % CI from a paired bootstrap over seeds (10 000 resamples, seed 4000, as the protocol). Cells are not pooled before differencing.

| ID | Prediction | Test | Supported if | Contradicted if |
|---|---|---|---|---|
| P1 | Changing wind raises practical rank over constant wind at the same (N, m, W, K, branch) | paired Δ(practical rank), changing − constant | CI entirely > 0 in ≥ 70 % of the feasible cells | CI entirely ≤ 0 in ≥ 70 % |
| P2 | Repeated identical snapshots add zero rank and scale information ∝ m | exact (oracle) and measured | rank Δ = 0 exactly, information ratio = m within 1e-9 | any violation → a code bug (stop) |
| P3 | At matched B, a long window with few sensors can match a snapshot with many sensors in rank **only when the wind changes** | paired Δ(rank), long-few − snapshot-many, by wind | stated per wind; no pooled claim | — |
| P4 | Branch B (J > 1) needs a longer window than Branch A for the same rank/p | rank/p vs W | descriptive plot with CI | — |

The 70 %-of-cells rule copies the protocol's supported/contradicted structure (§12), **applied by analogy**; the protocol states it for the placement hypothesis, not this one, so using it here is an adaptation to be approved (D4). "Unresolved" is any other outcome, including cells NOT RUN or practical rank 0 everywhere (then every P is *unresolved because nothing is above noise*, not contradicted). A positive effect is not required for EXP-006 to pass.

## 6. Temporally correlated observation errors (test plan)

Noise: white (primary) and AR(1), ρ = 0.8 per **hourly** step, same marginal σ, independent across sensors (as `noise_field`). When readings are not hourly-adjacent the correlation is ρ_Δ = 0.8^{Δ/1 h} (Δ in hours), so a sparse cadence is nearly white while a dense one is strongly correlated. This is a design consequence: at equal B a spread-out schedule loses less information to correlation than a contiguous one.

Planned tests (each a pass/fail oracle or a measured quantity):

1. **Analytic oracle (done).** For G = 1 (one constant column) and n equally spaced hourly readings under stationary AR(1) with unit marginal variance, information = (n(1−ρ)+2ρ)/(1+ρ). RUN-0019 compared the formula with `1ᵀR⁻¹1` computed numerically for n = 1, 5, 24, 168 (ρ = 0.8) and n = 10 (ρ = 0): agreement to 1e-15 relative. It is a check of the algebra the whitened `fisher(G, noise_cov=R)` must reproduce. Limit n → ∞: n(1−ρ)/(1+ρ) = n/9 at ρ = 0.8. So 168 contiguous hourly readings carry the information of ≈19.6 independent ones for a constant, not 168.
2. **White-vs-correct information.** Compare log det and λ_min of M under assumed white vs true AR(1) R, per window and cadence; report the overstatement (gate-04 / T-023 §5: 3.82 in log det on one example).
3. **Coverage under misspecification.** Re-run the gate-04 style Monte Carlo (2000 draws) for each window: nominal 95 % ellipse coverage with (a) correct R, (b) white assumed, for contiguous vs spread cadence. Expectation (a prediction): (a) ≈ 95 %; (b) falls with window density. The MC script must live in the repository this time (the gate-04 script sat in /tmp and made RUN-0018 non-reproducible from the commit).
4. **Does correlation change the rank claim?** Rank of W·G (W = R^{-1/2}) vs rank of G: whitening by an invertible matrix leaves exact rank unchanged, so any change is conditioning only. Test that equality, and report the change in practical rank (not an oracle: it is where correlation matters).
5. **Cross-sensor correlation** is not implemented in `noise_field` and is out of scope; said so rather than assumed away.

## 7. Validation oracles and failure cases (to be built before any production run)

| # | Oracle | Expected |
|---|---|---|
| O1 | G(Branch B, J = 1, b₀ = 1) equals Branch A's G exactly | max abs diff ≤ 1e-12 |
| O2 | Linearity: G applied to θ equals direct simulation with that source history (use g_k(t) = 1 + 0.5 sin) | relative L2 ≤ 1e-9 (same solver, same steps) |
| O3 | Causality: readings before a basis function switches on are unchanged (lower-triangular in release time) | exact zeros |
| O4 | Row duplication leaves rank unchanged; information ×m | exact |
| O5 | Rank ≤ min(N·m, p) for every cell | never violated |
| O6 | AR(1) information formula (§6.1) | **done, RUN-0019** |
| O7 | Rank at 1e-8 equals `rank_coherence`; `criteria` not used for ranks | by construction |
| O8 | Leakage: a reading at hour > issue time must not enter G (window ends at issue time) | frozen-test style assertion |

Failure cases the analysis must keep, not drop: practical rank 0 at the preregistered σ; windows infeasible at a given (N, m); singular M (CRB undefined); Branch B nonnegativity violated by the unconstrained estimate; exactly collinear sources with identical footprints (rank deficit that no extra time fixes).

## 8. Compute, design size and decisions required

- **Design size.** The sensitivity G is the costly object: one forward solve of 48 h + 168 h per (scenario seed, K, wind, basis function); every (N, m, W, noise) cell is a cheap row selection of the same 168-hour G. Counting solves: seeds 20 × K{2,4,8} (14 columns for Branch A, 14·J for B) × 3 winds. I did **not** time a solve at 32² or 64², so I give no core-hour estimate here; the first task must measure it and compare with the protocol's 48 core-hour local cap. The pre-declared reduction from the protocol (drop K = 8 × N ∈ {2, 4} first, NOT RUN) applies; no other silent cell removal.
- **Decisions for a human (none made by me):**
  - **D1 Noise scale.** At the preregistered σ the expected result is "practical rank 0 everywhere". Either report that (the safe, protocol-consistent option), or approve a *separately labelled* amplified scale *before* any run. a_ref sensitivity (×0.1, ×1, ×10) should be reported either way.
  - **D2 Constant-wind control.** Approve an exactly-constant wind option, or accept the ±10° block jitter in the "constant" arm.
  - **D3 Branch B constraint.** Unconstrained linear identifiability only (this design), or cone-constrained estimation as a secondary.
  - **D4 Decision rule.** Approve the 70 %-of-cells rule by analogy (§5) or supply another.
  - **D5 Budgets.** B ∈ {32, 64} here versus the proposal's N ∈ {2…32}; confirm.

## 9. Limitations

Perfect model, known locations and winds, ideal sensors, uniform wind and constant K, periodic temporal bases, a single mixing depth and a synthetic setup, so nothing extends to real air. Rank is a property of (network, model, basis), not of an estimator; field skill and parameter information are different quantities (gate-04 §5). The matched-budget fairness control equalises reading *count*, not information: correlated-noise cases show those can differ by an order of magnitude (§6). Random nested sensors are not placement-optimal, so any rank gain from time is **relative to this baseline** and may shrink or grow under optimised placement (E06–E08). Predictions P1–P4 are untested.

## 10. Provenance

| Item | Value |
|---|---|
| Written by | task T-027, experiment EXP-006, branch `exp/EXP-006-temporal-observability-sensor-count-vers` at base `40f9b43` |
| Computation | RUN-0019 only (`python3 -c`, AR(1) information formula vs `numpy.linalg.solve`, exit 0, 0 s). The command is stored with the run. No scenario, solver or sensitivity matrix was run for this document |
| Inputs read | `preregistered-protocol.md`, `gates/gate-04-identifiability.md`, `footprints-identifiability.md`, `src/pravah/{sensitivity,information,identifiability,observations,benchmarks}.py` (partial reads of the first two and of `benchmarks.py`) |
| Not read | the E04 CSV/meta, `tests/`, `truth-design.md`, Green (2006), Bhardwaj & Subramanian (2026); statements about the E04 numbers are taken from the gate-04 text and F-004, not recomputed |
| Code changed | none |
