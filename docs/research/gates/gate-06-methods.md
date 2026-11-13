# Gate 06: independent review of E06 (placement algorithms and information criteria)

**Verdict: CLEARED for E06 as a methods/reproduction step, with conditions before E07 reuses it.** The implementation is correct where I could test it. Every strategy sees the same candidate set, budget and spacing rule. The Bayesian D/A/E formulas and the a_ref/σ scaling are right. Tie-breaking is deterministic, and the old E04 greedy design is reproduced exactly. The QR pivots are standard Businger–Golub pivoting, which is also what PySensors 0.4.3 uses, and PySensors is MIT-licensed and used only as a dev-time test reference. The committed CSV reproduces bit-for-bit on the two scenarios I reran (108/108 rows). The chapter avoids winner claims and its numbers match the CSV. Two of its statements need rewording (F6).

**Conditions for E07.** Three properties of the setup would mislead a cross-objective comparison if they stay implicit:
1. "Training-only" means training *hours*, not training *information*. Sources are constant over the whole calendar, so the noise-free POD archive contains the same source strengths that E07 will score on held-out data (F1).
2. Every practical-rank comparison in E06 is in-sample (the same 96 h G that D/A optimise) and compares oracle-location designs against blind ones. Most of the advantage comes from knowing where the sources are, not from the optimisation (F2).
3. The greedy-vs-exhaustive result (D-opt 24/24 exact) comes from a candidate set on which the spacing rule never binds. When it binds, greedy D is exact in only 35/80 synthetic cases (F3).

The noise-aware POD mode count is a post-hoc choice whose noise accounting differs from D-opt's by a factor of √T. It decides whether POD/QR exists at all (F4).

**Reviewer:** Opus 5.5 subagent, plan-02 `e06-placement-algorithms` step 4. Same model family as the author, so independence comes only from the task split and re-derivation. I changed no repository file.

## 1. Provenance

| Item | Value |
|---|---|
| Repository | `/home/yubraj/research/pravah-new`, main, code read at HEAD `f0045fd` |
| Data audited | `reports/quarto/tables/e06-criteria.csv` (432 rows: 370 ok + 62 undefined), `e06-*.md`, `e06-meta.json` (run of `c4564d2`, 14.2 min) |
| Audit scripts (this directory) | `audit_csv.py` → `audit_csv.out` (re-analysis of the CSV); `audit_spot.py` → `audit_spot.out` (re-simulates veering/seed 0/K = 2 and steady_westerly/seed 0/K = 4 with G over 120 h; reproduction, mode amplitudes, tie margins, cand-order reversal, held-out practical rank, strength dependence of POD/QR); `audit_binding.py` → `audit_binding.out` (synthetic 5×5 lattice where spacing binds, greedy vs exhaustive); `audit_wrapper.py` + `old_identifiability.py` (copy of `a4f4827`) → `audit_wrapper.out` |
| Commands | `cd /home/yubraj/research/pravah-new && OMP_NUM_THREADS=1 uv run python <gate-06>/audit_{csv,spot,binding,wrapper}.py` (`audit_spot.py` takes ~6 min single core for both cases; env `AUDIT_CASES=wind:seed:K,...`) |
| Tests | `uv run pytest tests/test_placement.py tests/test_identifiability.py`: 63 passed (PySensors installed, so its test ran) |
| Reproduction | 2 scenarios × 2 candidate sets × 27 design rows: sites, status (including undefined) and D values identical to the committed CSV |

## 2. What I read

`AGENTS.md` §2; plan-02 `e06-placement-algorithms` and the first step of `e07-cross-objective`; `docs/research/placement-objectives.md`; `src/pravah/placement.py` (all); `tests/test_placement.py`; `scripts/e06_placement.py`; `src/pravah/identifiability.py` (now and at `a4f4827`); the relevant parts of `benchmarks.py` (`admissible_sites`, `split_hours`, `generate_scenario`, `Truth`), `sensitivity.scenario_response_matrix`, and `e02_examples.make_cfg`; every `e06-*` table; `06-placement-methods.qmd`; `state/experiments/e06.md` and the practical-rank statements in `e04.md` and `e05.md`; `gate-05-duration.md` (for style). I also read the installed PySensors `optimizers/_qr.py` (`QR.fit`) and its package metadata. I did not open the figure PNG, and I read none of the cited papers.

## 3. Findings

Severity: **High** = would invalidate a conclusion; **Medium** = a design property or confound that limits a claim or would mislead E07; **Low** = presentation or a latent issue. No High finding for E06 itself. F1 and F2 become High for E07 if they are not handled.

### F1 (Medium; High for E07). The "training-only" POD archive contains the test-period source strengths

- `training_snapshots` reads only `split_hours("train")` (hours 0–95; verified, and `test_podqr_uses_training_hours_only` poisons the other hours with NaN). In terms of time indexing, POD/QR is training-only.
- The sources are constant over the entire calendar (`generate_scenario` builds `S` once). The noise-free archive is exactly background + G·a: `max |X − bg − G3 a| = 3.9e-5` and `2.0e-5` µg m⁻³ in the two spot cases, against anomalies of 2.85 and 0.74. So the archive encodes the true strengths a, which are the very quantity a source-estimation task in E07 scores on validation/test hours.
- This does change the design. In veering/seed 0/K = 2, `podqr_num` at k = 2 picks sites [227, 219] with the true strengths and [234, 226] when the archive is rebuilt with equal strengths. D-opt is unchanged ([186, 195]), as it should be, because G depends only on locations.
- Every other source-informed design (dopt, aopt, diversity, source_near) uses the **true source locations**. The exclusion mask that all designs share, including uniform, random and coverage, does too. The objectives note documents both facts (access table, §1). What it does not state is that, with constant sources, "training period" and "test period" share the same strengths.
- **E07 must**: (a) state that every design except uniform/random/coverage is an oracle design (locations known, and strengths too for POD); (b) treat POD/QR with a noise-free archive as an optimistic upper bound, or preferably build the archive from *noisy* training readings or from a prior ensemble of strengths; (c) do the same for field reconstruction, where the training mean of a constant-source field already carries the answer.

### F2 (Medium). Practical-rank comparisons are in-sample and mostly measure knowledge of source location

- `practical_rank` in the CSV uses the same 96 h, noise-free, perfect-model G that dopt/aopt/diversity design on. It is not a held-out score.
- **source_near**, which uses only the locations and no G, σ or optimisation, has the same practical rank as dopt in 21 of 24 full-set cases (dopt higher in 3, `audit_csv.out`). Most of the "source-informed designs reach full rank" result is therefore an oracle-location effect, not an information-criterion effect. The chapter groups source_near with the source-informed designs, which is correct, but a reader can still take the paragraph as evidence for D/A optimisation.
- Spot check on the 24 held-out validation hours (G over hours 96–119), steady/seed 0/K = 4, k = 4: dopt 4 and source_near 4 (both 4 in training); aopt **3** and diversity **3** (4 in training); uniform 2; coverage 1; random 0 (training 2, 2 and 1). The ordering survives out of sample in this case, but the levels drop. The veering K = 2 case was unchanged except uniform at k = 2 (1 → 0) and random at k = 8 (2 → 1). Two scenarios, perfect model: illustrative only.
- The column that divides singular values by 3 to mimic AR(1) ρ = 0.8 (information about 1/9, as in the design note) is my rough approximation, not a computation. It lowers several designs by one more rank and should not be quoted as a result.

### F3 (Medium). The exhaustive-search check never covers binding spacing, and greedy is far from exact once spacing binds

- On the small set, every brute force enumerated all C(n, k) subsets (65/65 rows; n_cand 23–25; minimum design spacing 1875 m > 1250 m). The comparison is valid, and as the chapter says it is uninformative about binding spacing.
- Synthetic check (`audit_binding.py`: 5×5 lattice at 625 m, min_sep 1250 m, so 1987 of 12 650 4-subsets are feasible; plume-like footprints; 40 seeds × k = 3, 4):

| greedy | exact / 80 | median gap | max gap (relative) |
|---|---|---|---|
| D-opt | 35 | 0.046 nats | 3.89 nats (19.8 %) |
| A-opt | 16 | 0.0022 | 0.0078 (0.8 %) |
| QR (`podqr_num`) | 40 | 0.010 | 2.54 nats |

- The full candidate set binds (full-set designs have minimum spacing exactly 1250 m). Neither the journal's "D-opt 24/24" nor the E07 designs carry an optimality statement there. The objectives note claims no guarantee under spacing, which is correct. The journal headline should carry the qualifier, and E07 should add a binding exhaustive check (a contiguous 5×5 block of the real lattice is cheap) or a 1-swap local improvement to report how far the full-set designs are from optimal.

### F4 (Medium). The noise-aware POD mode count is post hoc and uses different noise accounting from D-opt

- `r_eff` counts modes whose largest per-site **per-reading rms** contribution is ≥ 1σ. That is the right scale for reconstructing a single snapshot. D/A and practical rank instead let information accumulate over all 96 independent hourly readings, which makes them √96 ≈ 9.8× more lenient per mode.
- With a time-integrated threshold (amp·√T ≥ 1), r_eff would be **5 instead of 2** (veering/seed 0/K = 2) and **1 instead of 0** (steady/seed 0/K = 4). "POD/QR undefined in 46/48 cases" is therefore mostly a consequence of this choice, made after a diagnostic run and documented as a revision. That is honest, but the chapter's "at most a couple of above-noise modes" holds only under the per-reading definition.
- Latent issue: `r_eff` counts *any* modes above threshold, but `podqr_design` uses the *leading* k modes. If a later mode clears the threshold while an earlier one does not, the count overstates the usable prefix. In both spot cases prefix = count, so this did not affect E06.
- `podqr_value` in the cross table uses r = min(|S|, r_num) modes, which include sub-noise "numerical" modes. Values range from −177 to −4.8 and are −inf in 14/185 full-set rows. They are not meaningful for comparing designs. The test comment in `test_podqr_undefined_beyond_above_noise_modes` says "min(|S|, r_eff)", but the code uses r_num.
- **For E07**: preregister one mode-count rule and state which task it serves (snapshot versus window reconstruction). Report podqr's undefined cells as a design outcome, not missing data.

### F5 (Low–Medium). Greedy A-opt is weak, even on its own objective

On the full set the A-opt design has a *lower* A value than the D-opt design in 21/24 cases (by up to 0.019; higher in 3, by up to 0.19). Under binding spacing it is exact in only 16/80 synthetic cases. A saturates near 1 (0.99x at k ≥ 4), and the smallest greedy step margin was 1.9e-6, still far above `TIE_TOL` = 1e-12. The chapter explains this correctly (A is not submodular). For E07, A-opt as implemented is a weak representative of its criterion. Either drop it or add a swap refinement, and do not interpret "A-opt vs D-opt" as a contrast between criteria.

### F6 (Low). Chapter and journal wording

- The chapter says source-informed designs reaching full rank "is new at the amended scale (a_ref = 1 g/s), where the E04/E05 smoke designs had practical rank 0". That is wrong as written. `state/experiments/e04.md` already reports greedy D-opt at 4/4 at 1 g/s, and E05 at 1 g/s had a nonzero practical rank in most cells (gate-05). Rank 0 was the old 100 µg/s scale. The journal's own "consistent with E04's single replicate" is the correct framing.
- The practical-rank table caption reads "mean [min–max] over 4 scenarios per K", but podqr cells have n = 1 and podqr_num cells at k = 4/8 have n = 2.
- `field_rel_err_val` is exactly 1 in 116/185 full-set rows (every median in the cross table is 1, except the single podqr case). The chapter calls it uninformative; it should be left out of E06 conclusions entirely.
- Add "oracle source locations; in-sample on the design window" to the practical-rank paragraph (F2), and "spacing never binds on the small set" to the journal headline (F3).

### F7 (Low). Tests and latent interface issues

- `test_identifiability_wrapper_unchanged` is tautological: it compares the wrapper with its own backend. I checked the real claim separately. The pre-E06 `greedy_design` (`a4f4827`) uses the same formula, scan order and 1e-12 tolerance, and gives identical designs in 50/50 random problems with binding spacing (`audit_wrapper.out`). E04 results are unaffected.
- `evaluate()` passes the scalar `sc.noise_sigma[0]` to `practical_rank`, while `Problem` uses the per-site array. That is equivalent only because the scenario is homoscedastic. Under `heteroscedastic=True`, E07 would silently mix the two.
- The noise model assumes independence across hours (documented). The benchmark's AR(1) variant would cut information about ninefold for contiguous hours and could reorder designs (F2 caveat).
- The prior is N(0, a_ref² I) while true strengths are 0.5–5 a_ref (spot cases 0.51–4.05). The D/A rankings do not depend on the mean, but the practical-rank threshold at a_ref is lenient, as gate-05 F6 found.

### F8 (Low). PySensors comparison and licence (plan item)

- **Code comparison.** PySensors 0.4.3 `QR.fit` is one line: `scipy.linalg.qr(basis.T, pivoting=True)`, keeping the pivot order. `placement.qr_select` re-implements the same Businger–Golub greedy explicitly (largest residual column norm, then project it out, with classical Gram–Schmidt repeated twice), so it can skip pivots that violate the spacing rule, which scipy cannot do. With no binding spacing the two agree: tests compare against scipy (random orthonormal basis) and against PySensors both at optimizer level and end to end (`SSPOR` + `SVD` basis on the same centred training snapshots). Both tests ran and passed. Not compared: PySensors' constrained optimizers (`GQR`/`CCQR`) against our spacing-constrained pivoting, and k > r oversampling, which is not implemented (documented).
- **Licence.** The distribution is `python-sensors` 0.4.3, licence MIT (`License: MIT`; classifier "OSI Approved :: MIT License", from installed metadata). It is listed only in the `dev` dependency group (`python-sensors>=0.4.3`; `uv.lock` pins 0.4.3) and imported only in a test via `pytest.importorskip`. No PySensors code is copied into `src/` (independent implementation), so the project has no MIT attribution obligation. If code is copied later, the MIT notice must go with it. Note that `importorskip` makes the test skip silently when dev dependencies are missing, and nothing in the repo records the licence; this file now does.
- The objectives note says "pinned PySensors". The pyproject specifier is a floor and only the lock pins the version.

### Checked and found sound

- **Candidate equality, budget and spacing.** One `Problem` per (scenario, candidate set) is shared by all nine strategies (n_cand constant within each group). Every design builds S with the same `feasible()` rule, and `design()` asserts k distinct, admissible, spaced sites. Nothing is repaired after the fact. Budgets are identical, and costs are equal by design (cost weights are out of scope, which is documented). The diversity brute force searches above-noise candidates only, matching its declared goal; its gap is NaN when greedy picked a sub-noise site (2/48 rows).
- **Bayesian D/A/E and scaling.** Posterior precision in prior units is a_ref²(1/a_ref² · I + GᵀG/σ²) = I + AᵀA with A = a_ref·G/σ. log det equals 2 × the mutual information. A = 1 − tr(M⁻¹)/K. E = log λ_min(M), evaluated only, never optimised. The identity prior makes an eigenvalue floor unnecessary, so there is no ad hoc regularisation. `test_criteria_values` checks closed forms. σ is one scalar per scenario (5 % of the median training truth, 0.250–0.253) and a_ref = 1e6 µg/s (A1). σ is derived from training truth and so weakly from the strengths, but the 5 µg m⁻³ background dominates it. The submodularity and (1 − 1/e) statement is correctly limited to non-binding spacing.
- **Determinism and ties.** Strict improvement by more than 1e-12, scanned in `cand` order. Reversing the candidate order gave the same D and A sets at k = 8 in both spot scenarios, and the smallest step margins (D: 1.8e-3, A: 1.9e-6) are far above the tolerance, so ties did not decide any E06 design I checked. Brute force breaks ties lexicographically, which does not affect the gap values.
- **Data access.** G covers hours 0–95, which equals the training split (`problem_from_scenario` checks that the window starts at 0 and does not exceed the training length). Validation fields are used only to score reconstruction. The test split is never read.
- **Reproducibility.** 108/108 rows reproduced. `e06-meta.json` records the git revision, but no `runs.jsonl` entry was checked.

## 4. Claims versus evidence

| Claim (chapter / journal) | Status |
|---|---|
| Greedy reached its own exhaustive optimum in 69/87 cases; D-opt 24/24 | **Reproduced from the CSV**; valid only for non-binding spacing (F3) |
| D-opt has the largest D among all designs in every full-set case | Supported (24/24); expected by construction |
| A-opt close to its optimum on the small set; D designs often have larger A on the full set | Supported (21/24 lower; F5) |
| Diversity often misses minimum coherence | Supported (exact in 11/22 evaluable cases) |
| QR pivots equal scipy and PySensors 0.4.3 | Supported (tests ran; F8), unconstrained case only |
| POD/QR undefined in 46/48 cases; 0–2 above-noise modes | Correct **under the per-reading rule**; 1–5 modes under a time-integrated rule (F4) |
| Source-informed designs reach full practical rank more often than baselines | Supported in-sample; driven by **oracle locations** (source_near equals dopt in 21/24); holds in held-out hours with lower levels in 1 spot case (F2) |
| "New at 1 g/s, where E04/E05 designs had practical rank 0" | **Incorrect**: E04 at 1 g/s already had D-opt 4/4 (F6) |
| Objectives choose different networks; no winner claim | Supported; no winner claim found |
| Design inputs are training-only | True for hours; **not** for information, because constant sources put the test-period strengths into the archive (F1) |
| `greedy_design` (E04) unchanged | Supported (50/50 identical vs `a4f4827`; the repo test is tautological, F7) |

## 5. Recommendations

**Small E06 cleanup (no rerun needed):**
1. Chapter: fix the "E04/E05 practical rank 0" sentence; label practical rank as in-sample with oracle source locations, and note that source_near equals dopt in most cases; fix the n in the prank caption; drop `field_rel_err_val` and the cross-table `podqr_value` from any conclusion (F2, F4, F6).
2. Journal: qualify "D-opt 24/24" with "spacing never binds there"; record that `r_eff` is a per-reading rule and that the time-integrated variant gives more modes (F3, F4).
3. Objectives note: add the constant-source caveat to the POD access row; say "lock-pinned" PySensors and record its MIT licence; fix the test comment r_eff → r_num (F1, F7, F8).
4. Commit the audit scripts and outputs beside this file.

**Before E07 (reuses `placement.Problem/design/STRATEGIES`):**
1. Declare every design's information set (blind / locations / locations + strengths), and include source_near as the location-only control, so that the optimisation and oracle effects can be separated (F1, F2).
2. Build the POD archive from noisy training readings, or from an ensemble of prior strength draws, rather than from noise-free truth. At minimum, report the noise-free version as an upper bound (F1).
3. Score on held-out hours with the same noise seeds for every design. Include the AR(1) arm; under it, per-hour independence in D/A overstates information (F2, F7).
4. Add a binding-spacing exhaustive check or 1-swap refinement on the real lattice (F3).
5. Preregister the POD mode-count rule (per-reading or time-integrated) and use a prefix count (F4).
6. Pass σ as the per-site array everywhere, including `practical_rank` (F7).
7. Treat A-opt as a weak greedy, or refine it (F5).

## 6. Not checked

- I re-simulated only 2 of the 8 scenarios, so the held-out practical rank and the mode-amplitude results rest on those two cases. The binding-spacing gap comes from a synthetic plume model, not the real lattice.
- I did not check A-opt or diversity optimality on the full candidate set, the uniform/coverage geometry beyond reading the code, or the figure (not opened).
- I did not compare against PySensors' constrained optimizers. I did not read Manohar 2018, Green 2006, Krause 2008 or the other cited papers, so I did not verify the literature links.
- AR(1) noise was not computed (only the crude rescaling noted in F2). I did not check physical plausibility of the 1 g/s scale or the forward model.
