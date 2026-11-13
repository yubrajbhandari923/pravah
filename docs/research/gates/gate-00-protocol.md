# Gate 00 — Independent protocol audit (T-004)

**Verdict: NOT CLEARED for execution of E01+ until a human resolves blockers B1–B3.** This is an audit, not a scientific approval. I changed no protocol value, metric, frozen test or code. Where design changes look necessary I report them and stop (`scientific_assumption`).

**Scope read:** `preregistered-protocol.md` (DRAFT), `reproducibility-contract.md`, `src/pravah/experiment_config.py`, `tests/test_experiment_config.py` (13 pass on re-run), `reports/quarto/`, `literature-replication.md`, `plans/01-foundations.yaml` (E00 tasks). **Not available to me:** any record of the researcher's approval of the protocol (no decision exists in the ledger; `research current` lists none), so "audit against the researcher-approved plan" can only be done against the plan YAML and `RESEARCH.md`.

**Provenance of numbers below:** one audit run, RUN-0001 (`python3 /tmp/t004/scales.py`, stdlib arithmetic on the protocol defaults, no PDE solved, script outside the repo). Values are back-of-envelope scales (uniform-wind Gaussian-plume estimate, `c ≈ a/(H·u·W)`, `W = 2·sqrt(σ² + 2K·5 km/u)`), not solver output. They need confirmation by E01/E02, but the gaps are orders of magnitude.

## 1. Blocking concerns

**B1 — Source signal is far below the specified noise (scientific_assumption).**
The protocol's units are self-consistent (`a_k` ug s⁻¹, Gaussian `phi_k` m⁻², `S = q/H`), but the magnitudes are not workable:
- Peak in-source rate for `a_ref` = 1.99e-7 ug m⁻³ s⁻¹; without advection the steady in-source scale is 8.6e-3 ug m⁻³ (`a_ref`) and 4.3e-2 (`a_max` = 5·`a_ref`).
- Plume-centreline anomaly 5 km downwind: 3.9e-5 to 1.9e-4 ug m⁻³ for `a_ref`, up to 2.9e-4 (u = 3) or 9.3e-4 (u = 0.5) for `a_max`.
- Background is 5 ug m⁻³. Noise σ = 5% of the median truth concentration is about 0.25 ug m⁻³ if "truth" includes background (§5 does not say whether it does).
So per-sample SNR is about 1e-3 to 4e-3. Sources would be unrecoverable at the 7-day window limit in most cells, and field RMSE for every method, persistence included, would sit at about the background level. The primary endpoint (10% RMSE reduction) would then be unresolvable or dominated by background handling. Fixing it means changing `a_ref`, H, σ, the noise definition (anomaly-relative?) or the background. All are scientific amendments. **Needs a human decision.**

**B2 — Protocol is still DRAFT and unapproved.** Its own header says its values bind only on human approval. The config defaults inherit these values, so nothing downstream is binding yet. No freeze commit or approval decision is in the ledger.

**B3 — Numerical gates may be unachievable as specified (risk, not shown).** At 64² (dx = 156 m): cell Péclet 9.4 (u = 3) to 15.6 (u = 5); first-order upwind numerical diffusion about 234–391 m² s⁻¹ vs the declared physical K = 50. The scheme would be solving a different, mostly numerical, diffusivity. Courant number at u = 5, dt = 300 s is 9.6, so the first-order backward-Euler time error may break the "dt-halving ≤ 1%" gate. Stop rule §12(1) would then halt the programme. Not settled by this audit; E01 must report it honestly. If the gates fail, loosening them is an amendment needing human approval, not an agent edit.

## 2. Other findings (non-blocking, need clarification or recording)

| # | Item | Issue |
|---|---|---|
| F1 | Noise σ (§5) | Ambiguous: "median truth concentration at the candidates" over which period/regime, with or without background; "fixed per scenario family" not defined. Must be pinned before any noise is generated. |
| F2 | Wind calendar (§7–8) | Test days 43–60 use held-out regimes (a)–(c), but the mapping of regimes to days/windows is unspecified. Estimators fitted on train-regime winds are evaluated on held-out winds with known wind input; the protocol should say how a regime change inside a 7-day window is treated. |
| F3 | Candidate lattices (§4, §6) | "12×12 source lattice" and "16×16 interior lattice of cell centres" are not defined geometrically (64 cells/16 = every 4th cell; "interior" offset undefined). Minimum separations (1.5 km, 1.25 km) are about 2 lattice steps, feasible, but exact coordinates must be fixed. |
| F4 | Primary hypothesis cells (§12) | N ≥ 8 gives 3×3×4 = 36 cells; 70% means ≥ 26 cells. Unspecified: how NOT RUN cells and snapshot-window cells (single sample) count, and that 20 seeds per cell make "CI entirely above 10%" cell-wise (no multiplicity control; the document labels the primary as the single confirmatory test, but 36 cell-wise CIs are used). |
| F5 | Placement without data | Placement is "train-period information only", yet the objective values (FIM, forecast skill) for the optimised networks need a defined wind/source assumption; for source-design objectives, are true source locations assumed known to the placer? State it. |
| F6 | Compute budget (§6) | 48 core-h cap vs the config's local profile (4 CPU × 2 h = 8 core-h per job). One 1488 h simulation is 17,856 implicit steps. No solver timing exists, so I cannot say whether the sweep fits; the budget check must be made after E01 timing, before any sweep. The per-source response matrix under time-varying seeded winds needs one run per source per wind calendar, which the protocol does not cost. |
| F7 | Config vs protocol coverage | `ExperimentConfig` does not encode the noise σ rule, AR(1) ρ = 0.8, bootstrap size 10,000, the 10%/70% thresholds, the E01 gates, or the 6 h warm-up (only in a metric text string). Preregistered thresholds therefore live only in prose and are not covered by the frozen tests or the config hash. |
| F8 | Truth/inference separation | Only advisory so far: `check_no_leak` relies on callers; `truth_id` hashes config + seed, not generated arrays (stated in the contract). No truth/inference API exists yet, so the estimator-level leakage test is untested. `truth_id` includes `winds` and `split`, so changing held-out regimes changes truth IDs (acceptable, but note it). |
| F9 | Quarto evidence | `reports/quarto/data/` is empty, `references.bib` is empty (0 bytes, though the literature map cites keys from it), and no render was run. Chapters have no results to show; fine for E00 but the bibliography must exist before any chapter cites. |
| F10 | Reference accuracy | `literature-replication.md` reports that the Green thesis was never read (scanned PDF, no renderer), and Bhardwaj/Choi/Manohar were read from abstract/excerpts only. Protocol §13 already says identifiability diagnostics are adaptations. Nothing in the protocol should be described as a reproduction. Reference accuracy is **unverified** (not audited by me beyond this record). |
| F11 | Bias/noise assumptions | Sensors ideal, noise white Gaussian independent of truth (so no concentration-dependent error), bias deferred to E14. Acceptable as a stated limitation, and must stay in scope text. |
| F12 | Planned ablations | The factor list covers K_s, N, window, noise, wind. AR(1) is a variant outside the primary 60 cells; diurnal sources and unknown background are secondary variants. No ablation changes an endpoint. Consistent. |

Arithmetic checked and consistent: 60 days = 1440 h; test 18 d = 432 h; minus 6 h = 426 issues; 3×5×4 = 60 cells; seed streams do not overlap for 20 replicates; the primary metric is single and has a declared window.

## 3. Decision log (no new human scientific approval is claimed)

| Status | Item |
|---|---|
| **Accepted for execution** (consistent with plan; non-scientific) | Typed config, seed streams, split/leak guard, provenance writer, report skeleton (13 tests pass). Local-only execution. Preregistered primary/secondary endpoint structure, windows and splits as written, *conditional on B2*. |
| **Deferred** (resolve before the stage that needs it) | F1 noise σ definition; F2 wind-day mapping; F3 lattice geometry; F4 counting rules; F5 placement assumptions; F6 budget check after E01 timing; F7 config coverage of thresholds; F9 bibliography; F10 primary-source verification of Green/Bhardwaj/Choi/Manohar. |
| **Blocked** (human scientific decision required) | B1 signal/noise scale; B2 protocol approval and freeze; B3 achievability of numerical gates (blocked from being relaxed by agents; E01 reports the result either way). |

## 4. Numerical oracle tests: NOT frozen

The task asks to freeze numerical oracle tests through Research OS. I did not, for two reasons. This task's only permitted output is this document, and adding test files would exceed it. And any oracle's tolerances and scales depend on B1/B3, which are open. Writing them now would risk fixing values the human may amend. Proposed oracles for E01, to be frozen once B1–B3 are resolved:
1. Manufactured solution (smooth analytic `c`, forcing derived symbolically) on 32²/64²/128²: observed L2 order ≥ 0.9 (protocol gate).
2. Pure decay with u = 0, K → 0 on a uniform field: `c(t) = c0·exp(−λt)` to the discretisation order in dt.
3. Global mass balance: domain integral change = source input − removal − boundary flux, relative residual ≤ 1e-3 (protocol gate), with the protocol's exact boundary rules.
4. Linearity/superposition: response to `a₁φ₁ + a₂φ₂` equals the sum of responses (needed by the NNLS Jacobian).
5. Nonnegativity of `c` for nonnegative sources and background.
6. Estimator leakage: perturbing observations after the issue time leaves the forecast bit-identical.
7. Unit check: a source of total rate `a` over time T adds `aT/(H·area)` mean concentration in a closed no-decay box.

## 5. Handoff for downstream experiments

- Verified commits so far on this branch: T-001 `e67b630`, T-002 `2947bf0`, T-003 `4afe93c` (T-003 is the only code; verified by its 13 frozen tests, which say nothing about scientific adequacy). This task adds this document only.
- Downstream needs: approval/amendment of the protocol (B1–B3); noise σ and lattice definitions (F1, F3); thresholds in config (F7); bibliography (F9); solver timing before any sweep (F6).
- Not done: any independent promotion review (required, must be a different agent profile); the ponytail skills are not installed, so built-in engineering standards were used.
