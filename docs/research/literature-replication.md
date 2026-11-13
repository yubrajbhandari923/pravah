# PRAVAH: literature replication targets, baselines and reuse policy

**Task:** T-001 (attempt 1) · **Experiment:** EXP-001 · **Date:** 2026-11-05 · **Agent:** claude-deep / claude-opus-5-5

## Attempt 4 update (2026-11-10, claude-balanced / claude-sonnet-5-5): web sources checked, Green thesis still unread

Web fetch now works. **Green 2006 is still unread**: the file reader again returned `pdftoppm is not installed`, so §1 is
unchanged and fully unverified. Checked this attempt (page text fetched 2026-11-10; "Verified" below means *the quoted item
was seen on the fetched page*, not that anything was reproduced):

- Bhardwaj & Subramanian: arXiv abstract page and v1 HTML (excerpts only, not the whole paper). §2 updated.
- Choi & Hummel: IOPscience article page (abstract, §3.2-3.3 prose; equations 3-6 did not render, so their exact form is not
  recorded). §3 updated.
- Manohar et al.: arXiv abstract page only (no equations seen). §4 updated.
- Licences/versions of PySensors, DAPPER, FiPy, FilterPy, BLDFM from GitHub and PyPI pages. §8 updated.
- **Not done:** DAPPER JOSS paper, PySensors/Manohar equations, Green thesis, full Bhardwaj/Choi text and supplements, and
  any repository LICENSE file other than FiPy's. No code was run and no run was registered: this task is a literature audit
  with no numerical result.

## Attempt 2 update (2026-11-09, claude-balanced / claude-sonnet-5-5): still no primary source verified

Copied this draft, the Green reading notes and the literature map from the main checkout into the experiment worktree
(they were untracked on main). Then tried again to reach the sources. Every route failed, so **no "Verified" cell below
has changed and every row stays unverified**:

- Green PDF: the file reader returned `pdftoppm is not installed` (poppler-utils missing), so no page could be rendered.
  Reading it by absolute path from main does not help; the blocker is the renderer, not the path.
- `WebFetch`, the Parallel `web_fetch` tool and the reference-lookup tools each returned "permission not granted" in this
  session. Nothing was fetched for Bhardwaj 2026, Choi 2026, Manohar 2018, PySensors, DAPPER, BLDFM, FiPy or FilterPy,
  so licences and versions are still unchecked.

Needed from the human: install `poppler-utils` (or allow another page renderer), and grant the web-fetch and
reference-lookup permissions to this agent profile in the session that actually runs. Then re-dispatch.

## Attempt 3 update (2026-11-10, claude-balanced / claude-sonnet-5-5): same blockers, nothing verified

Re-tested each tool once; all still refused, so every row stays unverified and no content was extended:

- Read of the Green PDF (pages 10-12, absolute path in main): `pdftoppm is not installed`.
- `mcp__plugin_research-desk_reference-lookup__lookup_reference`: permission not granted.
- `mcp__plugin_parallel_parallel-search__web_fetch`: permission not granted.
- `WebFetch`: permission not granted.

Stopped per the brief (no workaround attempted). The human's fix in INT-002 has not taken effect in this session.

## Status (attempt 1, retained): INCOMPLETE. No primary source was read

This attempt **could not open any primary source**:

- `docs/papers/Green_2006_Optimal_Sensor_Placement.pdf` is a scanned image PDF with no text layer. The environment has no
  `pdftoppm`/`pdftotext`, and running Python (PyMuPDF/OCR) required approval that a headless session cannot get.
- Web fetch and the reference-lookup tools (arXiv, DOI, GitHub) were denied, so no paper or repository page could be retrieved.

What follows comes **only from in-repository secondary text**: `docs/research/phase1-proposal.md`,
`docs/research/green-2006-reading-notes.md`, `docs/research/literature-map.md` and `reports/quarto/references.bib`.
None of these texts says it was checked against a primary source. Nothing below is a verified reproduction target.
Each row says what still has to be checked. **Do not cite any equation, number, version or license below as verified
until the "Verified" column says so with page/section and date.**

Legend: **Paper result** = what the source claims (to be confirmed). **PRAVAH adaptation** = what PRAVAH would do
differently. That is a project design choice, not a paper result.

## 1. Green (2006): Rice MSc thesis (`green2006`)

| Item | Secondary-source content (unverified) | Where to verify (PDF page, per reading notes) | Verified |
|---|---|---|---|
| Problem | Estimate parameters (diffusivity, advection coefficient) of a semidiscretized 1D advection–diffusion PDE from sensor time series; place sensors in advance to tighten the parameter estimates. Not field reconstruction, not many unknown sources. | pp. 10–20 (Ch. 2.2–2.3) | no |
| Inverse | Weighted least squares; under Gaussian noise this equals maximum likelihood (Appendix A). | pp. 21–29, 56–59 | no |
| Fisher information | `M(S) = G(S)^T R^{-1} G(S)`, `G = ∂y/∂θ`, covariance independent of θ; independent time samples add information; local covariance ≈ `M^{-1}`. | pp. 21–29, App. A | no |
| Criteria | D-opt: max `log det M`. A-opt: min `tr M^{-1}`. E-type: worst-informed direction. Singular `M` needs care. | pp. 34–46 (Ch. 3) | no |
| Design algorithm | Approximate (weighted) design over discrete candidate locations, nonnegative weights summing to 1. | pp. 34–46 | no |
| Numerical example | Optimized locations ≈ x = 0.025, 0.975 on [0,1]. 95 % confidence-ellipse areas: center ≈ 1.91e-3, equal spacing ≈ 4.66e-4, optimized ≈ 2.97e-4. | p. 54 (printed p. 45), pp. 47–54 | no |
| Missing entirely | Exact PDE coefficients, true θ, noise σ, time grid, FE discretization, number of sensors, weight-update algorithm and stopping rule. | Ch. 2.3, 3 | not extracted |

**Reproduction target (proposed, conditional on verification):** rebuild the 1D example and report the three ellipse
areas plus the optimized locations, with a tolerance set only *after* the exact settings are extracted. Do not invent
a precision. If the settings cannot be fully recovered, call it a qualitative reproduction.

**PRAVAH adaptation:** θ = nonnegative strengths of 2–8 known 2D source bases. `G` = the source-to-sensor response
matrix from one FiPy basis-source run per coefficient (linear in θ, so the FIM is exact, not local). Use greedy
discrete D-/A-opt with a prior or regularization when `M` is singular. This is a different problem from Green's
nonlinear parameter identification, so mark it "adapted from Green 2006 §… " every time.

## 2. Bhardwaj & Subramanian (2026, arXiv:2608.00050, preprint) (`bhardwaj2026`)

| Item | Secondary-source content (unverified) | Verified |
|---|---|---|
| Claim | Wind-conditioned source–sensor fingerprints. Background variation is separated or projected out. A good concentration fit does not prove that individual sources are identifiable. | yes (abstract, arXiv v1, 2026-11-10) |
| Metadata | arXiv:2608.00050 [eess.SP], v1 submitted 25 Jul 2026; text licence CC BY 4.0. Preprint. | yes |
| Model | `Ỹ = H̃_Φ c + Ẽ`, `H̃_Φ = P_Q^⊥ H_Φ^lag ∈ R^{N×J}`, `J = K·B` (sources × temporal-basis functions); `c ≥ 0`. `P_Q^⊥ X = X − U_r(U_rᵀX)` from a thin SVD of the background basis Q. | yes (§3-4, App. D, eq. 15) |
| Identifiability | Def. 4.1: `H̃c₁ = H̃c₂ ⇒ c₁ = c₂`. Prop. 4.2: identifiable for all `c ∈ R₊^J` iff `rank(H̃_Φ) = J`. Prop. 4.3: robustness via `‖H̃†‖₂ = 1/σ_J`. Primary score is the padded `σ_J` (rank-deficient ⇒ 0); `κ = σ₁/σ_J` (∞ if deficient). | yes |
| Diagnostics | numerical rank, `σ_J`, κ, effective rank `r_eff(τ_σ)`, visibility `v_j = ‖h̃_j‖₂`, background absorption, pairwise coherence ρ_ij, ray distance. Thresholds predeclared, "never tuned on source recovery"; defaults `τ_num = max(N,J)·ε·σ₁`, `τ_σ = σ_e√N`. | yes (App. B, G) |
| Estimator | Nonnegative least squares by projected FISTA (App. N). | yes (named only) |
| Sanity gates | App. M S1: 16×16 grid, four compact sources, four sensors, seed 123. | yes (named only) |
| Paper's numbers | Controlled conditioning: separated σ_J 3.27, κ 8.89; close σ_J 9.60, κ 2.26 (2 sources). New Delhi weeks 1-4 σ_J 3.71, 5.67, 10.15, 7.65. These are the paper's own units/setups, not PRAVAH targets. | yes (App. I.1, I.9, from fetched excerpt) |
| Not extracted | Full text beyond excerpts: transport kernel details (App. D eq. 13 only seen as a formula head), FISTA settings, thresholds table values, code and its licence (no code link seen on the fetched pages). | unverified |

**PRAVAH adaptation:** Stage 1 rank/singular-spectrum/coherence diagnostics and a deliberately non-identifiable
twin-source case. PRAVAH's response matrix comes from FiPy basis-source runs, not the paper's Gaussian-puff lagged kernel
and not its New Delhi data. The paper's identifiability statement is a standard rank result for a linear model; what is
paper-specific is the diagnostic panel and reporting protocol. Preprint status: cite as a preprint. Its claims are not
peer-reviewed evidence.

## 3. Choi & Hummel (2026, ERL 21, 014004) (`choi2026`)

| Item | Secondary-source content (unverified) | Verified |
|---|---|---|
| Claim | Stochastic advection–diffusion (SAD) + multimodal data fusion + VoI objective for low-cost PM2.5 network design. Different objectives give different networks. | yes (abstract, discussion; IOPscience 2026-11-10) |
| Metadata | Choi & Hummel, Environ. Res. Lett. 21, 014004; DOI 10.1088/1748-9326/ae2b89; published 2 Jan 2026; CC BY 4.0. | yes |
| VoI | Reduction in expected loss: prior vs posterior expected cost `C(ξ, A)` (ξ = PM2.5, A = action) for a selected set γ; optimum γ* maximizes net VoI. Equations 3-6 did not render, so exact forms are not recorded. | partly |
| Loss | AQI-category misclassification penalty, symmetric, linear in category deviation (0 to 4). | yes |
| Optimizer | Forward greedy by marginal VoI; 40 low-cost sensors, 100 synthetic measurement scenarios per location per round; 1 km grid, Corpus Christi Bay, Texas. | yes |
| Data | Open data URLs listed (SVI, NSI, EPA, GP-AQM, NCAR). No code repository seen on the fetched page. | yes / code unverified |
| Not extracted | Supplement (loss details, S3), SAD model equations, fusion model, uncertainty of results. | unverified |

**PRAVAH adaptation:** motivates the cross-objective evaluation table (proposal §6). A VoI objective is **not** in
the Phase-1 baseline set unless the human approves it.

## 4. Manohar et al. (2018) and PySensors (`manohar2018`, `desilva2021`, `karnik2026`)

| Item | Secondary-source content (unverified) | Verified |
|---|---|---|
| Manohar paper | IEEE Control Syst. Mag. 38 (2018) 63-86, DOI 10.1109/MCS.2018.2810460, arXiv:1701.07569 (v2, 34 pp.). Abstract: sensors found by SVD and QR pivoting on a tailored library from training data; contrasted with compressed sensing. | yes (abstract page only) |
| Method | Learn a POD (SVD) basis Ψ_r from training snapshots. Pick sensors by QR column pivoting on Ψ_rᵀ (oversampled p > r by pivoting on Ψ_r Ψ_rᵀ). Reconstruct with least squares `x̂ = Ψ_r (CΨ_r)^† y`. Pivoting greedily conditions `CΨ_r`. | no (standard method from memory of the paper; equations/notation not seen this attempt) |
| PySensors | scikit-learn-style package; `SSPOR` (reconstruction) and `SSPOC` (classification); QR optimizer plus cost-constrained (CCQR) and general constrained QR. GitHub README cites Manohar 2018 for reconstruction. MIT licence (GitHub and PyPI metadata). JOSS 2021 (de Silva et al., 6(58) 2828) and JOSS 2026 (Karnik et al., 11(118) 9265, "PySensors 2.0"). | yes (README, PyPI) |

**PRAVAH adaptation:** a minimal in-house QR selector, cross-checked against a pinned PySensors release on the same
snapshot matrix (sensor sets should match exactly when there are no ties). Snapshots come only from training scenarios.

## 5. DAPPER (Raanes, Chen & Grudzien 2024, JOSS 9(94) 5150) (`raanes2024`)

| Item | Secondary-source content (unverified) | Verified |
|---|---|---|
| Use | Twin-experiment benchmarking: synthetic truth from a dynamic + observation model, estimated from noisy observations; templates for DA methods with a battery of statistics; diagnostics averaged over subdomains then time, with autocorrelation-aware confidence intervals. | yes (GitHub README, 2026-11-10) |
| Version/licence | PyPI release page shows 1.8.1 (latest seen), MIT per PyPI metadata (licence file `LICENCE.txt`). README asks for citation of the JOSS paper (doi 10.21105/joss.05150) with the version used. Requires Python ≥ 3.12 per README; PyPI release 1.7.3 also lists ≥ 3.12. The raw `LICENSE.txt` URL returned 404, so the licence text itself was not read. | partly |
| Not extracted | The JOSS paper and code: exact diagnostic definitions (RMSE vs spread, forecast/analysis/smoother split, burn-in handling). | unverified |

**PRAVAH adaptation:** report pre-update (forecast), post-update (analysis) and offline-smoother statistics
separately. Use RMSE-vs-spread as one calibration check. Validate small linear filters against FilterPy first.

## 6. Other mapped papers (not opened, no extraction attempted)

`pennington2024`, `schlutow2026`, `mons2017`, `zhou2022`, `mano2022`, `krause2008`, `bartos2021`, `roy2018`,
`enkf2009`, `sharma2019`, `cheng2022`, `webster2022`, `mues2018`, `patel2026`. See `literature-map.md` for their
intended roles. The bib entries for `pennington2024`, `bartos2021`, `mues2018`, `roy2018` and `enkf2009` have
incomplete author lists, which must be fixed before submission.

## 7. Selected Phase-1 baseline algorithms (project choice from the proposal, not a paper result)

| Stage | Baseline | Reference | Implementation policy |
|---|---|---|---|
| Forward | FiPy finite-volume 2D advection–diffusion–removal | — | dependency (only forward backend) |
| Inverse | OLS (when identifiable), NNLS, Tikhonov LS, Gaussian-prior Bayesian | Green App. A (WLS/ML) | in-house, small |
| Placement | random (seeded), uniform grid, concentration hotspot, footprint diversity, greedy D-/A-opt, POD/QR | Green Ch. 3; Manohar 2018 | in-house; QR cross-checked vs PySensors |
| Sequential | open-loop, latest-snapshot, windowed batch, linear/reduced KF, then small EnKF | FilterPy, DAPPER | in-house; validated vs references |
| Deferred | GP mutual-information (Krause 2008), gsm_aqs gradient MI, VoI | — | requires human approval |

## 8. Code reuse and license policy

Checked 2026-11-10 from GitHub and PyPI pages. "Latest seen" is the newest version visible in the fetched PyPI data, not a
chosen pin: the human or the Stage-1 environment task must pin versions and record hashes.

| Upstream | Role | Licence (checked) | Latest version seen / pin |
|---|---|---|---|
| FiPy (https://github.com/usnistgov/fipy) | dependency | NIST "Terms of Use" (LICENSE.rst read): US-government work, free use/copy/modify/distribute if the notice and warranty disclaimer are kept; may be subject to foreign copyright. Not the same as a standard OSI licence: keep the notice. | 4.0.3 (PyPI, 2026-06-18); **not pinned** |
| PySensors (https://github.com/dynamicslab/pysensors) | cross-check | MIT (GitHub, PyPI metadata) | PyPI list shows 0.4.3 as the newest entry seen (0.4.2 was 2026-01-28); the 2.0 paper exists, so confirm the actual latest. **not pinned** |
| FilterPy (https://github.com/rlabbe/filterpy) | cross-check | MIT (README text read) | PyPI lists 2.0.0 uploaded 2026-10-26 (README still says 1.x and a planned 2.0 fork). Confirm before pinning. **not pinned** |
| DAPPER (https://github.com/nansencenter/DAPPER) | cross-check / diagnostics | MIT per PyPI metadata; licence file not read | 1.8.1 (PyPI release page); **not pinned** |
| BLDFM (https://github.com/SchlutowSM2Group/BLDFM; Zenodo 10.5281/zenodo.15487243) | conceptual / validation reference only | PolyForm Noncommercial 1.0.0 (README says so; LICENSE file not read) | GitHub release v1.1 (6 Oct 2025); **do not copy or vendor**. Zenodo record not opened. |
| gsm_aqs (https://github.com/sustainability-lab/gsm_aqs) | optional comparator | not stated | deferred |

Rules: depend on packages through pinned versions, never vendor them. Keep license notices. Mark each algorithm as
"in-house", "adapted from X" or "reused from X". No BLDFM code goes into PRAVAH unless the human gives separate
approval. The Green PDF stays out of git (`docs/papers/.gitignore`) until redistribution rights are checked.

## 9. What is needed to finish this task

1. **Still blocking:** a way to read the Green PDF: install `poppler-utils` (for page rendering) or allow an OCR/PyMuPDF step.
   Then extract Ch. 2.2–2.3, Ch. 3, App. A and the full numerical-example settings into §1, with page numbers.
2. Web access now works. Remaining: Manohar equations (arXiv PDF), DAPPER JOSS paper, Bhardwaj full text/App. N and
   thresholds, Choi equations 3-6 and supplement. Fill the "unverified" rows.
3. Read the DAPPER, PySensors, FilterPy and BLDFM LICENSE files, then pin a release tag/commit and hashes in §8.
4. Only then set the Green reproduction tolerance and freeze it in the preregistered protocol (EXP-001).
