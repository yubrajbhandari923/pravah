# Gate 01 — Independent numerical verifier gate for the forward solver (T-008)

**Verdict: CONDITIONALLY CLEARED for the three convergence oracles and three discrete-exact oracles on the closed-box/periodic verification set. NOT CLEARED as a statement about the `study` boundary conditions, the preregistered §3 numerical gates as literally written, or any physical claim.** This is an engineering audit. It is not a scientific approval and not a finding review (F-001 still needs a separate promotion reviewer on a different agent profile). I changed no code, frozen test, metric or claim.

**Reviewer:** claude-balanced / claude-sonnet-5-5. The code and tests were written by earlier tasks (T-005 doc/oracles, T-006 `src/pravah/forward.py`, T-007 `scripts/e01_verification.py`) under the same model family. Independence here is by task and by re-derivation, not by a different model. The mandatory independent promotion reviewer is still outstanding.

**Scope read:** `docs/research/pde-and-discretization.md`, `src/pravah/forward.py`, `tests/test_forward_oracles.py`, `tests/test_forward_verification.py`, `scripts/e01_verification.py`, `reports/quarto/tables/e01-verification.csv`, `reports/quarto/chapters/01-forward.qmd`, `preregistered-protocol.md` §3/§12, gate-00. I did not open the PNG and I did not render the Quarto chapter.

**Provenance.** One re-run, outside the run ledger: `PYTHONPATH=src <project .venv>/python -m pytest -q tests` gives **36 passed, 0 skipped** (47 s). It was not registered as a run, because EXP-002 is at its 5-run budget (synthesis required) and nothing new was computed. The system `python3` cannot import `pravah` or FiPy, so the run was repeated with the project venv. All numbers below are read from `e01-verification.csv`, the table written by RUN-0006 (the run recorded in F-001). I did not re-execute `scripts/e01_verification.py`.

## 1. PDE versus source code

| Doc item | Code (`forward.py`) | Verdict |
|---|---|---|
| Eq. (1): transient + upwind convection = diffusion − λ(c − c_bg) + source | `TransientTerm + UpwindConvectionTerm == DiffusionTerm(K) − ImplicitSourceTerm(λ+outrate) + λ c_bg + S` | Matches. The loss acts on the excess only (the unapproved protocol [DECIDE] item, doc §1, is implemented as written). |
| Backward Euler, direct solve (doc §5.4) | `eq.solve(dt)` per step with `LinearLUSolver` | Matches. |
| `noflux`: zero advective and diffusive flux on walls | Advective coefficient zeroed on exterior faces; FiPy default Neumann gives zero diffusive flux. | Matches. Closed-box mass is conserved to about 1e-13 (CSV `grid` rows). |
| `periodic` | `PeriodicGrid2D` | Matches. |
| `study`: inflow Dirichlet c_bg, outflow upwind free | Inflow exterior faces constrained to c_bg; outflow added as an implicit loss rate (u·n)/cell-width because FiPy drops the upwind flux on unconstrained exterior faces. | Plausible and consistent with doc §3. Tested only by (v) a uniform field (1e-8) and by a blob exiting through the downwind wall (mass < 1e-3 of the initial). There is no independent non-trivial `study` oracle (see §6). |
| Uniform u, non-uniform wind rejected (doc §2) | `np.shape(u) != (2,)` raises | Matches. |
| Source ≥ 0, scheme `upwind` only | Both validated | Matches. |
| Index convention c[i, j], FiPy index i + j·nx | `to_cells` and the reshape(ny, nx).T round trip | Consistent. The 16×8 contract test uses non-square shapes, so a transposition would show. |

I found no PDE-versus-code discrepancy.

## 2. Truth and oracle independence

- The references (`ref_diffusion_noflux`, `ref_advected_periodic`, and the recurrences for decay and mass) are closed-form functions of cell centres. They do not import `pravah.forward` or the inverse stack (Part A of the oracle tests passes without a solver). **Independent: yes.**
- `scripts/e01_verification.py` imports those same reference functions from the test module, so the report and the tests cannot diverge. No inverse-generated truth appears anywhere, and none exists yet.
- The discrete-exact oracles (iii, iv, v) are algebraic identities of the same backward-Euler scheme. They check that the implementation implements the stated recurrence (units, normalization, wall fluxes). They cannot detect an error in the *choice* of scheme or in the physics. This is a stated limit of the oracle, not a defect.
- **Weakness:** the Part A residual tests still use the *old* oracle (ii) regime (t = 3 h, σ₀ = 1.5 km) while Part B uses the revised regime (6 min, σ₀ = 2.5 km). The reference formula is the same, so this is harmless, but the revised regime itself has no Part A check.
- The oracle (ii) regime was revised (INT-003, human-approved) after a first regime failed, as disclosed in doc §5.3. I accept the disclosure, but note that this is a post-failure revision. Its derivation predates the solver run (it is in `oracle_ii_derivation.py`).

## 3. Dimensional consistency

The unit table in doc §1 and the code docstring agree: c g m⁻³, u m s⁻¹, K m² s⁻¹, λ s⁻¹, source g m⁻³ s⁻¹, θ g s⁻¹, b m⁻². The forcing is `theta*b/h` with b normalized on each grid (Σ b ΔA = 1), and the column mass is h·Σc·ΔA. `test_units_source_mass_rate` and `test_prescribed_mass_forcing_noflux` check this at 1e-8 relative on 16×8 and 32² grids. **No inconsistency found.** Concentrations in the tests are generic (5 and 5e-6 are both used as c_bg). The conversion to µg m⁻³ is not exercised by any test, and that conversion belongs to the later benchmark stage.

## 4. Convergence on three grids (16², 32², 64², constant Courant number)

From the CSV (`order` rows, finest pair 32→64; the floor from doc §5.3 is p − 0.2 = 0.8):

| Oracle | rel L2 16² / 32² / 64² | observed order 16→32 | observed order 32→64 | Floor 0.8 |
|---|---|---|---|---|
| diffusion (noflux) | 6.59e-3 / 2.10e-3 / 7.58e-4 | 1.65 | 1.47 | pass |
| advect K = 20 | 4.93e-2 / 2.59e-2 / 1.32e-2 | 0.93 | 0.97 | pass |
| advect K = 200 | 4.77e-2 / 2.52e-2 / 1.29e-2 | 0.92 | 0.97 | pass |

The errors decrease monotonically, the values are close to the a-priori prediction of the oracle (ii) derivation (about 0.049, 0.025 and 0.013, order 0.979), and the discrete-exact oracles reach round-off (about 1e-13 to 1e-15 mass error).

**Deviation from the preregistered gate (needs a human decision, not an agent edit).** `preregistered-protocol.md` §3 says the observed order must be **≥ 0.9 on the 32/64/128 grids** for the first-order scheme. What was run is 16/32/64 with a floor of 0.8. The project brief says 32² then 64², and the doc §5.3 floor of 0.8 is a stated a-priori allowance. These do not equal the literal protocol gate.
- The advection orders (0.967 to 0.969 at 32→64) would also clear 0.9, but **the 64→128 pair was never computed**, so the protocol gate as written is **unevaluated, not passed**.
- The protocol is still DRAFT (gate-00 B2). Whether the 0.8 floor and the 16–64 ladder replace the 0.9 floor and the 32–128 ladder is a human decision.

**Other preregistered §3 gates.**
- *Mass-balance residual ≤ 1e-3:* the closed-box maximum relative error against the discrete-exact recurrence is 2.0e-14. This is far below 1e-3 but it is a *closed-box, discrete-recurrence* check. It does not cover `study` boundary mass.
- *dt-halving changes the endpoint field by ≤ 1% relative L2:* **not evaluated.** The `dt` rows record the error to the reference, not the change between successive dt fields. For the translating Gaussian the error to the reference changes from 0.0259 (dt 60) to 0.0227 (dt 30), so the field change is at least 0.0032 of the reference norm. A direct measurement is needed. The primary endpoint is not the field in this oracle anyway, so this gate also needs a human reading of "primary-endpoint field".

## 5. Failure and negative cases (recorded, not hidden)

- **Run history on EXP-002:** RUN-0003 failed (a pytest run; I did not inspect its log, so I do not state its cause. The doc separately reports that the first oracle-(ii) regime was pre-asymptotic). T-007 attempt 1 was marked failed with `resource_limit` after RUN-0006 hit the 5-run budget. The ledger blocker "RUN-0003 failed — not yet reviewed" is still open.
- **Numerical diffusion is large at study-like settings** (CSV `cfl_peclet`, 32², K = 20, dt = 60 s): rel L2 is 0.025, 0.054, 0.124, 0.275 for U = 1, 2, 4, 8 m s⁻¹; peak ratio 0.97 → 0.73; cell Péclet 15.6 → 125. Doc §5.2 puts upwind plus backward Euler at about 37 × K at 3 m s⁻¹ on 32². This confirms the gate-00 B3 risk: the scheme converges to the right answer only with a numerical diffusivity far above the physical K, so `K` is not physically resolved at study resolution. **Convergence is not accuracy at 32² or 64².**
- **Positivity:** the domain minimum is positive in every advect row (min/max ≥ 1.4e-4 for U = 2 to 8 and larger in the other rows I read). This agrees with the M-matrix argument. It holds only for first-order upwind, and the doc states that other schemes would print and record any undershoot.
- No oracle failed in the final record. A failure of the *regime*, not the solver, occurred earlier and led to the revision. I cannot show from the ledger what the failing numbers were beyond the doc's account.

## 6. Gaps the tests do not cover (must be closed or accepted by the human)

1. **No independent oracle for the `study` boundary** beyond a uniform steady state and a blob leaving the domain. Inflow and outflow face handling for a non-uniform plume (the case that matters for sensors near the boundary) has no reference solution. The `study` mass budget is "diagnostic only" (doc §3), and no run records the boundary fluxes.
2. **Wind reversal / face reclassification (R3, gate-00 item 3)** is not tested. Doc §8 says e01 must add it. It is not in `test_forward_*`. Faces are classified once per `simulate` call, so time-varying wind requires a call per step, and that path is untested.
3. **Time-varying θ, the s_bg term, a non-square domain with `Lx ≠ Ly` under an oracle** (the contract test uses 16×8, but only for shape, conservation and units), and any scheme other than upwind.
4. The 64→128 grid pair and the literal protocol gates (see §4).
5. No mutation test of the *solver*. The doc reports one mutation for the references (4 of 8 self-checks failed). I did not run a solver mutation (no run budget left in EXP-002). A reviewer could break the outflow rate or the inflow constraint and confirm that the tests fail.

## 7. No unsupported physical claims

- Everything is a simulated passive tracer. The chapter, doc and F-001 say so and make no PM2.5 or exposure claim. F-001 is *preliminary*, and I did not change its status. `01-forward.qmd` is a data-driven chapter reading from the CSV. It contains no typed results or run identifiers in its prose. It has not been rendered with Quarto and `_quarto.yml` still lists `e01-forward.qmd` (T-007 caveat, unchanged).
- Sentences to keep out of any downstream text: "solver verified against physics" (it is verified against its own scheme and three analytic Gaussians), "K = 50 m² s⁻¹ is resolved" (numerical diffusion is larger), "real winds" (uniform u only).

## 8. Frozen tests

The contract asked for the verification tests to be frozen. I request freezing `tests/test_forward_oracles.py` and `tests/test_forward_verification.py` as they are (36 tests, no edits). Freezing records their hashes. It does **not** approve the scheme (protocol §12.1 option decision) or the 0.8 floor. Unfreezing for any physics-assumption change (loss on excess, the oracle (ii) regime, the floor, the scheme) requires human approval.

## 9. Handoff for downstream experiments

- **Verified commits on this branch:** `58f60d2` (T-005 PDE and oracles), `180a3ea` (T-017 oracle (ii) regime), `f1f640f` (T-007 figures, CSV and chapter). `src/pravah/forward.py` (T-006) is included in `180a3ea`. "Verified" here means the 36 tests pass and the code matches the doc. It does not mean the science is validated.
- **Downstream needs:** (a) human decision on the 0.9/32–128 versus 0.8/16–64 gate (and on whether dt-halving is to be measured on the endpoint field); (b) a `study`-BC oracle and the R3 face-switch test before E02 truth generation; (c) a human decision on the scheme (§12.1) given the numerical-diffusion numbers above; (d) the loss-on-excess form and λ values; (e) EXP-002 synthesis (budget reached) and independent review of F-001; (f) the protocol approval (gate-00 B2) and the signal-to-noise concern B1 still stand, and this gate does not address them; (g) a documented way to run the tests (project venv with FiPy; the system python and the `RESEARCH_PYTHON` interpreter cannot).
- **Not done:** the Quarto render, the 128² grid, solver mutation tests, an independent promotion review.
