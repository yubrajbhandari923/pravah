# B3 — advection scheme: upwind vs FiPy VanLeer (option (a), human-approved)

**Branch** `experiment/e01-advection-scheme`. Exploratory engineering study. **No protocol value, physical parameter, gate or
tolerance was changed.** Evidence: `docs/research/gates/b3/` (`b3_scheme_study.py`, `cost_64.py`). CSVs are in
`vanleer_unsplit/` (FiPy's term as shipped) and `vanleer_split/` (the version now in `forward.py`).

## What FiPy offers

FiPy 4.0.3 has one TVD/higher-order convection term, `VanLeerConvectionTerm`. The others (`Central`, `Exponential`,
`Hybrid`, `PowerLaw`) are either non-monotone at cell Péclet > 2 or reduce to upwind at the study Péclet of 9–16.
`VanLeerConvectionTerm` subclasses `ExplicitUpwindConvectionTerm`. That makes it:
1. **Explicit in time.** It is unstable for Courant > 1. At protocol dt = 300 s on 64² it blew up: peak ×1e8 within 1 h,
   at u = 3 m/s and C = 5.8. `simulate` now rejects vanleer when per-axis Courant > 1.
2. **Nonlinear in c**, because of the limiter. It is positively homogeneous (f(3c) = 3f(c) to 1e-15) but **not
   additive**. The superposition error is 1.0% (split) or 3.3% (unsplit) on a two-blob test at 32², 1 h.
3. **Missing the cross term when unsplit.** The per-face (h − u_n dt) correction cancels forward Euler's
   anti-diffusion on each axis but leaves an off-diagonal −u_x u_y dt/2. Along an oblique wind this gives
   K_eff = K − u_x u_y dt e_x e_y. For a 3 m/s diagonal wind at 64², dt 30 s, the fit gives **K_eff,along = −16.3 m²/s**
   against a predicted −17.5, and K_eff,cross = 118 against 117.5. The along-wind diffusion is anti-diffusive. At
   32², dt 60 s the field undershoots to min/max = −0.073 (`vanleer_unsplit/puff_k_eff.csv`).

**Fix used:** Lie dimensional splitting with the same built-in term. Each step solves x advection, then y advection
together with diffusion, removal, source and the y-wall outflow. Upwind keeps its single equation and is
**bit-identical** to `main` on study, noflux and periodic runs (checked directly and by `test_default_is_upwind_bit_identical`).

## Results (64² unless stated; periodic Gaussian puff σ0 = 1 km, K = 50, T = 1 h; full-covariance fit)

K_eff (m²/s) along / across the wind, and rel. L2 error against the analytic solution:

| wind | scheme | dt 300 | dt 150 | dt 60 | dt 30 | dt 15 |
|---|---|---|---|---|---|---|
| 3 m/s x | upwind | 1371 / 49 (0.70) | 906 (0.60) | 546 (0.47) | 417 (0.40) | 351 (0.36) |
| 5 m/s x | upwind | 2933 (0.78) | 2039 (0.75) | 1156 (0.64) | 807 (0.56) | 625 (0.50) |
| 3 m/s diag | upwind | 1407 / 208 (0.72) | 855 / 212 | 478 / 214 | 348 / 215 (0.46) | 282 / 215 (0.42) |
| 3 m/s x | vanleer split | rejected (C>1) | rejected | rejected | 50.9 / 49.6 (0.0033) | 51.8 / 49.5 (0.0072) |
| 5 m/s x | vanleer split | rejected | rejected | rejected | 49.9 / 49.7 (0.0014) | 52.1 / 49.5 (0.0068) |
| 3 m/s diag | vanleer split | rejected | rejected | 50.0 / 50.0 (0.0022) | 50.7 / 50.7 (0.0047) | 51.2 / 51.2 (0.0096) |

- **Upwind K_eff matches its modified equation**, K + |u_i|h/2 + |u|²dt/2, to within 2% when Courant ≤ 1 (e.g. 417
  measured vs 419 predicted). At dt = 300 s the backward-Euler time term u²dt/2 dominates: 1350 of the predicted
  1634 m²/s at 3 m/s. The leading-order formula over-predicts there (measured 1371). Spatial upwinding (u h/2 = 234)
  remains even as dt → 0. **At 64² both error sources are 5–60× the physical K**, and 128² does not fix this
  (dt 15 s: 234 m²/s, rel L2 0.27).
- **Split VanLeer recovers K = 50 within 2–4%** at 64² once dt ≤ 30 s, and within 0.2% at 128² (`vanleer_split/puff_k_eff.csv`).
  At 16² and 32² it still over-diffuses (88–200 and 57–69 m²/s), because σ0 = 1 km is only 1.6 to 3 cells wide there.
  Its error *rises* slightly as dt falls at a fixed grid (0.0033 → 0.0072). This is expected for a MUSCL-Hancock-type
  scheme, whose correction is exact at C = 1, so do not read the dt-refinement column as a pure time error.

**Oracle convergence** (e01 oracles, constant Courant 0.58, `oracle_convergence.csv`). The rel. L2 error and observed
order for the steps 16→32, 32→64 and 64→128:
- upwind advect K20: 0.0493 / 0.0259 / 0.0132 / 0.0067; orders 0.93 / 0.97 / **0.99**.
- vanleer advect K20: 0.0065 / 0.0016 / 0.0004 / 7e-5; orders 2.02 / 2.05 / 2.44.
- K200 behaves the same way (vanleer orders 2.6–2.7). The diffusion oracle is identical for both schemes because u = 0.
- The 64→128 pair is now computed, so the literal protocol gate (order ≥ 0.9 on 32/64/128) is **met by both schemes**.

**dt-halving gate** (protocol §3: endpoint field change ≤ 1% rel. L2), measured on the puff at 64² (`dt_halving_64.csv`):
- upwind: 300→150 changes the field by 18.5% (3 m/s), 6.7% (5 m/s) and 22% (diag); even 30→15 changes it by 5.7–10%. **It fails at every dt tried.**
- split vanleer: 30→15 changes it by 0.62%, 0.71% and 0.59%; 60→30 on the diagonal by 0.54%. **It passes.**

**Mass and positivity:**
- Closed-box mass with a source matches θt to 1e-15.
- Periodic puff mass is conserved to below 1e-13 for every stable run.
- Study-BC uniform background is exact. A blob leaving the domain loses mass monotonically. A restart in 10-step
  chunks is bit-identical to one long run.
- No negative values for split vanleer in any run (min/max ≥ 0). The only nonzero mass error, 3e-9, is the run with
  the guard deliberately switched off.

**Cost** (`vanleer_split/cost_64.csv`; serial, idle machine, 64², study BC). The last column is one 1488 h protocol
simulation.

| scheme, dt | s per simulated hour | h per 1488 h run |
|---|---|---|
| upwind, 300 s | 0.46 | 0.19 |
| upwind, 30 s | 4.1 | 1.7 |
| split vanleer, 30 s | 7.8 | 3.2 |
| split vanleer, 15 s | 16.1 | 6.6 |

Vanleer costs about 17× upwind at dt 300 s: 10× from the smaller dt and 2× from the two solves per step.

## Recommendation

- **For 64² truth generation, use `scheme="vanleer"` (split) with an internal step dt ≤ h / max|u_i|.** That is 30 s
  for winds up to 5 m/s. With the truth-wind speed factor 1.2 (6 m/s) use 25 s. With it, K = 50 is actually resolved
  and the convergence, mass and dt-halving gates are all met on these tests.
- **Upwind cannot meet B3 at 64²** at any dt. Spatial upwinding alone is ≥ 4.7× K.
- **Human decision needed (protocol reading, not an agent edit):** `dt_s: 300` cannot be the solver step under
  vanleer. Keeping 300 s as the output and observation cadence, with 10–12 substeps inside it, seems the natural
  reading, but it changes compute about 17× (F6 budget: 3.2 h per 1488 h run per wind calendar, before any per-source
  response runs).
- **Human decision needed (consequential):** under vanleer the forward map is not linear. The NNLS, Fisher and
  footprint code (`sensitivity.py`, `information.py`, `inverse.py`) builds per-source columns and sums them. Options:
  - (i) vanleer for truth and upwind columns for the estimator. This is honest model mismatch, but the estimator then
    sees K_eff ≈ 400–1400.
  - (ii) vanleer columns, accepting a ~1% superposition error.
  - (iii) a linear higher-order alternative, such as an unlimited Lax-Wendroff/QUICK. This is not in FiPy as an
    implicit term and was not tried.
  - Note: the limiter is scale-invariant, so a single-source column scales exactly.

## Unresolved

- No independent `study`-BC oracle for vanleer (the same gap gate-01 records for upwind). Only checks: uniform-state
  exactness, blob exit, positivity.
- Lie splitting is first order in time. x and y advection commute for uniform wind on a periodic box but not exactly
  with walls. Strang splitting was not tried.
- Wind changes per call: the Courant check runs per call, so time-varying-wind drivers must choose dt from the
  calendar's maximum speed.
- The superposition error was measured on one case only. Its size on the benchmark's source footprints is not
  quantified.
- Puff tests use periodic BCs and one initial width. Effects on the primary endpoint (placement and forecast skill)
  are untested.
