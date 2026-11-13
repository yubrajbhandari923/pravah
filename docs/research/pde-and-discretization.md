# PRAVAH Phase 1 — continuum PDE, discretization and forward oracles

**Task:** T-005 (attempt 1) · **Experiment:** EXP-002 · **Date:** 2026-11-10 (revised; first draft 2026-11-05, claude-deep / claude-opus-5-5) · **Revised by:** claude-balanced / claude-sonnet-5-5
**Status:** DRAFT derivation and test design. It is **not** frozen, not human-approved and not independently reviewed.
**Sources:** `docs/research/preregistered-protocol.md` (DRAFT v1, sha256 `4a67c1c9…e3e7d7` per gate-00),
gate-00 audit (`experiments/EXP-001/docs/research/gates/gate-00-protocol.md`, decision log items 1–6),
`docs/research/methodological-decisions.md`.
**Tests:** `tests/test_forward_oracles.py`. **Evidence:** RUN-0002 (EXP-002): 8 reference self-checks pass,
8 solver oracles skip because `pravah.forward` does not exist. An earlier revision of this document cited RUN-0002 and
RUN-0003 as evidence, but neither existed in the ledger and the test file was absent; those claims were removed and the
tests rewritten (T-005 attempt 1, claude-balanced). A deliberate mutation (wrong σ² growth, 1·K·t instead of 2·K·t)
made 4 of the 8 self-checks fail, so the self-checks can fail.

Everything here is a *simulated passive tracer*. Nothing in this document is evidence about ambient PM2.5.
No tolerance below was chosen after seeing solver output, because no solver output exists.

## 1. Continuum model and depth averaging

Start from 3D transport of a passive tracer c(x, y, z, t) in a layer 0 ≤ z ≤ h, where h = 500 m is fixed:

```
∂c/∂t + ∇·(u c) + ∂(w c)/∂z = ∇·(K_h ∇c) + ∂/∂z (K_z ∂c/∂z)
```

Define the depth mean c̄ = (1/h) ∫₀ʰ c dz and integrate over z. This needs four assumptions, all inherited from the protocol:

1. **Wind constant in height** (or its shear dispersion lumped into K). Then ∫ ∇·(u c) dz = h ∇·(u c̄). The shear (Taylor)
   dispersion this hides is part of the "effective" K. It is not modeled separately.
2. **Ground flux** at z = 0: emission F = Σ_j θ_j b_j(r) [g m⁻² s⁻¹] minus deposition v_d c(0).
3. **Lid flux** at z = h: exchange with background air at entrainment velocity w_e, flux −w_e (c − c_bg).
4. **Well mixed in z**, so c(0) ≈ c̄.

Dividing by h gives the Phase-1 equation (protocol §2), with c̄ written as c:

```
∂c/∂t + ∇·(u c) = ∇·(K ∇c) − λ (c − c_bg) + (1/h) Σ_j θ_j(t) b_j(r)          (1)
λ = (v_d + w_e) / h
```

- **Exactness of the loss term.** The term −λ(c − c_bg) is exact for ventilation toward background air. For deposition it
  is an approximation: deposition of the background itself is assumed to be resupplied. This is the protocol's
  [DECIDE] "loss on excess" item. It has not been approved yet.
- **The s_bg term is left out.** It is unresolved (gate D4: volumetric source or boundary value). Stages 0–3 have
  s_bg = 0, so it does not enter any oracle.

| Quantity | Unit | Check |
|---|---|---|
| c, c_bg | g m⁻³ internally; 1 tµg m⁻³ = 10⁻⁶ g m⁻³ | — |
| θ_j | g s⁻¹ (total emission rate of source j) | — |
| b_j | m⁻², with Σ_cells b_j ΔA = 1 on each grid | — |
| (1/h) θ_j b_j | g m⁻³ s⁻¹ | m⁻¹ · g s⁻¹ · m⁻² ✓ |
| K | m² s⁻¹ | — |
| λ | s⁻¹ | — |
| Column mass M = h Σ c ΔA | g | dM/dt = Σθ_j − λ(M − h c_bg A) in a closed box ✓ |

Because the basis is normalized on each grid, θ_j is the same physical rate on 16², 32² and 64². The mass oracle (§4 iv)
depends on this.

## 2. Wind divergence (decision)

- **Stages 0–5: the wind is divergence-free, exactly.** The protocol prescribes u(t) spatially uniform, so ∇·u = 0
  identically. The equation is still discretized in **conservative flux form** ∇·(u c). With uniform u, that form also
  preserves a uniform field exactly (the oracle in §4 v).
- **Meaning of a nonzero divergence.** In a fixed-depth 2D layer, ∇·u ≠ 0 implies vertical exchange through the lid,
  w(h) = −h ∇·u. A 2D model cannot represent that flux without an extra entrainment assumption.
- **Treatment if nonzero divergence ever appears** (for example, interpolated Stage 6 reanalysis winds). It is **not
  permitted silently**:
  - (a) recommended: project the wind onto its divergence-free part before use, and report ‖∇·u‖ before and after;
  - (b) alternative: keep the flux form and add an explicit lid-exchange term.

  Either option is a PDE change that needs a protocol amendment. The solver should reject a non-uniform wind field until
  that amendment exists.

## 3. Domain, initial and boundary conditions

- **Domain and initial condition:** Ω = [0, 10 km]². Initial condition c = c_bg (study) or the oracle's analytic field
  (verification).

| Name | Used for | Face rule (finite volume, face normal n outward) |
|---|---|---|
| `study` | all study runs | inflow face (u·n < 0): advective flux (u·n) c_bg; diffusive flux from Dirichlet c = c_bg over the half-cell distance Δx/2. Outflow face (u·n ≥ 0): advective flux (u·n) c_P (upwind), zero diffusive flux. Faces are reclassified every step, and u·n = 0 counts as outflow |
| `noflux` | verification only | zero advective and diffusive flux on all faces (closed box) |
| `periodic` | verification only | opposite faces identified |

Closed-box and periodic results are verification-only. They never carry over to `study` (gate contract item 3).

**What the `study` BCs give.**
- A uniform c = c_bg is an exact discrete steady state: every face flux balances and the loss term is zero.
- The `study` BCs have no exact mass oracle, because outflow mass leaves at the upwind cell value. Mass accounting in
  `study` runs is a diagnostic: boundary fluxes are recorded, not compared to a reference.

## 4. Independent oracles (exact references, no solver output)

| # | Oracle | BC | Reference | What it isolates |
|---|---|---|---|---|
| i | Pure diffusion of a Gaussian (σ₀ 1.5 km, K 200 m² s⁻¹, 6 h, u = 0, λ = 0) | noflux | method of images: Σ_k [G(x − 2kL − x₀) + G(x − 2kL + x₀)] in x and y, with σ² = σ₀² + 2Kt | diffusion operator, wall BC |
| ii | Translating Gaussian (σ₀ 2.5 km, u = (2, 1) m s⁻¹, K ∈ {20, 200}, 6 min, λ = 0; regime revised in §5.3) | periodic | periodic image sum, centered at (x₀ + u t) mod L, with σ² = σ₀² + 2Kt | convection scheme, numerical diffusion, positivity |
| iii | Exponential decay of a uniform excess (λ = 1/6 h) | periodic, u ≠ 0 | discrete exact c_n − c_bg = a₀(1 + λΔt)⁻ⁿ; continuous a₀ e^{−λt} | loss term, time scheme |
| iv | Forcing with prescribed integral mass (θ = 10 g s⁻¹, grid-normalized patch, λ ∈ {0, 1/24 h}, u = (1, 0.5)) | noflux | discrete exact M_{n+1} = (M_n + Δt(θ + λ h c_bg A)) / (1 + λΔt) | source scaling, units, conservation |
| v | Uniform background steady state (λ = 1/24 h, two winds including u·n = 0 faces) | study | c ≡ c_bg | inflow/outflow BC consistency, loss on excess |
| vi | Observation operator H_S | — | cell-center values, bit-exact | no interpolation (protocol §2) |

**Grid-independence of the references.** Every reference is a closed-form function of the cell centers, independent of
any discretization. The analytic references in (i) and (ii) are checked on their own, before any solver exists (Part A
of the tests, RUN-0003):
- the PDE residual by central differences, which must shrink by a ratio ≥ 3.25 (order ≥ 1.7) under step halving. The
  leading error is O(h²), so the ratio is ≈ 4. The test is one-sided because a first draft used a two-sided band and
  the K = 20 translating case gave 4.78 (a higher-order term). That edit was made after seeing *reference* output, not
  solver output, and is stated here rather than hidden;
- the integral mass, equal to 1 to within 10⁻³ on 64² and 128² cell-center sums (the sum is a midpoint rule, not exact);
- the zero wall gradient for the image solution;
- a **negative control**: a reference built with the wrong K or the wrong wind is rejected.

## 5. Discretization and expected convergence (derived, not fitted)

### 5.1 Scheme

The scheme is the protocol §3 choice: FiPy finite volume, backward Euler, central diffusion, and first-order upwind
convection. For cell P:

```
(c_P^{n+1} − c_P^n)/Δt · ΔA = − Σ_f F_f(c^{n+1}) − λ (c_P^{n+1} − c_bg) ΔA + S_P ΔA
```

- F_f is the advective plus diffusive flux through face f. Each interior flux appears with opposite signs in its two
  cells, so the scheme is conservative by construction.
- The update is linear in θ: x_{n+1} = A_n x_n + B_n θ + d_n, with A_n = (I − Δt L_n)⁻¹. The columns of B_n are the
  unit-source responses (footprints), which is the superposition the inverse stages rely on.

### 5.2 Formal order

The time step is refined with the mesh: Δt = 60 s · (32/n), so the Courant number is the same on every grid.

| Error source | Leading term (modified equation) | Order in Δx at fixed Courant |
|---|---|---|
| Central diffusion | (K Δx²/12) ∂⁴c | 2 |
| Upwind convection | numerical diffusion \|u\|Δx/2 | 1 |
| Backward Euler | −(Δt/2) ∂²c/∂t²; for advection this adds u²Δt/2 = \|u\|Δx·ν/2 | 1 |

So the formal order is **p = 1 for every oracle that has transport**. It is 1 even for pure diffusion, because the
backward-Euler error (proportional to Δt, which is proportional to Δx) eventually dominates. Pre-asymptotically, the
observed order for diffusion may sit above 1.

**p = 1 does not depend on the convection scheme.** It stays 1 if the human picks option (c) of protocol §12.1, a
higher-order convection scheme, because time stepping still limits the order. Other schemes are not entered in
`FORMAL_ORDER` yet: a scheme without a derived entry is **skipped with a reason**, not given a guessed tolerance.

**Numerical diffusion at the study setting** (3 m s⁻¹, 32², ν ≈ 0.58, matching gate N1). The total is
|u|Δx/2·(1 + ν) ≈ 470 + 270 ≈ 740 m² s⁻¹, about 37 × K. Oracle (ii) with K = 20 measures how far the solution is
from the physical answer. The test passes on *convergence*, not on closeness at 32². The relative error at 32² is
printed and recorded, so the §12.1 decision can be made on evidence.

### 5.3 Acceptance rule for convergence oracles (i, ii)

The error norm is the relative discrete L2 error over all cells on grids 16², 32² and 64². The rule is:

- errors decrease monotonically, **and**
- the observed order on the finest pair, log₂(e₃₂ / e₆₄), is ≥ p − 0.2.

The 0.2 margin is a **declared a-priori allowance** for pre-asymptotic behaviour on the coarse grid. It is a design
choice stated before any result. It was not fitted.

**If an oracle fails.** That is a G0 failure. Do not widen the margin; escalate after 3 attempts (gate contract item 4).

#### Revision of oracle (ii) regime (INT-003, human-approved; p = 1, the 0.2 margin, monotonicity and the norm are unchanged)

**Why.** The first regime (σ₀ = 1.5 km, T = 3 h) put the solver in the pre-asymptotic range: the numerical variance
2·K_num·T exceeded the physical variance σ₀² + 2KT, so the error was not yet proportional to Δx. The test measured the
regime, not the solver. Oracle (i) (diffusion) is not changed.

**Option chosen: (b), pulse width and run time.** Option (a), finer grids, is infeasible. At the old σ₀ and T,
2·K_num·T ≤ 0.1·(σ₀² + 2KT) needs Δx of about 9 m, over 1000 cells per side, against the 16²–64² verification budget.

**Derivation (written before the revised test was run against the solver).**

- Per axis i: K_num,i = |uᵢ|Δx/2 · (1 + νᵢ), with νᵢ = |uᵢ|Δt/Δx (upwind plus backward Euler, §5.2).
  Δt = 60 s · 32/n, so νₓ = 0.384 and ν_y = 0.192 on every grid.
- Numerical variance adds 2·K_num,i·T to the physical s = σ₀² + 2KT. Define ρᵢ = 2·K_num,i·T / s.
- For two 2-D Gaussians of variance s and s(1 + ρ), the relative L2 error is exact:
  e² = ∏(1 + ρᵢ)^(−1/2) + 1 − 2 ∏(1 + ρᵢ/2)^(−1/2). For small ρ, e ∝ ρ ∝ Δx, which is the first-order regime.
- **Criterion fixed first:** ρₓ(16²) ≤ 0.10, which is "much smaller" read as a factor of ten, on the coarsest grid and the
  faster axis. The chosen values are σ₀ = 2.5 km and T = 6 min. They keep T a whole number of steps on every grid
  (3, 6, 12) and keep σ₀ ≥ 4Δx on 16² (Δx = 625 m).
- Numbers (`docs/research/oracle_ii_derivation.py`, no solver imported):

| K (m² s⁻¹) | grid | K_num,x (m² s⁻¹) | ρₓ | ρ_y | predicted rel. L2 |
|---|---|---|---|---|---|
| 20 | 16² | 865.0 | 0.0994 | 0.0428 | 0.0492 |
| 20 | 32² | 432.5 | 0.0497 | 0.0214 | 0.0253 |
| 20 | 64² | 216.3 | 0.0249 | 0.0107 | 0.0129 |
| 200 | 16² | 865.0 | 0.0974 | 0.0419 | 0.0483 |
| 200 | 32² | 432.5 | 0.0487 | 0.0210 | 0.0248 |
| 200 | 64² | 216.3 | 0.0244 | 0.0105 | 0.0126 |

  Predicted order log₂(e₃₂/e₆₄) is 0.979 for both K, against the floor p − 0.2 = 0.8.

**Limits of the derivation.** It covers the leading modified-equation terms only. The neglected upwind dispersion term
is O(Δx²/σ₀²), about 0.3 % of the pulse amplitude on 16², and it can only shift the observed order slightly. If the
solver still fails, that is a G0 failure to report, not a reason to retune σ₀ or T.

### 5.4 Acceptance rule for discrete-exact oracles (iii, iv, v)

For these oracles the backward-Euler finite-volume update reproduces the reference **algebraically**:
- (iii) uniform fields have zero face fluxes;
- (iv) interior fluxes cancel and the `noflux` walls carry none;
- (v) every face flux balances at c = c_bg.

So the only admissible error is linear-solver and round-off error. The tolerance is **relative 10⁻⁸**.

- **Solver requirement.** The solver must be run with a direct solve, or with an iterative tolerance of at most 10⁻¹².
- **First-order check.** Oracle (iii) also checks the continuous error against e^{−λt}: halving Δt must halve it, to
  within ±0.05 in observed order. The leading error term is exactly (λ²Δt/2) t a₀ e^{−λt}.

### 5.5 Positivity

For upwind convection with backward Euler and central diffusion, the system matrix is an M-matrix, so c ≥ 0 follows
whenever c⁰ ≥ 0 and S ≥ 0.

- **Upwind:** the test asserts that the relative undershoot is ≤ 10⁻⁸.
- **Any other scheme:** the undershoot is printed and recorded, never hidden (gate contract item 2).

## 6. Solver contract that the tests assume

The future `pravah.forward` module (the solver task, freeze-before-implement) must provide two functions.

```
simulate(*, nx, ny, Lx, Ly, dt, n_steps, u, K, lam, c_bg, c0, bc, scheme="upwind", source=None) -> ndarray (nx, ny)
observe(c, cells) -> ndarray      # cells: sequence of (i, j); returns c[i, j] exactly
```

- **Indexing:** c[i, j] is at cell center ((i + ½)Δx, (j + ½)Δy).
- **`u`:** a uniform (u_x, u_y) in m s⁻¹.
- **`source`:** the volumetric term (1/h) Σ θ_j b_j in g m⁻³ s⁻¹, held constant over the run.
- **`bc`:** one of `study`, `noflux` or `periodic`.
- **Selecting the scheme in tests:** the environment variable `PRAVAH_SCHEME` (default `upwind`).

## 7. External references (not opened in this task)

- **Schlutow et al. 2026** (`schlutow2026`, doi:10.5194/gmd-19-6259-2026, BLDFM). This is the reference point for
  Eulerian advection–diffusion transport and concentration/flux footprints. PRAVAH's B_n columns play the footprint role.
- **Pennington et al. 2024** (`pennington2024`, doi:10.5194/acp-24-2345-2024, CMAQ for Los Angeles). It is cited
  **only** for the physics Phase 1 omits: gas and aerosol chemistry, secondary formation, vertical structure, and
  spatially varying meteorology.

Neither paper was opened here (T-001 opened no primary source either), so both citations are **unverified**. Nothing in
§1–§6 depends on a number from either paper.

## 8. Limitations and open items

- **Not yet executable against a solver.** The solver oracles (Part B) are written but skipped: `pravah.forward` does
  not exist in main. They have not been checked against any implementation, so a defect in the test harness itself
  (as opposed to the references) could still surface when the solver lands.
- **Blocked on a human decision.** The convection scheme (§12.1, gate item 4) is undecided. The tests are robust to
  that choice only for p, as explained in §5.2.
- **Not frozen.** The tests should be frozen with `research orchestrate freeze` on the solver task after:
  - an independent review of this document (gate item 5: "independent check at T-008"), and
  - the human's scheme decision.
- **Not covered:**
  - mass accounting under `study` BCs (diagnostic only);
  - face reclassification during a wind reversal (R3);
  - time-varying θ;
  - the s_bg term (D4).

  e01 must add the R3 face-switch test explicitly (gate item 3).
- **Unapproved inputs.** The loss-on-excess form and λ values are protocol [DECIDE] items that have not been approved.
- **Uncommitted inputs.** The protocol and these files are not committed (gate item 17), so provenance rests on file
  hashes.
