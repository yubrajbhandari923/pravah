# PRAVAH v0 — implementation plan

## Context
`/Users/yubraj/Code/pravah` is empty apart from `AGENTS.md`, `.claude/` (implementer agent, model sonnet) and an
initialised `.research/` (no plans/tasks/experiments yet; policy: 5 runs before synthesis, findings stay
preliminary until independent review). Not a git repo. Python 3.11/3.12 and `uv` 0.8 available (system default
3.14 is too new for FiPy). The team needs a reproducible synthetic baseline: FiPy advection–diffusion–removal
forward model, source-amplitude inversion, sensor-placement comparison, three notebooks, CLI, docs, Research
Panel records. All results are idealised synthetic experiments (not empirical validation for Belauri).

Environment facts that shape execution:
- `implementer` is **not** in this session's agent-type list (agents load at session start). Delegation will try
  `subagent_type: implementer`; if refused, use `general-purpose` with `model: "sonnet"` and the implementer.md
  rules pasted into the brief. Scientific reviews use `general-purpose` with `model: "opus"` and a read/verify-only
  brief; a project `.claude/agents/scientific-reviewer.md` (model: opus, tools Read/Grep/Glob/Bash) is written for
  future sessions. Deviations reported at the end.
- opusplan: this plan is Opus; execution runs on Sonnet.
- `git init` locally (no remote, no push) so runs can record revision/dirty state and diffs stay reviewable;
  local milestone commits.

## Fixed design decisions

**Environment**: `uv venv --python 3.12 .venv`; `uv pip install -e ".[dev]"`; freeze to `requirements-lock.txt`.
Deps: fipy, numpy, scipy, xarray, pandas, matplotlib, jupyter/ipykernel, (opt) ipywidgets; dev: pytest, pyright,
nbformat, nbclient. If FiPy rejects numpy 2, pin `numpy<2` and record why.

**Grid / flattening**: uniform `Grid2D(dx,dy,nx,ny)` (periodic mode: `PeriodicGrid2D`). FiPy cell id
`k = i + j*nx` ⇒ `values.reshape(ny, nx)` gives `[y, x]`; all public arrays are `(…, y, x)`, coordinates are cell
centres in metres, ascending.

**PDE** `∂c/∂t + ∇·(u c) = ∇·(K∇c) + Σ q_j s_j(t) f_j(x,y)/H − λc` (units as spec). Sources: Gaussian evaluated at
cell centres, normalised **discretely** so `Σ f_j A = 1` exactly (resolution-invariant emission); quadrature effect
reported as discrete-vs-continuous (erf, domain-truncated) mass ratio. `s_j(t)` piecewise-constant dimensionless
profile (default 1).

**FiPy discretisation** (`forward.py`):
`TransientTerm() + ExponentialConvectionTerm(u_int) == DiffusionTerm(K) − ImplicitSourceTerm(λ + a_out + a_in) + S_q + S_in`
- `u_int`: rank-1 FaceVariable = u on interior faces, **0 on exterior faces**, so FiPy's convection contributes no
  boundary flux and FiPy's unconstrained diffusion gives zero boundary flux. All boundary physics is then explicit
  and identical to what the diagnostics use:
  - outflow faces (u·n>0): `a_out = Σ (u·n)A_f / V` (upwind export of the boundary cell value; no diffusive flux).
  - inflow faces (u·n<0): face value prescribed `c_b`: advective `|u·n| A_f c_b` plus Dirichlet diffusive
    `2K A_f (c_b − c_P)/d` → `a_in = Σ 2K A_f/(dV)`, `S_in = Σ (|u·n| + 2K/d) A_f c_b / V`.
  - tangential faces (u·n=0): zero diffusive flux. Classification is per face, recomputed per wind segment.
- Monotone convection: exponential (Scharfetter–Gummel) weighting gives an M-matrix for all Péclet (cell Pe =
  u·dx/K = 2 here, where pure upwind would add ≈ u·dx/2 = 100 m²/s numerical diffusion, equal to K). `upwind`
  selectable via `SolverOptions.convection`. Verified by nonnegativity + refinement tests, not assumed.
- Time: implicit (backward) Euler with `LinearLUSolver` (tight, so superposition holds to ~1e-10). Step plan =
  union of output times, wind-segment starts, emission-profile breakpoints; each interval split into
  `ceil(len/dt)` equal steps. Default dt=60 s: Courant 0.3, diffusion number 0.15. First-order in time — a
  dt-refinement study (60/30/15 s) on the reference case is reported; dt only changed if that study demands.
- Boundary modes: `open` (above), `periodic` (`PeriodicGrid2D`, uniform u), `closed` (no flux; raises `ValueError`
  for any nonzero wind segment — a uniform wind cannot be tangent to all walls).
- Budget per step (all implicit at n+1, consistent with the scheme): emission `dt·Σq s`, removal `dt·λHΣcA`,
  inflow `dt·H·Σ_in(|u·n|c_b + 2K(c_b−c_P)/d)A_f` (signed; can be negative), outflow `dt·H·Σ_out u·n c_P A_f`;
  residual `ΔM − (E − R + In − Out)`, accumulated per output interval. Negative-value count / min reported, never
  clipped.
- ICs: zero / uniform / Gaussian bump; inflow background `c_b` scalar. Base solution = zero emissions + IC + c_b
  (simulated, not assumed uniform).

**Public contracts** (frozen dataclasses, `src/pravah/`):
- `config.py`: `Domain`, `GaussianSource(name,x,y,sigma,profile)`, `WindSegment(t_start,u,v)`, `Transport(wind,K,lam,H)`,
  `Boundary(mode, background)`, `Initial(kind,…)`, `Times(duration,dt,output_interval)`, `Scenario` (all of the
  above — **no emission rates**), `SolverOptions(convection, lu)`, `load_config(path)` (JSON, optional one-level
  `"extends"` merge), `to_dict`/validation (ranges, finiteness, ordering).
- `simulate(scenario, q, solver=SolverOptions()) -> FieldResult` — `FieldResult(concentration: xr.DataArray (time,y,x),
  budget: xr.Dataset|None, status, runtime_s, diagnostics: dict)`; no FiPy objects escape.
- `observations.py`: `SensorLayout(ids,x,y)`, `ObservationSpec(times, noise_std, seed)`,
  `sample(field, layout, times) -> DataArray(time,sensor)` (noiseless; bilinear on cell centres +
  linear in time, `ValueError` outside centre hull/time range), `observe(...) -> Observations(values, noise_std,
  layout, times)` (noisy only, local `np.random.default_rng`, negatives kept).
- `inverse.py`: `fit_sources(scenario, obs, InferenceOptions(method="linear"|"direct", q_scale, alpha, prior,
  heldout_ids)) -> FitResult(q_hat, success, message, weighted_residual, singular_values, rank, condition,
  q_std|None, method, options)`; `reconstruct(scenario, fit) -> FieldResult`; `ResponseCache` (in-memory + optional
  `.npz` dir), key = sha256 of canonical JSON of scenario + solver + times + `SCHEMA_VERSION`. Responses: base
  (q=0) and per-unit `g_j` (zero IC, c_b=0, q_ref=q_scale_j then ÷q_ref). Whitening by declared σ; σ=0 ⇒ unit
  weights (documented). Tikhonov rows `√α (θ−θ_prior)` in scaled units. `lsq_linear(bounds=(0,∞))` vs
  `least_squares(bounds, x_scale)` on direct solves. Linear route refuses `scenario.linear_in_sources=False`.
- `metrics.py`: weighted space–time MSE/RMSE (trapezoid in time over [t_excl, T], cell area, w≥0; zero denominator
  → error), source relative error, held-out RMSE, synthetic population weight map.
- `design.py`: candidate grid (10×10 at 500…9500 m), allowed mask + min separation + infeasibility errors;
  `random_layouts`, `spread_layout` (farthest-point from centre-most site, lowest-index ties), `greedy_layout`
  (min mean design MSE; ties → lowest index); `DesignCase` (truth scenario, inference scenario, q_true, noise CRN
  array per site); `score_layout`, `select_layout`, `evaluate_layouts` (long-form DataFrame).
- `experiments.py`: scenario ensembles (q × U(0.5,1.5), wind speed 0.6–1.6 m/s, direction nominal ±40°, some
  two-segment winds; disjoint design/test seeds), benchmark + mismatch runners writing run directories.
- `io.py`: run-dir creation (refuse overwrite unless `--overwrite`), provenance JSON (config, seeds, versions,
  platform, git rev/dirty, timings), NetCDF fields, CSV/JSON metrics (NaN→null). `plotting.py`: small figure helpers.
- `cli.py` + `__main__.py`: `simulate | inverse | benchmark | verify` with `--config --out [--overwrite]`.
  `benchmark` dispatches design comparison or mismatch by config `"experiment"`.

**Configs**: `reference.json` (spec defaults), `smoke.json` (25×25 grid, 5×5 candidates, 2+2 cases, budgets 1,3,
1 noise rep, 5 random), `benchmark.json` (100 candidates, budgets 1,2,3,5,8, 6+6 cases, 3 reps, 20 random),
`mismatch.json` (reads design-stage layouts from a benchmark run dir; refined 100×100 truth, hidden wind
rotation/speed error, extra 4th source, one deliberately poor upwind-corner layout).

## Task bundles (dependency order, ownership, gates)
| ID | Owner | Files | Depends | Gate |
|---|---|---|---|---|
| A | coordinator | pyproject, .gitignore, configs/reference.json, config.py, docs/specification.md, architecture.md, implementation-plan.md, Research plan/tasks/experiments | — | install works; `pytest tests/test_config.py` |
| B | worker 1 | sources.py, domain.py, forward.py, tests/test_forward.py, scripts/verification.py | A | invariants, Fourier convergence (3 res), removal-only vs (1+λdt)^-n and dt→0, closed conservation, open export + budget residual, nonnegativity, source integrals |
| C | worker 2 (∥B) | observations.py, io.py, plotting.py, tests/test_observations.py | A | affine-exact interpolation, rejection of invalid inputs, noise reproducibility |
| C2 | worker | notebooks/01_transport.ipynb | B,C | executes via nbclient |
| D | worker | inverse.py, metrics.py, tests/test_inverse.py, notebooks/02_inverse.ipynb | B,C | superposition ≤1e-6 rel (nonzero IC+bg), noise-free recovery ≤1e-4 rel, degenerate flagged, direct≈linear, cache invalidation |
| E | Opus review | read-only | D | defects fixed by Sonnet worker |
| F | worker | design.py, tests/test_design.py | D | uniqueness/constraints/cardinality/determinism, greedy step == exhaustive best on tiny fixture, no test-truth access |
| G | worker | experiments.py, configs smoke/benchmark/mismatch, notebooks/03_sensor_design.ipynb | F | smoke < few min |
| H | coordinator (+1 worker for docs) | cli.py, __main__.py, scripts/verify_notebooks.py, scripts/register_research.py, README, docs/extending.md, validation.md, scientific-reviewer agent | G | pyright 0 errors; notebooks pass; CLI commands run |
| I | Opus review → Sonnet fixes | — | H | frozen `results/reference-v0`, synthesis, findings (preliminary), checkpoint |

At most two concurrent workers (only B∥C). Coordinator reviews every diff, reruns tests itself, and is the sole
writer to `.research` (plan, tasks with `cmd:` checks, experiments EXP numerical-verification / inverse-recovery /
deployment-comparison / mismatch registered before their runs, runs via `research run exec`, metrics/artifacts
registered, synthesis every ≤5 runs, checkpoint at end). Progress notes via `research task note` survive compaction.

## Numerical risks & mitigations
- FiPy per-step overhead → measure one reference solve first; if >~10 s, reduce sweep sizes transparently (never
  silently) or reuse responses more.
- ExponentialConvectionTerm Péclet weighting must actually see K (FiPy passes the equation's diffusion term) —
  test exponential ≠ upwind and both monotone.
- Exterior-face convection semantics: tested by budget residual ~1e-10 in open mode with c_b>0 and by steady
  1-D-like through-flow (outflow ≈ inflow + emission − removal).
- Implicit Euler temporal error → explicit dt-refinement evidence, not assumed.
- Ill-conditioning with 1–2 sensors → report σ_i, rank, cond; never hide with prior.

## Verification (end-to-end)
```
uv venv --python 3.12 .venv && uv pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/pyright
.venv/bin/python -m pravah verify --out results/verification
.venv/bin/python -m pravah simulate --config configs/reference.json --out results/reference-v0
.venv/bin/python -m pravah inverse  --config configs/reference.json --out results/inverse-v0
.venv/bin/python -m pravah benchmark --config configs/smoke.json --out results/smoke
.venv/bin/python -m pravah benchmark --config configs/benchmark.json --out results/benchmark
.venv/bin/python -m pravah benchmark --config configs/mismatch.json --out results/mismatch
.venv/bin/python scripts/verify_notebooks.py
```
Scientific runs go through `.research/bin/research run exec EXP-… -- <cmd>`. Measured numbers go to
docs/validation.md.

## Status / deviations (added after v0)

The plan above is kept as written; where the implementation differs, the code and `docs/specification.md` are
authoritative. Results and tolerances: `docs/validation.md`. Differences found by comparing this plan with `src/`:

- No `python -m pravah verify` subcommand: the refinement studies are `scripts/verification.py --out results/verification`.
- No `domain.py`; the grid lives in `config.Domain` and `sources.cell_centres`.
- `SolverOptions` has `convection` and `tolerance` (not `lu`). Linearity is the function `inverse.is_linear(scenario)`,
  not a `Scenario.linear_in_sources` field. Unit responses use a fixed `q_ref = 1e5` ug/s (a numerical scaling only), not `q_scale_j`.
- Greedy selection is `design.greedy_trace` / `select_layout` (nested layouts), not `greedy_layout`. `FitResult` also carries
  `effective_rank` and `q_std` (always an array). `fit_layout` and the benchmark report both.
- Added after the plan: `forward.ForwardSolver` / `check_field_result` and the optional `forward=` keyword (inverse, design);
  `configs/benchmark_unregularized.json` (alpha = 0 sensitivity); `greedy_weighted` and the uniform/population metrics.
- The headline benchmark uses alpha = 1, adopted after the alpha = 0 run showed unbounded errors for plume-blind layouts.
- The numerical-diffusion estimate of the baseline scheme was first stated as +31% of K (space term only); the measured value at the
  default grid and dt = 60 s is +61% (space +0.31, backward-Euler time term +0.30).
- Still not done: a solver other than FiPy (the interface exists, nothing implements it), 2-D oblique-wind numerical-diffusion
  measurement, CLI/`run_benchmark` support for `forward=`.
