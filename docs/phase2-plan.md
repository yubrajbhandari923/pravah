# PRAVAH phase 2 — solver interchangeability, external/local pollution, Belauri geography, geographic sensor design

## Context
v0 (committed `07f9769`, 196 tests, Pyright 0, 3 notebooks) is a verified synthetic baseline. Verified today: 196 tests pass; network
works (geoBoundaries, Nominatim, Copernicus DEM on AWS S3); GIS wheels resolve for py3.12 (geopandas 1.2, rasterio 1.5, osmnx 2.1,
pyproj 3.8, shapely 2.2). Research Panel: PLAN-001 done except stale T-008 (close it first); F-001..F-008 all *preliminary* (F-006/7/8 awaiting
independent review); baseline EXP-002; D-002; CP-002.
Environment facts: `implementer` is discoverable only if present at session start — try `subagent_type: implementer`, fall back to
`general-purpose` + `model: sonnet` + the implementer rules (done in v0; report fallback). Opus reviews: `general-purpose` + `model: opus`
(or `scientific-reviewer` if it now resolves). This plan = Opus (opusplan); execution = Sonnet; only the two reviews are Opus.
Inventory of direct FiPy calls outside the interface (verified by grep): `experiments.py:481,745,749,755,829` (truth sims, mismatch
roles), `cli.py:41,65`, `scripts/verification.py` (intentional FiPy physics checks), notebooks (01–03 call `simulate` as the demo of the
reference); `design.py`/`inverse.py` already take `forward=` but default to `simulate` and `run_forward` special-cases `forward is simulate`.

## Terms (used literally everywhere, incl. docs/code names)
local = emitted inside study polygon; external = outside it (explicit source or outer-inflow-boundary input); international
transboundary = crossing a *verified national* boundary (decided by ADM0 polygons, never by the municipality edge); indoor = inside a
building (separate one-way-coupled box). Study polygon ≠ simulation domain ≠ outer inflow boundary; the polygon is only a reporting
mask/weight, never a no-flux wall.

## Stage A — solver interchangeability (GATE before any new physics)
1. **Before editing**: coordinator runs `scripts/capture_regression.py --out results/solver-regression/before` (fixed seeds): reference
   simulate (fields .nc), inverse linear + a 3-source direct-vs-linear fit, smoke benchmark, smoke alpha-selection, smoke mismatch.
   Registered as EXP-008 runs via `research run exec`.
2. `forward.py`: add `BackendInfo(name, version, settings, capabilities{linear, boundary_bases, ...})`, class `FipyForward(convection,
   tolerance)` (callable, `.info`, `.forward_id`), module constant `FIPY`, `get_forward(spec)` resolver: `None|"fipy"` → FIPY,
   `"module:callable"` (+ optional JSON kwargs `--solver-kwargs`) → factory called once. `simulate` function kept (backward compatible;
   equals `FIPY`). `forward_info(f)` reads `.info` else derives from identity. `FieldResult.diagnostics["backend"]` = BackendInfo dict.
3. `inverse.py`: cache key ALWAYS includes `BackendInfo` (identity+version+settings); `ResponseCache.get` raises unless
   `forward_info(f).capabilities.linear` — explicit opt-in, never assumed. `run_forward` validates every backend (incl. FiPy under tests).
4. `experiments.py`: `_Setup` carries `truth_forward` and `inference_forward` from `cfg["solver"]["truth"|"inference"]` (+ settings);
   replace the five `simulate(` sites by `st.truth_forward`; cache/build/fit use `st.inference_forward`. Provenance stores both BackendInfo.
   `cli.py`: `--solver SPEC`, `--truth-solver`, `--inference-solver` on simulate/inverse/benchmark; resolved once, passed down. Defaults
   unchanged. Notebooks 01–03 keep `simulate` (documented as the FiPy reference) but their helpers take `forward`.
5. Tests `tests/test_solver_routing.py`: RecordingForward (counts calls per role: simulate, response basis, direct fit, greedy case build,
   benchmark truth, mismatch truth vs inference, CLI via a tiny module:callable) proves every expected call reaches the injected solver and
   FiPy is *never* called (sentinel FiPy patched to raise); a `DistinguishableForward` (returns `fipy + 1.0` marker) detects hidden FiPy
   calls — labelled *interface test, not physics*; real alternate FiPy configuration (`FipyForward(convection="upwind")`) changes results,
   changes cache keys, doesn't reuse entries; non-linear-declared backend refused by response cache.
6. **Regression**: `scripts/compare_runs.py before after` — equality on fields (`xr.testing.assert_identical`), q_hat, layouts.json, raw
   metrics, mismatch metrics, excluding `runtime_s/wall_s/timestamps/paths/git/backend-info additions`. Require bit-identical; any
   difference must be explained (tolerance never loosened silently). Output → `results/solver-regression/{before,after,comparison.json}`.
7. `docs/solver-interchangeability.md`. Then full pytest + Pyright once; commit.

## Shared contracts (coordinator writes `config.py` extensions + stubs FIRST, then workers fill)
- `Domain` unchanged (local metres, origin 0); geography keeps a separate `GeoRef(epsg, x0, y0)` in the prepared manifest.
- `GaussianSource.group: str = "local"` (tags); new `RasterSource(name, group, field: ndarray (compare=False), sha256)` — serialised by hash
  (config JSON points to a prepared `.npy` + checks hash); `source_fields` normalises both so Σ f·dA = 1.
- `StudyRegion(x0,y0,x1,y1)` (exact cell-overlap fractions) | `StudyMask(sha256, fraction array)`; `Scenario.study` optional; `study_weights(domain)`.
- `InflowBasis(name, sides: tuple[str], profile: ((t,v),...))` in `Boundary.bases`; amplitude unit µg/m³; acts per face only where
  u·n<0 under the current wind; static `background` remains. Parameter vector θ = [emission amplitudes (µg/s)…, basis amplitudes (µg/m³)…];
  `scenario.param_table()` → names, kinds, units. `q` argument of `simulate/ForwardSolver` is θ (unchanged when no bases ⇒ bit-identical).
- `WindField` (optional extension): `StreamfunctionWind` per segment — face fluxes from ψ at cell corners are exactly divergence-free
  (uniform wind is the special case ψ = u·y − v·x, tested equal to the existing path). Validates units, divergence, interpolation.
  Synthetic "channelled flow" demo only; labelled not terrain physics.

## Stage B — external vs local (synthetic)
- `forward.py` (worker B1): per-wind-segment unit boundary source arrays `s_in_k` (ny,nx) per basis side; step plan adds basis-profile
  breakpoints; total source = s_q + s_in_bg + Σ a_k φ_k(t) s_in_k; budget inflow uses same coefficients; wind reversal flips inflow faces.
  Tests: basis acts only on inflow faces and flips at reversal; mass budget closes; uniform-basis equals scalar background; stream-function
  uniform equals legacy; constant solution invariant under divergence-free non-uniform ψ with matching background.
- `inverse.py` (B2): `Responses` generalised to parameters with kinds; `base` split into `base_initial`, `base_background` (sum = base);
  boundary-basis unit responses use zero emissions/IC/background (baseline + increments, no double counting); `InferenceOptions.q_scale`
  defaults per kind (emission 1e5 µg/s, basis 0.05 µg/m³ — documented scaling only); `FitResult` gains param_names/units; q_std etc. per kind.
- `attribution.py` (B2): `contributions(resp, θ)` → fields by tag {local, external_sources, boundary, background, initial}; sum == total (also
  checked against a direct total simulate incl. nonzero background and a wind-reversal re-entry case: local stays local); `external_fraction(
  contrib, weights, t0, eps)` = concentration-derived Σw c_ext / Σw c_total over study cells (area fraction) × trapezoid time weights over
  [t_exclude,T], NaN→null when denominator < eps·signal scale (named quantity; inward mass-flux fraction is NOT substituted);
  component RMSE for local/external; source-amplitude error only for emission kind; never compares µg/s with µg/m³.
- Tags never reach `fit_sources` (type-level: Observations only); used for synthetic evaluation.
- Synthetic scenario `configs/transboundary_*.json` (documented defaults): study 10×10 km at the centre; simulation domain A 20×20 km and
  B 30×30 km (same absolute source/study coordinates), dx=250 m (A: 80², B: 120²), K=100, H=500, λ=1/21600, T=6 h, out 600 s, dt 60 s.
  Sources: L1 1.0e5, L2 0.7e5 µg/s (σ≈500 m) inside; E1 2.0e5 µg/s (σ=1 km) ≈4 km W of study edge (arrives after ≈5–7 ks at u=1 m/s, onset
  profile); boundary uniform basis 0.03 µg/m³ with 2 h onset; four side-sector bases. Named scenarios: `local_only`, `external_sources`,
  `boundary_only`, `mixed` (no plume counted twice; a documented test that source E1 and boundary basis are never combined to represent the same plume).
  Cases: confounded (E1 collinear upwind of L1 along wind), no-arrival (wind away / T < travel time ⇒ sv≈0, q_std=inf, estimate = prior,
  *not* recovered), wind mismatch, prior sensitivity (alpha grid chosen on validation seeds only; weak/no prior reported), noise level set
  relative to the documented signal scale (reported in µg/m³ and % of peak), interior-only vs layouts with an upwind/background site, a case where
  the total field is predicted well but local/external decomposition fails (reported with condition numbers).
  Domain-size study: A vs B with fixed study region and sources (differences in study-region fields + boundary-basis equivalence to a distant explicit source).
- Fresh seeds: design 101, validation 202, test 303 (+noise ranges disjoint) — **audit** of v0 alpha history is written to
  `docs/transboundary.md` (v0 final already uses validation-frozen alpha; exploratory runs stay labelled).
- `indoor.py` + notebook section: `IndoorBox(P,a,k_dep,V)`, exact piecewise-linear integration of dc_in/dt = P a c_out − (a+k_dep)c_in + q_in/V;
  outdoor concentration interpolated at the building (observations.sample); estimate ≤1 indoor amplitude (linear, nonneg) with building params
  known; tests: constant-forcing analytic exponential, steady state P a c/(a+k_dep) + q/(V(a+k_dep)), a→0/k_dep→0 limits, paired readings; ambiguity
  demo (indoor-only sensor: q_in vs outdoor amplitude collinear for constant forcing; singular values). One-way coupling, no double-counted exhaust.
- Notebook 04 `notebooks/04_transboundary_pollution.ipynb` per brief; bounded < ~5 min.

## Stage C — Belauri geography (optional extra `[geo]`, offline after prepare)
- `pyproject`: extra `geo = [geopandas, shapely, pyproj, rasterio, osmnx, requests]`; `requirements-lock-geo.txt`. Package modules
  `pravah/geo/{fetch.py,prepare.py,regionio.py}` import GIS lazily; core imports never need them.
- CLI: `python -m pravah geo fetch --config configs/belauri_geo.json` (bounded, cached under `data/belauri/raw/`, polite User-Agent, rate
  limits, no secrets) and `geo prepare` (offline). Manifest `data/belauri/manifest.json` (URLs, IDs/release, retrieved_utc, license,
  sha256, CRS, bbox, units, resampling, source→grid mapping, assumptions). Raw rasters git-ignored; processed small arrays (`.npz`) +
  manifest + checks tracked if small (<~10 MB) else regenerated.
- Boundary: discover geoBoundaries levels from API metadata (do not assume ADM3), find "Belauri" under Kanchanpur / Sudurpashchim; verify name,
  parent district, level meaning, validity, area, centroid, extent; independent locator = OSM relation/Nominatim (feature ID recorded) —
  agreement within tolerance reported; if only approximate, mark `boundary_status: approximate` and the requirement incomplete.
  International boundary: ADM0 NPL/IND from the same source; distances from the municipality; wind-sector crossing classification.
- Projection: UTM zone from the centroid (expected EPSG:32644, verified from `pyproj`), round-trip tests, axis order recorded.
- DEM: Copernicus GLO-30 COG tiles (public S3) via windowed rasterio read; nodata mask; native resolution recorded vs simulation cell;
  hillshade, slope, N–S/E–W profiles; report where relief is (distances, ranges). Interpretation: surface model (DSM).
- Features: OSMnx roads, building/landuse (residential), schools/facilities within bbox; counts; coverage warnings (OSM gaps ≠ zero sources).
- Grid: rectangular metric mesh over municipality + transport buffer sized from travel time (u·T) and diffusion length (√(2KT)), two
  resolutions (≈500 m, ≈250 m) + study-polygon fractional weights (shapely intersection-area, tested sum = polygon area).
- Source hypotheses: road-associated, built-up, external patch (inside simulation domain, outside polygon, on Indian or Nepali side as
  geometry dictates), each normalised Σ f dA = 1, stored as `RasterSource`; labelled *synthetic spatial priors*.
- Blocked-source policy: documented alternates; if unavailable mark milestone incomplete, never fabricate.

## Stage E — Belauri forward (notebook 05)
Prescribed synthetic uniform/two-segment winds (and optional streamfunction demo wind), local-only/external-only/combined, arrival traces, regional
summaries with area weights, smaller/larger outer domain, 500 m vs 250 m mesh + dt halving, K_eff diagnosis of numerical diffusion at the
Belauri mesh, sensitivity plots; every map labelled "Real geography; synthetic emissions and meteorology."; DEM shown as context unless a layer
actually alters source geometry/constraints (stated per layer).

## Stage F — geographic sensor design (notebook 06)
- Candidates (stable IDs, lon/lat + x/y, category, flags): regular ≈1.5–2 km grid clipped to the polygon (~50–100 sites); separate reference
  pool (outside polygon, Nepal side practical; across-border hypothetical, excluded by default); road/facility variants.
- Two settings: all-interior; same budget with ≤2 external reference sites (counted in budget).
- Methods: random (20 seeds), spread, greedy-total (existing objective), greedy-attribution (normalised: ½(MSE_loc/⟨c_loc²⟩ + MSE_ext/⟨c_ext²⟩), the
  scales computed from DESIGN truths only, quadratic-form speedup per component); tiny exhaustive fixture; layouts called *greedy*, not optimal.
- `design.py`: objective registry on `DesignCase` (named quadratic forms) keeping the existing total objective bit-identical.
- Ensembles design/validation/test disjoint (seeds above); common noise within design pairs, fresh test noise; alpha/prior chosen on
  validation only (reuse `run_alpha_selection`); budgets 1,2,3,5,8; counts declared after timing a pilot.
- Spatial-block held-out evaluation (min train–test separation; infeasible budgets reported), random held-out sites, robustness (unseen
  winds, boundary time profile, omitted source, refined-grid truth), selection stability across mesh refinement. Exports: CSV + GeoJSON layouts with provenance.

## Task bundles and ownership (≤2 concurrent workers)
| ID | Owner | Files | Dep | Gate |
|---|---|---|---|---|
| P0 | coord | close T-008; plan/tasks/experiments in Research Panel; docs/phase2-plan.md | — | records exist |
| A1 | coord | capture_regression, EXP-008 before-run | P0 | outputs saved |
| A2 | worker | forward/inverse/design/experiments/cli routing, tests, compare_runs, docs/solver-interchangeability.md | A1 | bit-identical + routing tests + pytest + pyright |
| contracts | coord | config.py/source stubs/param_table | A | tests |
| B1 ∥ C1 | worker1: forward bases/wind field/tags-physics; worker2: geo fetch/prepare/regionio (+fixtures) | forward.py,sources.py | contracts | unit tests |
| B2 ∥ C2 | worker1: inverse/attribution/indoor/nb04/configs/docs; worker2: Belauri prepare run, geometry checks, docs/geography.md | | B1 / C1 | EXP-009 runs |
| D | Opus review 1 (B) → Sonnet fixes | | B2 | |
| E | worker: Belauri forward + nb05 | | B,C | EXP-011 |
| F | worker: candidates, objectives, design runs, nb06 | | E | EXP-012 |
| G | Opus review 2 (geo design) → fixes; clean notebooks; docs/phase2-validation.md; Research synthesis, independent finding reviews (reviewer runs `research --agent scientific-reviewer finding review …` itself), checkpoint | | F | |

## Verification
`pytest -q`, `pyright`, `scripts/verify_notebooks.py` (all 6, offline), `scripts/compare_runs.py`, `python -m pravah geo prepare` (offline from
cache), runs wrapped by `research run exec` (EXP-008 routing, EXP-009 external/local identifiability, EXP-010 geography verification, EXP-011
Belauri forward sensitivity, EXP-012 geographic design); synthesise before the 5-run budget; findings stay preliminary until reviewed.
Final deliverable: requirement/status table (docs/phase2-validation.md) with file/result references; unresolved items named (terrain dynamics,
authoritative boundary if approximate, empirical validation).
