# PRAVAH

An **idealised, synthetic** testbed for regional air-quality work: a 2-D advection-diffusion-removal model (FiPy),
a source-amplitude inverse problem, and a sensor-placement benchmark. The region we eventually care about is
Belauri (Nepal), but **nothing in this repository is a Belauri estimate**: domain size, wind, diffusivity, removal
rate, source positions and amplitudes are demonstration values chosen to make plumes visible on a 10 km x 10 km box.
There are no observations, no chemistry and no forecasting. Everything is verified against
mathematics (exact solutions, conservation, refinement), not against measurements. The text above and the v0 sections
below describe the synthetic baseline; **Phase 2** (section below) adds real Belauri *geography* (polygon, roads, buildings,
DEM context) under **synthetic emissions and meteorology**, still with no real data.

> Documentation map: `docs/README.md` (one row per question). Agents: start with `AGENTS.md`.

## Install

Tested with Python 3.12.7 on macOS arm64 (the package declares `>=3.11`; the system Python 3.14 was judged too new for FiPy and not tried).

```bash
uv venv --python 3.12 .venv
uv pip install -e ".[dev]"                      # newest compatible versions
# or the exact tested versions (the lock file contains one machine-specific "-e file://..." line; drop it):
grep -v '^-e ' requirements-lock.txt > /tmp/constraints.txt
uv pip install -e ".[dev]" -c /tmp/constraints.txt
```

Tested versions (`requirements-lock.txt`): fipy 4.0.3, numpy 2.5.3, scipy 1.18.1, xarray 2026.9.0, pandas 3.0.6,
matplotlib 3.11.2. Run everything with `.venv/bin/python` (or activate the venv).

## Quick start

All commands take `--config` and `--out`; an existing non-empty `--out` is refused unless you add `--overwrite`.

```bash
python -m pravah simulate  --config configs/reference.json --out results/my-sim            # ~3 s solve
python -m pravah inverse   --config configs/reference.json --out results/my-inv            # linear route, ~20 s
python -m pravah inverse   --config configs/reference.json --out results/my-inv2 --direct  # + direct route, ~2 min
python -m pravah benchmark --config configs/smoke.json     --out results/my-smoke          # ~20 s, coarse, plumbing check
python -m pravah benchmark --config configs/smoke_alpha_selection.json --out results/my-smoke-alpha  # plumbing check
# final procedure, in this order (alpha is chosen on validation scenarios BEFORE the test scenarios are run):
python -m pravah benchmark --config configs/alpha_selection.json --out results/alpha-selection   # validation ensemble
python -m pravah benchmark --config configs/benchmark.json --out results/benchmark-final         # reads the frozen alpha
python -m pravah benchmark --config configs/mismatch.json  --out results/mismatch-final          # same alpha, same layouts
python -m pravah benchmark --config configs/benchmark_unregularized.json --out results/benchmark-unregularized-final  # alpha = 0 SENSITIVITY run, not a selected alpha
```

Order: `alpha-selection` -> `benchmark-final` -> `mismatch-final` (the last two read the frozen alpha file; `mismatch-final`
also reads the layouts of `benchmark-final`). `benchmark_unregularized` is a sensitivity run, not part of the chain.
The notebooks and `configs/smoke.json` use alpha = 1 as a plumbing/demo value; it is **not** the selected alpha.

Alpha (Tikhonov strength) procedure: `configs/alpha_selection.json` fits alpha-independent layouts (random + spread)
on a separate **validation** ensemble (seeds 44 / 9000) for a grid of alphas and writes the lowest-criterion alpha to
`results/alpha-selection/alpha_selected.json`. `configs/benchmark.json` and `configs/mismatch.json` have `design.alpha:
null` and `design.alpha_selection_file`, so they read that frozen alpha (if `alpha` is also set it must agree, else
the run stops) and record the file's sha256. The test ensemble (seeds 33 / 7000) is fresh and independent of both
the design and validation ensembles. Runs with a numeric alpha and no selection file are labelled
`not selected: sensitivity` in `provenance.json` / `summary.json`.

`configs/mismatch.json` re-uses the layouts of a finished benchmark: it reads them from `mismatch.layouts_from`
(`results/benchmark-final`); if you wrote your benchmark elsewhere, extend the config and change that path.
The `benchmark` command runs any of the three experiments depending on the config's `"experiment"` key
(`benchmark`, `alpha_selection`, `mismatch`).

**`results/reference-v0` (and `inverse-v0`, `smoke`, `alpha-selection`, `benchmark-final`,
`benchmark-unregularized-final`, `mismatch-final`, `verification`, plus the exploratory `benchmark`,
`benchmark-unregularized`, `mismatch`) are the frozen results quoted in `docs/validation.md`. Do not overwrite them:
write alternatives to `results/<your-name>`.**

Notebooks (`.venv/bin/jupyter lab notebooks/`), in this order. Run times are one measurement each on the machine
described in `docs/validation.md`, not a controlled benchmark.

| Notebook | What it does | Run time |
|---|---|---|
| `01_transport.ipynb` | forward model, snapshots, mass budget, "change one thing" demos | 25 s |
| `02_inverse.ipynb` | recover source amplitudes from noisy sensors; identifiability | 18 s |
| `03_sensor_design.ipynb` | random vs spread vs greedy layouts, mismatch (smoke-sized) | 35 s |

## What is modelled

`dc/dt + div(u c) = div(K grad c) + sum_j q_j s_j(t) f_j(x,y)/H - lambda c`

* Units: x, y in m; t in s; c in ug/m^3; u in m/s; K in m^2/s; lambda in 1/s; mixing depth H = 500 m (fixed);
  emission rates q_j in ug/s.
* Sources: Gaussian shapes f_j (sigma = 500 m), sampled at cell centres and **normalised on the grid** so that
  `sum(f_j * cell_area) = 1`: the integrated emission is q_j regardless of resolution. `s_j(t)` is a
  piecewise-constant factor (default 1).
* Boundaries (per face): `open` (inflow faces prescribe a background concentration, default 0; outflow faces
  export by advection), `periodic` (uniform wind), `closed` (zero wind only).
* Wind may be piecewise constant in time (`configs/wind_change.json`). Time stepping is backward Euler.
* Defaults: 50 x 50 cells, dt = 60 s, 4 h, wind (1.0, 0.3) m/s, K = 100, lambda = 1/21600 1/s, q = (1e5, 1.5e5, 8e4).
* **Inverse problem:** only the three amplitudes q_j >= 0 are fitted. Wind, K, lambda, H, source geometry, initial
  state and boundary forcing are *known*. Sensors read bilinear-interpolated concentration plus Gaussian noise of
  known sd (0.005 ug/m^3, about 3% of the reference peak of 0.161) every 600 s.

Full definitions: `docs/specification.md`; discretisation and design: `docs/implementation-plan.md`.

## What the experiments do and do not establish

* **Matched benchmark** (`results/benchmark-final`, alpha = 0.1 selected on validation scenarios): truth and inference use the *same* solver and grid (an "inverse
  crime"), with the true wind given to the inversion. It shows what sensor layout can do when the model is right.
  It does not show performance on real data.
* **Mismatch study** (`results/mismatch-final`): fixed layouts tested when the truth differs in one way at a time (finer
  grid and smaller dt; hidden +20% wind speed and 15 degree rotation; an unmodelled 4th source). One magnitude per kind.
* Counts are modest (6 design + 6 test scenarios, 3 noise draws). The reported **sd is the spread of individual
  results, not a confidence interval**; the paired tables give win fractions, not significance tests.
* An accurate reconstructed field is **not** correct attribution: check `effective_rank` and `q_std`.
* The headline fits use a ridge (Tikhonov) prior with zero prior mean (it does not use the true emission rates) whose
  strength alpha = 0.1 was **selected on a separate validation ensemble** (seeds 44 / 9000; random and spread layouts
  only, because greedy layouts depend on alpha) and frozen in `results/alpha-selection/alpha_selected.json` before the
  test scenarios (seeds 33 / 7000) were run. `q_scale = 1e5` is only a numerical scaling: the penalty is
  alpha/q_scale^2 * ||q||^2, so only that ratio matters. The criterion is flat between alpha = 0.1 (0.01431) and
  alpha = 1 (0.01457), so 0.1 is not a sharp optimum.
* Headline results (details in `docs/validation.md`; mean test RMSE in ug/m^3, uniform weights, 18 rows per layout):
  greedy 0.0205, 0.0173, 0.0015, 4.5e-4, 4.1e-4 for 1, 2, 3, 5, 8 sensors; spread 0.0225, 0.0203, 0.0144, 0.0144, 0.0013;
  random (20 layouts) 0.0339, 0.0244, 0.0196, 0.0093, 0.0047. Greedy's advantage over spread is clear from 3 sensors
  (wins in 78-100% of the 18 paired rows) but is **not established at 1-2 sensors** (wins in 78% and 61%, mean difference
  smaller than the sd of the differences). At 2 sensors greedy's mean test RMSE is 2.6x its design RMSE (pooled 3.2x),
  so design-stage scores should not be quoted as expected performance. `greedy_weighted` is essentially identical to
  `greedy`. With alpha = 0 the greedy mean blows up at 1-2 sensors too (0.44 and 0.34), because a few rows are unconstrained.
* **Exploratory (superseded).** The first runs `results/benchmark`, `results/benchmark-unregularized` and
  `results/mismatch` (test seeds 22 / 5000, alpha = 1) chose alpha **after** the alpha = 0 test results had been seen.
  The earlier text calling that choice "a priori" was wrong. Those runs are not an independent evaluation and are
  kept only for the record; the evidence is in the `*-final` runs.
* Greedy selection is a heuristic, not an optimum (one tiny exhaustive test fixture showed greedy's design MSE at 3.7x the optimum, about 1.9x in RMSE).

## Known limitations

* **Numerical diffusion:** at the default 50 x 50 grid and dt = 60 s the scheme's effective diffusivity is about
  K x 1.61 (+0.61 K: +0.31 from space, +0.30 from the backward-Euler time term; measured, see
  `docs/specification.md`). Treat K as a *nominal* value. Time stepping is first order.
* v0 itself has no geography or terrain (phase 2 adds geography, with terrain as plotting context only), no observations, no
  chemistry, no forecasting, no neural solver (the `ForwardSolver` interface for plugging one in is described in
  `docs/extending.md`; nothing implements it yet; phase 2 adds FiPy variants and test wrappers only).
* Wind and K are uniform in space; the layer depth H is fixed.

## Change one assumption and compare

Configs can `extend` another file (relative path); dicts merge recursively, lists and scalars are replaced.

```bash
cat > configs/my.json <<'EOF'
{"extends": "reference.json", "name": "my-run",
 "scenario": {"transport": {"wind": [{"t_start": 0.0, "u": 0.4, "v": 0.9}]}}}
EOF
python -m pravah simulate --config configs/my.json --out results/my-run
python -c "
import xarray as xr
ref = xr.open_dataarray('results/reference-v0/concentration.nc')
new = xr.open_dataarray('results/my-run/concentration.nc')
print('max |change| %.3f ug/m3; final peak %.3f -> %.3f' % (abs(new - ref).max(), ref.isel(time=-1).max(), new.isel(time=-1).max()))"
```

When we ran this: max |change| 0.166 ug/m^3, final peak 0.161 -> 0.226. In a notebook, `(new - ref).isel(time=-1).plot()`
shows where the plume moved. Notebook 01 has the same idea as ready-made demos. More worked examples: `docs/extending.md`.

## Phase 2: solvers, external vs local, Belauri geography, geographic design

Everything below is a **synthetic or provisional simulation study**. **Real geography; synthetic emissions and meteorology.**
Nothing here is validated against measurements, none of it identifies a real emitter, and nothing is a field-ready deployment.

**What is new**

* **Solver interchangeability.** Every workflow runs through one injected forward-solver object (FiPy by default; its results are
  bit-identical to v0, see `docs/solver-interchangeability.md`). Truth and inference can use different solvers.
* **External vs local study.** Explicit external sources versus boundary (inflow) bases, origin tags, an external-fraction quantity,
  identifiability flags, and an **indoor box** (one-way coupled): `docs/transboundary.md`, notebook 04.
* **Belauri geography.** Municipality polygon, roads, buildings, facilities, DEM context, national outlines, prepared offline
  with a sha256/licence manifest: `docs/geography.md`. Forward simulations on it: `docs/belauri-forward.md`, notebook 05.
* **Geographic sensor design.** Candidate sites, random / spread / greedy-total / greedy-attribution layouts, with and without
  reference sites outside the polygon, held-out blocks, robustness: `docs/geodesign.md`, notebook 06.

Terms (used literally): *local* = emitted inside the study polygon; *external* = outside it (an explicit source or an input through
the outer boundary); *international transboundary* = crossing a verified national border (not asserted anywhere here); *indoor* = inside
a building. The study polygon is only a reporting mask, never a wall. See `docs/architecture.md`.

**Install the optional extra** (core code never needs it):

```bash
uv pip install -e ".[dev,geo]"          # geopandas, shapely, pyproj, rasterio, osmnx, requests
# exact tested versions: requirements-lock-geo.txt (drop its machine-specific "-e file://..." line, as above)
```

**Select a solver** (all of `simulate`, `inverse`, `benchmark`; defaults stay FiPy; in a config, `"backends": {"truth": {"spec": ..., "kwargs": {...}}, "inference": {...}}`):

```bash
python -m pravah benchmark --config configs/smoke.json --out results/my-up --inference-solver fipy --inference-solver-kwargs '{"convection":"upwind"}'
python -m pravah simulate  --config configs/smoke.json --out results/my-up2 --solver fipy --solver-kwargs '{"convection":"upwind"}'
python -m pravah inverse   --config configs/reference.json --out results/my-inv --solver mypkg.mod:make_solver --solver-kwargs '{"k":1}'
#   --solver SPEC (both roles)  --truth-solver SPEC  --inference-solver SPEC  --solver-kwargs JSON  (also --truth-/--inference-solver-kwargs)
```

`module:callable` is a factory called once; the solver must declare `capabilities["linear"]` to be used by the cached route. A worked, run
example is in `docs/extending.md` (examples 9-10).

**Fetch and prepare the geography** (network once, then offline):

```bash
python -m pravah geo fetch   --config configs/belauri_geo.json   # bounded, cached under data/belauri/raw/, sha256-recorded
python -m pravah geo prepare --config configs/belauri_geo.json   # offline: writes data/belauri/processed/ and manifest.json
python scripts/geo_report.py --manifest data/belauri/manifest.json --out results/geography   # offline figures + report
```

The OSM step tries a list of Overpass endpoints; in our acquisition the main endpoint (`overpass-api.de`) was unavailable and all four OSM
layers came from the `maps.mail.ru` mirror, which may lag the main database. After `prepare`, **the notebooks run offline**.

**Notebooks** (`scripts/verify_notebooks.py` runs one in a clean kernel; run times are single measurements, not benchmarks):

| Notebook | What it does | Run time |
|---|---|---|
| `04_transboundary_pollution.ipynb` | external vs local, identifiability, indoor box (smoke-sized grids) | about one minute (`docs/transboundary.md`) |
| `05_belauri_forward.ipynb` | Belauri forward simulations: winds, outer domain, mesh | not yet recorded here; see `docs/phase2-validation.md` |
| `06_belauri_sensor_design.ipynb` | candidate sites, layouts, robustness (runs both smoke profiles; shows saved production tables if present) | about 2.5 minutes (`docs/geodesign.md`) |

**Reproduce the studies** (fresh `--out` each time; frozen outputs are not to be overwritten):

```bash
python -m pravah benchmark --config configs/transboundary.json     --out results/<name>   # external vs local, production (27.8 min recorded)
python -m pravah benchmark --config configs/belauri_forward.json   --out results/<name>   # Belauri forward (285 s recorded)
python -m pravah benchmark --config configs/geodesign.json         --out results/<name>   # geographic design (708 s recorded)
python -m pravah benchmark --config configs/geodesign_robust.json  --out results/<name>   # needs robust.layouts_from = the geodesign run
# smoke profiles (plumbing only): configs/transboundary_smoke.json, belauri_forward_smoke.json, geodesign_smoke.json, geodesign_robust_smoke.json
```

**Inspect provenance.** Every run directory has `provenance.json` (git revision and dirty flag with a diff hash and dirty-file list,
library versions, seeds, the config, and in `extra` the backend infos, the sha256 of `data/belauri/manifest.json`, of the raw
geography files, of `alpha_selected.json` and the config hash) and `timings.json`. `data/belauri/manifest.json` lists every dataset (URL, release or commit,
retrieved UTC, licence and attribution, sha256, CRS, resampling), the boundary checks, counts and assumptions;
`tests/test_geo_real.py` re-hashes all files. `results/geodesign/alpha_selected.json` holds the alpha table chosen on validation
cases only. `provenance.json` flags a run made from a dirty tree.

**Caveats that travel with every figure and table:** "Real geography; synthetic emissions and meteorology." The DEM, hillshade and slope are plotting
context (no solver reads them); terrain-driven flow is not modelled. The scheme is numerically diffusive (effective diffusivity several times the nominal K at the
Belauri meshes), and layouts are greedy heuristics, not optima.

**What phase 2 does NOT establish**

* Any agreement with real concentrations, real emissions or real winds (no empirical validation).
* Identification of any real foreign or distant emitter, or any statement about a real border emission. The external patch is not asserted
  to be international.
* An authoritative boundary: the Belauri polygon is geoBoundaries ADM3 cross-checked against an OSM relation (IoU 0.9384), not a Survey
  Department product; the ADM2 coverage check fails (71.7 %) and is reported as informational.
* Terrain-resolved wind or mixing: relief inside the domain is a few tens of metres, the first substantial relief begins roughly 20 km
  beyond the municipality's northern edge, and terrain dynamics are unresolved.
* That boundary bases reproduce a localised external plume (they removed part, not all, of its error), or that sensor layouts chosen here
  beat simple ones by a clear margin (differences are modest and mixed; reference sites were not a clear win).
* That a mapped candidate site is permitted, powered, secure or connected.

Status of each requirement and measured results: `docs/phase2-validation.md`.

## Optional: Research Panel

The repo is also tracked with Research OS (`.research/`, see `AGENTS.md`). This is optional and the `pravah`
package never imports it. To register a finished run directory as artifacts and metrics:
`python scripts/register_research.py RUN-001 EXP-001 results/<dir>`.

## Layout

```
src/pravah/   config forward sources observations inverse metrics design experiments io plotting cli
              attribution transboundary indoor belauri geodesign geodesign_robust geo/(fetch prepare regionio scenarios)   [phase 2]
configs/      reference smoke smoke_alpha_selection alpha_selection benchmark benchmark_unregularized mismatch wind_change (JSON, "extends" supported)
              transboundary(_smoke) belauri_geo belauri_forward(_smoke) geodesign(_smoke) geodesign_robust(_smoke)   [phase 2]
notebooks/    01_transport 02_inverse 03_sensor_design   04_transboundary_pollution 05_belauri_forward 06_belauri_sensor_design
scripts/      verification.py  verify_notebooks.py  register_research.py  capture_regression.py compare_runs.py geo_report.py
              geo_relief_profile.py transboundary_tables.py
tests/        pytest suite (scientific checks, no mocks of the solver)
docs/         specification  architecture  implementation-plan  extending  validation  phase2-plan  solver-interchangeability
              transboundary  geography  belauri-forward  geodesign  phase2-validation
results/      frozen outputs (see docs/validation.md and docs/phase2-validation.md; benchmark, benchmark-unregularized, mismatch are
              exploratory/superseded)
data/belauri  prepared geography (manifest.json tracked; raw downloads git-ignored)
```

Checks: `pytest -q`, `pyright`, `python scripts/verify_notebooks.py` (status in `docs/validation.md`).
