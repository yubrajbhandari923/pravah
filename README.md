# PRAVAH

An **idealised, synthetic** testbed for regional air-quality work: a 2-D advection-diffusion-removal model (FiPy),
a source-amplitude inverse problem, and a sensor-placement benchmark. The region we eventually care about is
Belauri (Nepal), but **nothing in this repository is a Belauri estimate**: domain size, wind, diffusivity, removal
rate, source positions and amplitudes are demonstration values chosen to make plumes visible on a 10 km x 10 km box.
There is no real geography, no observations, no chemistry and no forecasting. Everything is verified against
mathematics (exact solutions, conservation, refinement), not against measurements.

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
python -m pravah benchmark --config configs/benchmark.json --out results/my-bench          # ~5 min
python -m pravah benchmark --config configs/mismatch.json  --out results/my-mismatch       # ~8 min
```

`configs/mismatch.json` re-uses the layouts of a finished benchmark: it reads them from `mismatch.layouts_from`
(`results/benchmark`); if you wrote your benchmark elsewhere, extend the config and change that path.
The `benchmark` command runs either experiment depending on the config's `"experiment"` key.

**`results/reference-v0` (and `inverse-v0`, `smoke`, `benchmark*`, `mismatch`, `verification`) are the frozen
v0 results quoted in `docs/validation.md`. Do not overwrite them: write alternatives to `results/<your-name>`.**

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

* **Matched benchmark** (`results/benchmark`): truth and inference use the *same* solver and grid (an "inverse
  crime"), with the true wind given to the inversion. It shows what sensor layout can do when the model is right.
  It does not show performance on real data.
* **Mismatch study** (`results/mismatch`): fixed layouts tested when the truth differs in one way at a time (finer
  grid and smaller dt; hidden +20% wind speed and 15 degree rotation; an unmodelled 4th source). One magnitude per kind.
* Counts are modest (6 design + 6 test scenarios, 3 noise draws). The reported **sd is the spread of individual
  results, not a confidence interval**; the paired tables give win fractions, not significance tests.
* An accurate reconstructed field is **not** correct attribution: check `effective_rank` and `q_std`.
* The headline fits use a weak ridge prior (alpha = 1). That choice was made **after** seeing the alpha = 0 run
  fail on plume-blind layouts (means of 1e19); it was not tuned, and the alpha = 0 run is kept as a sensitivity result.
* Greedy selection is a heuristic, not an optimum (one tiny exhaustive test fixture showed greedy's design MSE at 3.7x the optimum, about 1.9x in RMSE).

## Known limitations

* **Numerical diffusion:** at the default 50 x 50 grid and dt = 60 s the scheme's effective diffusivity is about
  K x 1.61 (+0.61 K: +0.31 from space, +0.30 from the backward-Euler time term; measured, see
  `docs/specification.md`). Treat K as a *nominal* value. Time stepping is first order.
* No real geography or terrain, no observations, no chemistry, no forecasting, no neural solver (the
  `ForwardSolver` interface for plugging one in is described in `docs/extending.md`; nothing implements it yet).
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

## Optional: Research Panel

The repo is also tracked with Research OS (`.research/`, see `AGENTS.md`). This is optional and the `pravah`
package never imports it. To register a finished run directory as artifacts and metrics:
`python scripts/register_research.py RUN-001 EXP-001 results/<dir>`.

## Layout

```
src/pravah/   config forward sources observations inverse metrics design experiments io plotting cli
configs/      reference smoke benchmark benchmark_unregularized mismatch wind_change (JSON, "extends" supported)
notebooks/    01_transport 02_inverse 03_sensor_design
scripts/      verification.py  verify_notebooks.py  register_research.py
tests/        pytest suite (scientific checks, no mocks of the solver)
docs/         specification  architecture  implementation-plan  extending  validation
results/      frozen v0 outputs (see docs/validation.md)
```

Checks: `pytest -q`, `pyright`, `python scripts/verify_notebooks.py` (status in `docs/validation.md`).
