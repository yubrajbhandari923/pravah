# Extending PRAVAH

Worked examples, each run in `.venv` against the code in this repository (small grids; every Python block takes
seconds, the last one about 15 s). Run them from the repository root. Blocks marked `# example N` are
self-contained scripts. Nothing here changes `src/` except section 2, which shows a patch you apply yourself.

Conventions to remember: `Scenario` holds everything except the emission rates `q` (ug/s), which are passed to
`simulate(scenario, q)` separately; all dataclasses are frozen, so change them with `dataclasses.replace`; fields
are `xarray` objects with dims `(time, y, x)` and cell-centre coordinates in metres.

## 1. Change a source (position, width, time profile)

```python
# example 1
import dataclasses as dc
import numpy as np
from pravah.config import GaussianSource, load_config, scenario_from_dict
from pravah.forward import simulate

cfg = load_config("configs/reference.json")
cfg["scenario"]["domain"].update(nx=25, ny=25)                       # coarse grid: seconds, not minutes
cfg["scenario"]["times"].update(duration=7200.0, dt=120.0)
sc = scenario_from_dict(cfg["scenario"])
q = np.array(cfg["q"])

# (a) move S1 and widen it
s1 = dc.replace(sc.sources[0], x=3000.0, y=2000.0, sigma=800.0)
sc_a = dc.replace(sc, sources=(s1, *sc.sources[1:]))

# (b) S2 only emits between 1800 s and 3600 s (profile = ((t_start, factor), ...); first t_start must be 0)
s2 = dc.replace(sc.sources[1], profile=((0.0, 0.0), (1800.0, 1.0), (3600.0, 0.0)))
sc_b = dc.replace(sc, sources=(sc.sources[0], s2, sc.sources[2]))

for name, s in [("reference", sc), ("moved S1", sc_a), ("S2 pulse", sc_b)]:
    r = simulate(s, q)
    print(f"{name:10s} emitted {float(r.budget['emission'].sum()):.3e} ug, "
          f"max c {float(r.concentration.max()):.4f}, budget residual {r.diagnostics['max_rel_budget_residual']:.1e}")
```

Output when run:

```
reference  emitted 2.376e+09 ug, max c 0.1615, budget residual 2.3e-15
moved S1   emitted 2.376e+09 ug, max c 0.1607, budget residual 2.9e-15
S2 pulse   emitted 1.566e+09 ug, max c 0.1401, budget residual 3.1e-15
```

Check by hand: S2 loses 1.5e5 ug/s x 5400 s = 8.1e8 ug, and 2.376e9 - 8.1e8 = 1.566e9. Moving or widening a
source does not change the emitted mass (the shape is normalised on the grid). The step plan inserts a time step
boundary at every profile breakpoint, so the pulse edges are exact; the profile is right-continuous (a breakpoint
belongs to the later segment). For a JSON config put `"profile": [[0, 0], [1800, 1], [3600, 0]]` on the source.
If you add or remove a source, `q` and any inversion `q_scale`/`prior` must have the same length.

## 2. Add a physical term (example: a second first-order removal, "deposition")

A term that is linear in `c` and independent of `q` keeps the model affine in `q`, so the cached-response inverse
and the sensor-design code remain valid. Four edits, then a test. (The patch below was applied to a scratch copy of
`src/pravah` and the test passed; it is **not** applied in the repository.)

```diff
--- src/pravah/config.py
-SCHEMA_VERSION = "1"
+SCHEMA_VERSION = "2"        # forward-relevant behaviour changed: part of every response-cache key
@@ class Transport
     H: float = 500.0  # m
+    dep: float = 0.0  # 1/s, extra first-order removal; appended LAST so positional Transport(...) calls still work
 
     def __post_init__(self) -> None:
+        _need(self.dep >= 0, "dep must be >= 0")
--- src/pravah/forward.py                                   (class _Segment, the ImplicitSourceTerm coefficient)
-        coef = fp.CellVariable(mesh=grid, value=scenario.transport.lam + (self.a_out + self.a_in).reshape(-1))
+        coef = fp.CellVariable(mesh=grid, value=scenario.transport.lam + scenario.transport.dep
+                               + (self.a_out + self.a_in).reshape(-1))
@@ in simulate(), the budget accumulation
-        acc["removal"][i_out] += h * tr.H * tr.lam * float(c.sum()) * dA
+        acc["removal"][i_out] += h * tr.H * (tr.lam + tr.dep) * float(c.sum()) * dA
```

Why these four places and nothing else:

* **The equation** (`ImplicitSourceTerm`) removes `coef * c` implicitly; `lam` and the open-boundary coefficients
  `a_out`, `a_in` already share it, so the new rate is one more addend.
* **The budget** must use the same rate, otherwise `residual = mass_change - (emission - removal + inflow - outflow)`
  stops closing. Always re-check `diagnostics["max_rel_budget_residual"]` after touching physics.
* **Config and cache.** `Transport(**dict)` in `scenario_from_dict` picks up the new field from JSON automatically
  (`"transport": {"dep": 1e-4}`), and `to_dict` is `dataclasses.asdict`, so the new field is hashed into
  `inverse.cache_key` with no extra code. We checked that the key differs for `dep=0` and `dep=1e-4`. Bump
  `SCHEMA_VERSION` anyway: it invalidates disk caches (`ResponseCache(directory=...)`) written by the old physics
  when you did *not* change a field value (e.g. you fixed a bug), and it marks the change.
* **`is_linear`.** `inverse.is_linear(scenario)` returns `True` today and the cached route calls it. A first-order
  removal is linear, so leave it. If you add anything nonlinear in `c` (for example removal proportional to `c**2`,
  or emissions that depend on the concentration), make `is_linear` return `False` for the affected scenarios: the
  cached/design route then refuses with a clear error and only `InferenceOptions(method="direct")` is valid.
  Superposition, `ResponseCache`, `design.py` and the noise-free-recovery claims all assume linearity.

The verification test (add as `tests/test_deposition.py`): in a periodic box a uniform field decays by exactly
`(1 + (lam + dep) dt)**-n` under backward Euler, and the budget still closes.

```python
# tests/test_deposition.py
import numpy as np
from pravah.config import Boundary, Domain, Initial, Scenario, Times, Transport, WindSegment
from pravah.forward import simulate


def test_uniform_decay_with_deposition():
    dt, n, lam, dep = 100.0, 10, 1e-3, 2e-3
    sc = Scenario(domain=Domain(10000.0, 10000.0, 10, 10),
                  transport=Transport((WindSegment(0.0, 1.0, 0.3),), 100.0, lam, 500.0, dep),
                  boundary=Boundary("periodic"), initial=Initial("uniform", 3.0),
                  times=Times(duration=n * dt, dt=dt, output_interval=n * dt))
    r = simulate(sc, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(r.concentration.values[-1], 3.0 * (1 + (lam + dep) * dt) ** -n, rtol=1e-10)
    assert r.diagnostics["max_rel_budget_residual"] < 1e-9   # budget "removal" uses lam + dep too
```

It passes (1 test, 0.7 s) on the patched copy. This is the same trick `tests/test_forward.py` uses for `lam` alone.
Also add the new field to `tests/test_inverse.py::test_cache_key_changes_with_scenario`'s parametrisation.

## 3. Time-varying wind (piecewise-constant segments; this extension already exists)

`Transport.wind` is a tuple of `WindSegment(t_start, u, v)`; the first must start at 0 and times must increase.
`configs/wind_change.json` is the config form: it `extends` the reference and replaces the `wind` list
(lists are replaced, not merged) with three segments.

```python
# example 3
import dataclasses as dc
import numpy as np
from pravah.config import WindSegment, load_config, scenario_from_dict
from pravah.forward import simulate, step_times

cfg = load_config("configs/wind_change.json")        # extends reference.json; 3 wind segments
cfg["scenario"]["domain"].update(nx=25, ny=25)
cfg["scenario"]["times"].update(dt=120.0)
sc = scenario_from_dict(cfg["scenario"])
q = np.array(cfg["q"])

# the same idea built in Python: a wind that turns at t = 5400 s
wind = (WindSegment(0.0, 1.0, 0.3), WindSegment(5400.0, -0.6, 0.8))
sc2 = dc.replace(sc, transport=dc.replace(sc.transport, wind=wind))

res = simulate(sc2, q)
b = res.budget
print("steps:", res.diagnostics["n_steps"], " 5400 s is a step boundary:", bool(np.isclose(step_times(sc2), 5400.0).any()))
for t0 in (4800.0, 5100.0, 5400.0, 5700.0, 6000.0):
    i = int(np.argmin(abs(b["t_start"].values - t0)))
    print(f"[{t0:6.0f}, {t0 + 300:6.0f}) s  outflow {float(b['outflow'][i]):.3e}  inflow {float(b['inflow'][i]):+.3e}  "
          f"residual {float(b['residual'][i]):+.1e} ug")
print("max relative budget residual:", res.diagnostics["max_rel_budget_residual"])
```

Output when run:

```
steps: 144  5400 s is a step boundary: True
[  4800,   5100) s  outflow 2.872e+07  inflow -2.570e+04  residual -1.5e-07 ug
[  5100,   5400) s  outflow 3.249e+07  inflow -2.709e+04  residual -1.9e-07 ug
[  5400,   5700) s  outflow 2.905e+06  inflow -1.040e+07  residual -3.3e-07 ug
[  5700,   6000) s  outflow 4.729e+06  inflow -5.158e+06  residual -3.9e-07 ug
[  6000,   6300) s  outflow 7.396e+06  inflow -2.647e+06  residual -4.1e-07 ug
max relative budget residual: 5.268087290754222e-15
```

What changes at a switch:

* The step plan splits at the switch time (here as at any output time and emission breakpoint), so no step straddles
  the discontinuity. A switch need not be a multiple of the output interval (`test_wind_change_off_output_grid_runs`).
* Face classification is redone per segment: a face is inflow if `u.n < 0`, outflow if `u.n > 0`. After the turn above
  the east and north faces, which exported the plume, become **inflow faces with prescribed background (0)**. Mass in
  the boundary cells then leaves through that diffusive Dirichlet term, which is why the `inflow` budget term turns
  strongly **negative** (-1.0e7 ug) while `outflow` drops by an order of magnitude. `inflow` is signed and can be negative.
* The budget still closes (relative residual 5e-15) because the budget uses the same boundary coefficients as the equation.
* Each segment's FiPy equation is built once and reused, so many segments cost little extra.
* The inverse problem must be told the wind: it is "known". In `make_ensemble`, every second benchmark scenario uses a
  two-segment wind.

## 4. Substitute a forward solver

`pravah.forward.ForwardSolver` is a `Protocol`: any callable `solver(scenario, q, solver_options) -> FieldResult`.
The inverse and design code take an optional keyword `forward=simulate` (`ResponseCache.get`,
`compute_responses`, `fit_sources`, `reconstruct`, `design.build_case`). The CLI and `run_benchmark` /
`run_mismatch` do not take it yet; call the library functions as below.

**What it must return** (`check_field_result(res, scenario)` raises `ValueError` with a specific message otherwise;
the inverse code calls it for every non-default solver):

* `res.concentration`: `xr.DataArray`, dims exactly `("time", "y", "x")`, coords `time = scenario.output_times`,
  `x`, `y` = `sources.cell_centres(scenario)`, `attrs["units"] == "microgram/m^3"`, all values finite;
* `res.status` a `str`; `res.runtime_s`, `res.diagnostics` present. `res.budget` may be `None`.

**What it need not support:** negative `q`, the `initial_field` argument, other grids than the one it was built for
(raise a clear `ValueError` instead), budgets, boundary modes it was not trained on. A solver only needs to cover
its scenario family.

**What it should be:** the cached inverse and the design code assume the output is affine in `q`
(`base + sum_j q_j g_j`) and compute `g_j` from n+1 calls. A surrogate that is not affine in `q` makes the fast route
disagree with its own `direct` route; use `InferenceOptions(method="direct")` for such a solver, and expect the
sensor-design speed-up to be lost.

**Cache identity.** The response-cache key includes the solver's `module.qualname` and an optional `forward_id`
string attribute (give it a weights hash or version), in addition to scenario, solver options, `SCHEMA_VERSION`,
package and FiPy versions. The default `simulate` keeps the old keys. Two different surrogates with the same class
name and no `forward_id` would share cache entries; set `forward_id`.

```python
# example 4
import numpy as np
import xarray as xr
from pravah.config import FieldResult, SolverOptions, load_config, scenario_from_dict
from pravah.forward import check_field_result, simulate
from pravah.inverse import InferenceOptions, ResponseCache, cache_key, fit_sources
from pravah.observations import ObservationSpec, SensorLayout, observe
from pravah.sources import cell_centres

cfg = load_config("configs/reference.json")
cfg["scenario"]["domain"].update(nx=20, ny=20)
cfg["scenario"]["times"].update(duration=3600.0, dt=120.0)
sc = scenario_from_dict(cfg["scenario"])
n = len(sc.sources)


class TableSurrogate:
    """Stand-in for a trained surrogate: replays stored unit responses.  Valid for ONE scenario family only."""
    forward_id = "table-v1"                       # appears in the response-cache key

    def __init__(self, scenario):
        self.scenario = scenario
        self.g = [simulate(scenario, np.eye(n)[j] * 1e5).concentration.values / 1e5 for j in range(n)]

    def __call__(self, scenario, q, solver=None):
        if scenario.domain != self.scenario.domain or scenario.times != self.scenario.times:
            raise ValueError("this surrogate was built for another grid / time axis")
        x, y = cell_centres(scenario)
        data = np.tensordot(np.asarray(q, float), np.stack(self.g), axes=1)          # (time, y, x)
        conc = xr.DataArray(data, dims=("time", "y", "x"), name="concentration",
                            coords={"time": scenario.output_times, "y": y, "x": x},
                            attrs={"units": "microgram/m^3"})
        return FieldResult(conc, None, "ok", 0.0, {})                                 # budget=None is allowed


sur = TableSurrogate(sc)
q_true = np.array(cfg["q"])
check_field_result(sur(sc, q_true), sc)           # raises ValueError with a clear message if the contract is broken
print("contract ok; cache keys differ:", cache_key(sc, SolverOptions(), 1e5) != cache_key(sc, SolverOptions(), 1e5, sur))

bad = sur(sc, q_true)
bad.concentration.attrs["units"] = "ppb"
try:
    check_field_result(bad, sc)
except ValueError as e:
    print("rejected:", e)

layout = SensorLayout(("a", "b", "c"), (4000.0, 6000.0, 7500.0), (3500.0, 5000.0, 4500.0))
truth = simulate(sc, q_true).concentration
obs = observe(truth, layout, ObservationSpec((600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0), 0.005, seed=1))
fit = fit_sources(sc, obs, InferenceOptions(alpha=1.0), ResponseCache(), forward=sur)   # the surrogate supplies A and b
print("q_hat", fit.q_hat.round(0), "q_true", q_true, " effective rank", fit.effective_rank)
```

Output when run: `contract ok; cache keys differ: True`, then
`rejected: concentration attrs['units'] must be 'microgram/m^3', got 'ppb'`, then
`q_hat [ 98180. 147662.  80785.] q_true [100000. 150000.  80000.]  effective rank 3`.
The recovery is good only because this surrogate replays the very solver that made the data (the inverse
crime again); a real surrogate must be judged against `simulate` on held-out scenarios first, and against the
mismatch tables in `validation.md` for how model error propagates.

## 5. Change the inference (alpha, prior, `q_scale`, held-out sensors, linear vs direct)

`InferenceOptions(method, q_scale, alpha, prior, heldout_ids, solver, q_ref)`. The optimiser works in
`theta = q / q_scale >= 0` and minimises `||(b + A q - y)/sigma||^2 + alpha ||theta - prior/q_scale||^2`, so **alpha is
measured in units of `q_scale`**: changing `q_scale` changes the effective prior strength (only alpha/q_scale^2 matters).
The `alpha=1` values in the examples below are demo values; the benchmark's alpha (0.1) is selected on a validation ensemble (`results/alpha-selection`).

```python
# example 5
import time
import numpy as np
from pravah.config import load_config, scenario_from_dict
from pravah.forward import simulate
from pravah.inverse import InferenceOptions, ResponseCache, fit_sources
from pravah.observations import ObservationSpec, SensorLayout, observe

cfg = load_config("configs/reference.json")
cfg["scenario"]["domain"].update(nx=20, ny=20)
cfg["scenario"]["times"].update(duration=3600.0, dt=120.0)
sc, q_true = scenario_from_dict(cfg["scenario"]), np.array(cfg["q"])
truth = simulate(sc, q_true).concentration

# S1..S3 sit downwind of the sources, S4 sits in the upwind corner (it sees almost nothing)
layout = SensorLayout(("S1", "S2", "S3", "S4"), (4000.0, 6000.0, 7500.0, 500.0), (3500.0, 5000.0, 4500.0, 9500.0))
obs = observe(truth, layout, ObservationSpec(tuple(np.arange(600.0, 3601.0, 600.0)), 0.005, seed=1))
cache = ResponseCache()                       # reuse it: the n_src + 1 forward solves happen once

def show(label, **kw):
    f = fit_sources(sc, obs, InferenceOptions(**kw), cache)
    print(f"{label:34s} q_hat={f.q_hat.round(0)}  eff.rank={f.effective_rank}  q_std={f.q_std.round(0)}  [{f.runtime_s:.2f} s]")
    return f

show("default (alpha 0)")
show("alpha=1 (weak ridge toward 0)", alpha=1.0)
show("alpha=1, prior = 80% of truth", alpha=1.0, prior=tuple(0.8 * q_true))
show("alpha=1, q_scale 1e4", alpha=1.0, q_scale=(1e4,) * 3)
show("S1 held out", heldout_ids=("S1",))
lin = show("linear, S4 held out", heldout_ids=("S4",))
t0 = time.perf_counter()
dr = fit_sources(sc, obs, InferenceOptions(method="direct", heldout_ids=("S4",)), cache)
print(f"direct (least_squares on forward solves) {time.perf_counter() - t0:.1f} s; "
      f"max rel diff to linear {np.max(np.abs(dr.q_hat - lin.q_hat) / lin.q_hat):.1e}")
```

Output when run (fits take ~1 s the first time, because of the response solves, then milliseconds):

```
default (alpha 0)                  q_hat=[100005. 145545.  80207.]  eff.rank=3  q_std=[ 5278. 25466.  4961.]
alpha=1 (weak ridge toward 0)      q_hat=[ 99875. 136952.  80339.]  eff.rank=3  q_std=[ 5269. 24678.  4950.]
alpha=1, prior = 80% of truth      q_hat=[ 99976. 144044.  80225.]  eff.rank=3  q_std=[ 5269. 24678.  4950.]
alpha=1, q_scale 1e4               q_hat=[80593. 22695. 69295.]  eff.rank=2  q_std=[ 4657. 9303. 4390.]
S1 held out                        q_hat=[150517. 127662.  78275.]  eff.rank=2  q_std=[95871. 42382.  6165.]
linear, S4 held out                q_hat=[100005. 145545.  80207.]  eff.rank=3  q_std=[ 5278. 25466.  4961.]
direct (least_squares on forward solves) 6.8 s; max rel diff to linear 6.6e-09
```

Read it as: a held-out sensor is simply excluded (S1 removed -> S1's amplitude is badly determined, `q_std` 9.6e4,
`effective_rank` 2). The same `alpha=1` that is harmless with `q_scale=1e5` pulls hard when `q_scale=1e4`, because the
data are then only weakly informative relative to the penalty. `q_std` is a linear-Gaussian approximation that ignores
the `q >= 0` bound. The direct route reproduces the linear route to ~1e-8 at 7 s instead of milliseconds; use it
only to check or when the model is not linear. `heldout_ids` must name sensors present in the observations.

## 6. Add a placement strategy

A strategy is any function returning a `Layout`: a sorted tuple of unique candidate indices. Plug it into
`design.evaluate_layouts`, which accepts only `role="test"` cases (so it cannot be tuned on the test data) and returns a
long table. Use the greedy and spread layouts from `design` for reference.

```python
# example 6
import numpy as np
from pravah.config import load_config, scenario_from_dict
from pravah.design import (Constraints, DesignOptions, Layout, build_case, candidate_grid, evaluate_layouts,
                           select_layout, spread_layout)
from pravah.experiments import make_ensemble
from pravah.inverse import ResponseCache

cfg = load_config("configs/smoke.json")                       # 25x25 grid, 7200 s, dt 120 s
base, q0 = scenario_from_dict(cfg["scenario"]), cfg["q"]
cands = candidate_grid(base.domain, n=5)                      # 5 x 5 candidate sites; Layout = tuple of indices
opts = DesignOptions(noise_std=0.005, alpha=1.0)
cache = ResponseCache()

def cases(role, seed, noise_seed0, n=2):
    return [build_case(s.name, role, s.scenario, np.array(s.q), s.scenario, cands, opts, noise_seed0 + i, 2, cache)
            for i, s in enumerate(make_ensemble(base, n, seed, q0, prefix=role))]

design, test = cases("design", 11, 1000), cases("test", 22, 5000)    # disjoint seeds: the guard checks this

# ---- two home-made strategies: a function (candidates, budget) -> Layout (sorted tuple of unique indices) ----
def farthest_from_sources(cands, budget) -> Layout:
    src = np.array([(s.x, s.y) for s in base.sources])
    d = np.hypot(cands.x[:, None] - src[:, 0], cands.y[:, None] - src[:, 1]).min(axis=1)
    return tuple(sorted(int(i) for i in np.argsort(-d)[:budget]))

def row_at(y):
    def strategy(cands, budget) -> Layout:
        on_row = np.flatnonzero(np.isclose(cands.y, y))
        return tuple(sorted(int(i) for i in on_row[:budget]))
    return strategy

budget = 3
layouts = {
    "greedy": [select_layout(cands, budget, design, Constraints(), opts)],       # sees design cases only
    "spread": [spread_layout(cands, budget)],
    "farthest": [farthest_from_sources(cands, budget)],
    "row_y5000": [row_at(5000.0)(cands, budget)],
}
df = evaluate_layouts(layouts, test, opts)                    # test cases only (role guard), fresh noise
print(df.groupby("method")[["rmse", "source_rel_err", "effective_rank"]].mean().round(4).sort_values("rmse"))
print({m: [cands.ids[i] for i in l[0]] for m, l in layouts.items()})
```

Output when run:

```
             rmse  source_rel_err  effective_rank
method
greedy     0.0011          0.0294             3.0
spread     0.0147          0.3664             2.0
row_y5000  0.0150          0.3771             2.0
farthest   0.0241          0.5949             1.0
{'greedy': ['c008', 'c014', 'c017'], 'spread': ['c000', 'c004', 'c012'], 'farthest': ['c019', 'c020', 'c024'], 'row_y5000': ['c010', 'c011', 'c012']}
```

This is a 2-design / 2-test-scenario demonstration on a coarse grid, **not** a result: with so few cases the numbers
only show that the pipeline works (the real comparison is `results/benchmark-final`, 6 + 6 scenarios on the full grid, with alpha = 0.1 selected on separate validation scenarios).
A naive "far from the sources" rule is worst because it puts sensors where the plumes are not (effective rank 1).
To get paired statistics, pass several methods and use `design.paired_comparison(df, "greedy", "row_y5000")`.
To respect separation or exclusion zones use `Constraints(allowed=mask, min_separation=...)` and
`design.feasible_sites`; custom strategies must apply such constraints themselves.
