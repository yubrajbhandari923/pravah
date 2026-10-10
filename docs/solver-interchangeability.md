# Solver interchangeability

Every workflow (simulate, direct fit, response-basis creation, greedy case building, benchmark, alpha selection,
mismatch truth **and** mismatch inference, CLI) runs through one injected forward solver object. FiPy is the default
(`pravah.forward.FIPY`) and its results are bit-identical to the pre-interface code (`REGRESSION_RESULT` below).

## Supported paths

| Where | How a solver is passed | Default |
|---|---|---|
| `inverse.compute_responses`, `ResponseCache.get`, `fit_sources`, `reconstruct`, `run_forward` | `forward=` | `FIPY` |
| `design.build_case` | `forward=` (inference responses **and** the truth run unless `truth_field` is given) | `FIPY` |
| `experiments.run_benchmark`, `run_alpha_selection`, `run_mismatch` | `truth_forward=`, `inference_forward=` | from config, else `FIPY` |
| config file | `"backends": {"truth": {"spec": ..., "kwargs": {...}}, "inference": {...}}` (separate from `"solver"`, which is still the `SolverOptions` kwargs) | `fipy` / `fipy` |
| CLI `simulate`, `inverse`, `benchmark` | `--solver SPEC` (both roles), `--truth-solver SPEC`, `--inference-solver SPEC`, `--solver-kwargs JSON`, `--truth-solver-kwargs JSON`, `--inference-solver-kwargs JSON` | FiPy |
| `scripts/verification.py` and notebooks 01-03 | call the FiPy reference explicitly (`FIPY` / `simulate`) | FiPy by design |

Roles: the *truth* solver generates every synthetic truth (matched, refined-grid, wind-perturbed, extra-source);
the *inference* solver builds response bases, cases, fits and reconstructions. `simulate` (the function) is the FiPy
implementation entry and is recognised as `FIPY`. Explicit `truth_forward`/`inference_forward` arguments win over the
config; CLI flags win over `backends` in the config. The solvers are resolved once and passed down as objects; no
internal function falls back to FiPy when a solver was supplied.

## Invocation

```bash
# default (FiPy)
python -m pravah benchmark --config configs/smoke.json --out runs/smoke
# an alternative numerical scheme of FiPy, for the inference role only
python -m pravah benchmark --config configs/smoke.json --out runs/up \
    --inference-solver fipy --inference-solver-kwargs '{"convection": "upwind"}'
# a custom solver factory for both roles
python -m pravah inverse --config configs/reference.json --out runs/inv \
    --solver mypkg.surrogate:make_solver --solver-kwargs '{"weights": "w.npz"}'
```

```python
from pravah.forward import FIPY, FipyForward, get_forward
from pravah.experiments import run_benchmark
run_benchmark(cfg, out, truth_forward=FIPY, inference_forward=FipyForward(convection="upwind"))
solver = get_forward("mypkg.surrogate:make_solver", {"weights": "w.npz"})
```

`get_forward(spec, kwargs)`: `None`/`"fipy"` gives `FIPY` (or `FipyForward(**kwargs)`); `"module:callable"` imports the
callable, which is a **factory**: it is called once with `**kwargs` and must return the solver. Malformed specs,
import failures, factory errors and non-callable results raise `ValueError`. With `--solver` alone both roles share
one object; role-specific flags create separate ones.

## BackendInfo and capabilities

`forward_info(f)` returns `BackendInfo(name, version, settings, capabilities)` (`.to_dict()`): `f.info` if present;
`simulate` is `FIPY`; otherwise it is derived (`name = module.qualname`, `version = "unknown"`, `settings =
{"forward_id": f.forward_id}`, capabilities from an optional `f.capabilities` dict). Capability keys, all `False`
unless declared:

* `linear`: superposition of source responses is valid. **Explicit opt-in, never assumed**: `ResponseCache.get`
  and `compute_responses` raise `ValueError` otherwise (this includes the cached operator used for the diagnostics of
  `method="direct"`).
* `budget`: the result carries a mass budget.
* `boundary_bases`: supports boundary-inflow basis parameters (`InflowBasis`, theta entries of kind `inflow`). `True` for the FiPy reference since the phase-2 physics landed; a replacement solver that does not set it is refused for scenarios with bases.

`FipyForward(convection=None, tolerance=None)` applies these overrides on top of the `SolverOptions` of each call and
records them in `info.settings`; `FIPY.info.version` is `"fipy <ver>; pravah <ver>"`. FiPy results carry
`FieldResult.diagnostics["backend"]` (the BackendInfo dict); this does not change any numerical output.

## What a replacement must return

A `FieldResult` from `solver(scenario, q, solver_options=SolverOptions())` with `concentration` dims
`("time", "y", "x")`, coords `time = scenario.output_times`, `x`/`y` = cell centres (`sources.cell_centres`),
`attrs["units"] == "microgram/m^3"`, finite values, and a `str` `status`. `budget` may be `None`; `runtime_s` and
`diagnostics` should be filled. `inverse.run_forward` checks every result (FiPy included) with
`forward.check_field_result` and fails with a clear `ValueError`.

## Cache keys

`inverse.cache_key` always includes `forward_info(forward).to_dict()` (name, version, settings, capabilities) next to
the scenario, `SolverOptions`, schema, `q_ref` and the pravah version. Different solvers, versions or settings
(e.g. `FipyForward(convection="upwind")`) never share in-memory or on-disk entries. Keys of FiPy responses differ from
those written before this change (old `.npz` entries are simply not found and are recomputed).

## Provenance

`run_benchmark`, `run_alpha_selection` and `run_mismatch` write both BackendInfo dicts to
`provenance.json["extra"]["backends"]`; benchmark and mismatch also write them to `summary.json["backend"]`. The CLI
`simulate` records `extra.backend`, `inverse` records `extra.backends`.

## Limits

* A replacement only needs to support the scenarios it was built for; unsupported ones should raise.
* No automatic differentiation; `method="direct"` uses finite-difference Jacobians of the injected solver.
* `boundary_bases` and `wind_field` (stream-function winds) are `True` for FiPy; a replacement solver declares what it supports in its `BackendInfo`.
* Declaring `linear=True` is the author's claim; the framework does not test linearity.
* Test doubles in `tests/helpers_solver.py` (recording / +1 ug/m^3 marker wrappers around FiPy) are interface tests,
  not physics benchmarks.

## Regression comparison

Measured by the coordinator (EXP-008, RUN-0012 before / RUN-0013 after, `scripts/capture_regression.py` +
`scripts/compare_runs.py`; code revisions `cb8146e` -> `699b76e`, fixed seeds, default FiPy):

* **21 files compared, 0 differ** — reference field (`concentration.nc`, `budget.nc`), `metrics.json`, inverse
  (linear and direct, `inverse_metrics.json`), smoke benchmark (raw metrics, summaries, paired tables, `layouts.json`),
  smoke alpha selection (`alpha_selected.json`, `validation_raw.csv`) and smoke mismatch (raw, paired, poor-layout tables).
* Excluded as volatile: runtime/wall-time keys and columns, timestamps, git state, paths, `provenance.json`,
  `timings.json`, `config.json`, `manifest.json`, figures, and the new `backend` entries (additive metadata).
* The baseline itself is deterministic: two independent captures of the *old* code also compared 21 files, 0 differ.
* Negative control: running the same simulate job with `FipyForward(convection="upwind")` (a real alternate FiPy
  configuration) is flagged — max concentration 0.1608 -> 0.1517 ug/m^3 and different budget terms — so the comparison
  can detect differences. This control is not a physics claim about either scheme.
* Result files: `results/solver-regression/{before,after}/manifest.json`, `results/solver-regression/comparison.json`.
* Tests: 218 passed (196 earlier + 22 in `tests/test_solver_routing.py`), Pyright 0 errors.
