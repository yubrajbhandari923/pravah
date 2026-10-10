# Architecture

`src/pravah/`: `config` (frozen dataclasses, JSON loading, `FieldResult`), `sources`/`forward` (FiPy
solver behind `simulate`; `ForwardSolver` protocol and `check_field_result` define what a replacement solver must provide), `observations` (`sample`, `observe`), `inverse` (`fit_sources`, `reconstruct`,
`ResponseCache`), `metrics`, `design` (`score_layout`, `select_layout`, `evaluate_layouts`), `experiments`
(benchmark / mismatch runners), `io`, `plotting`, `cli`.

Data flow: Scenario + q -> `simulate` -> `FieldResult` (xarray, no FiPy objects) -> `observe` -> `Observations`
-> `fit_sources` -> `FitResult` -> `reconstruct` -> `FieldResult` -> `metrics`.
A replacement solver (e.g. a neural surrogate) only has to satisfy `forward.ForwardSolver`: `simulate`'s return type (same
quantities, coordinates, times). `inverse` and `design` take it as the optional `forward=` keyword; see `extending.md`.
Research Panel is used only from `scripts/`, never imported by the package.
See `implementation-plan.md` for the discretisation and conventions.
User docs: `../README.md`, `extending.md` (worked examples), `validation.md` (measured results and tolerances).
Scripts: `scripts/verification.py` (refinement studies), `scripts/verify_notebooks.py`, `scripts/register_research.py`.
