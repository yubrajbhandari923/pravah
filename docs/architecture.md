# Architecture

`src/pravah/`: `config` (frozen dataclasses, JSON loading, `FieldResult`), `sources`/`domain`/`forward` (FiPy
solver behind `simulate`), `observations` (`sample`, `observe`), `inverse` (`fit_sources`, `reconstruct`,
`ResponseCache`), `metrics`, `design` (`score_layout`, `select_layout`, `evaluate_layouts`), `experiments`
(benchmark / mismatch runners), `io`, `plotting`, `cli`.

Data flow: Scenario + q -> `simulate` -> `FieldResult` (xarray, no FiPy objects) -> `observe` -> `Observations`
-> `fit_sources` -> `FitResult` -> `reconstruct` -> `FieldResult` -> `metrics`.
A future solver only has to provide `simulate`'s return type (same quantities, coordinates, times).
Research Panel is used only from `scripts/`, never imported by the package.
See `implementation-plan.md` for the discretisation and conventions.
