# Agent instructions — pravah

## Start here: PRAVAH orientation (project-specific; written for agents resuming work)

PRAVAH = regional air-quality toolkit: a FiPy finite-volume advection-diffusion-removal model, nonnegative source/background inversion, sensor-placement
experiments, and a real Belauri (Nepal) geography layer. **Everything quantitative is synthetic** (emissions, winds, noise); real geography alone validates nothing.

### Read in layers (stop as soon as you have what you need)
0. This file (rules + map). 1. `.research/bin/research current` and `research resume` (~2k tokens: goal, baseline, findings + status, open questions, plans,
decisions, what needs review). 2. `docs/README.md` (index: which doc answers which question), then ONE area doc. 3. On demand: `research show <ID>`,
`research search "…"`, `research task context T-…`. 4. Evidence: `results/<run>/` (raw rows, `provenance.json`, `summary.*`), `data/belauri/manifest.json`.
Do not read everything; do not re-derive what a finding or doc already states. Findings carry a status: only **supported** ones were independently reviewed.

### Where things are
`src/pravah/`: `config` (frozen dataclasses, theta kinds), `forward` (FiPy + `ForwardSolver`/`BackendInfo`), `inverse`, `attribution`, `design`, `experiments`,
`transboundary`, `indoor`, `belauri`, `geodesign`, `geodesign_robust`, `geo/` (fetch/prepare/regionio/scenarios; optional GIS extra), `cli.py`
(`python -m pravah {simulate,inverse,benchmark,geo}`; `benchmark` dispatches on `cfg["experiment"]`). Configs in `configs/`, notebooks `01`–`06` in `notebooks/`.

### Commands
Env: `uv venv --python 3.12 .venv && uv pip install -e ".[dev,geo]" -c requirements-lock-geo.txt`. Tests `.venv/bin/pytest -q` (447 tests, ~13 min; geo/real-data
tests skip without `data/belauri/manifest.json`). Types `.venv/bin/pyright` (0 errors expected). Notebooks `.venv/bin/python scripts/verify_notebooks.py`.
FiPy regression gate: `scripts/capture_regression.py` + `scripts/compare_runs.py results/solver-regression/before <new>` must print `21 files compared, 0 differ`.
Geography: `python -m pravah geo fetch --config configs/belauri_geo.json` (network; Overpass main endpoint is flaky, a mirror fallback is built in) then `geo prepare` (offline).
Scientific runs go through `.research/bin/research run exec EXP-… -- <cmd>` (register the experiment first; synthesise before the 5-run budget).

### Rules learned the hard way
* **Do not touch the user's notebook kernels or notebook files**: the six notebooks in the working tree contain outputs the user saved from VS Code; do not clear,
  commit or revert them without asking (commit with `git add -A -- . ':!notebooks'`). Idle `ipykernel` processes whose parent is VS Code are the user's.
* **Commit early and often.** After every finished task, passing test run, or completed experiment stage, commit with a descriptive message (what changed and why, not "wip"),
  in logical commits, before starting the next piece. Run the relevant tests first. Stage explicit paths; notebooks go in their own commit and only when the user asks;
  never commit `data/*/raw`, `.venv` or `node_modules`.
* Never select a hyperparameter (alpha, layouts, noise level) using test-ensemble quantities; design / validation / test seeds and noise ranges are disjoint and guarded.
* Never loosen a check or tolerance to make something pass; supersede findings (`research finding supersede`), never rewrite history. Report numbers from files, not memory;
  an earlier error (a layout-coordinate bug in the boundary-basis study) was only caught by independent review.
* Legacy FiPy outputs must stay bit-identical (regression gate above). No `sys.path` hacks, no absolute paths, no global config.
* Terms are exact: **local** = emitted inside the study polygon; **external** = outside it; **international transboundary** only across a verified national boundary
  (the Indian outline in geoBoundaries is coarse); **indoor** = inside a building (separate one-way box). The study polygon is a mask, never a PDE wall.
* Emission amplitudes (µg/s) and inflow-basis amplitudes (µg/m³) are different parameter kinds: never compare raw magnitudes. The scale vector acts as a prior width when alpha > 0.

### Known state and open items (see docs/phase2-validation.md "Wrap-up log" for detail)
Done and reviewed: solver interface, external/local study, verified Belauri geography, Belauri forward runs, geographic design + robustness. Known limits: numerical diffusion
dominates the nominal K (K_eff/K ≈ 2.4–9.4); small-budget sensor rankings are unresolved at 6 test scenarios; identifiability flags are saturated in the geodesign runs; test noise level
was derived from the test ensemble (2.6% off the design value); no terrain-coupled flow. Unshipped fixes for these are in `git stash list` entry **G2-unshipped**.
Best next experiment: add a chi-square-per-dof model check to the robustness study and rerun it with the noise level fixed from the validation scale.

<!-- research:begin — managed by `research init`; edit outside these markers freely -->
## Research OS

This repo uses Research OS for research memory and task coordination. Every tool below is also available as an
MCP tool if the `research` MCP server is configured (`research mcp install`).

### Running the CLI (no install needed)
Use `.research/bin/research` from the project root (e.g. `.research/bin/research current`). It is a project-local
copy of the CLI, kept up to date by the VS Code extension and by `research bin install`; it needs only Python 3.
`research` works too where it is on PATH (VS Code terminals, or after `pip install`). The examples below write
`research` for short. Don't `pip install` anything just to get the CLI. If `.research/bin/research` is missing, ask
the human to open the project in VS Code with the Research Panel extension, or to run `research bin install`.

### Quick start
1. Get current state: `research current`
2. Get your task: `research task next`, then `research task context T-001` (focused brief: goal, checks, deps, skills)
3. Look things up instead of reading history: `research search "…"`, `research show <ID>`
4. Discover tools: `research --help`

### Core rules
- All runs must belong to an experiment
- Claim before working (`research task start T-001`); leave progress notes as you go (`research task note T-001 "…"`)
- Register artifacts and findings as you work
- After `max_runs_without_synthesis` runs, you must synthesize
- Findings stay preliminary until an independent review; never review your own
- Never rewrite history; supersede findings/decisions instead
- Large outputs stay where produced; just register paths
- Exit code 3 means a policy blocked you: read the hint, don't work around it

### Recording work
```bash
# Run with provenance
research run exec EXP-001 -- python train.py

# Inside code
import research
research.log_metric("rmse", 0.12)
research.register_artifact("plot.png", description="loss curve")

# Findings
research finding create "..." --supports EXP-001 RUN-001

# Checkpoint when done
research checkpoint create
```

### Task workflow (if using plans)
```bash
research task start T-001          # claim + mark running
# ... do the work, `research task note T-001 "..."` along the way ...
research task complete T-001 --result "summary of what happened"   # runs the task's checks first
```

For skills: `research skill show <name>`. For details: `research show <ID>`.
<!-- research:end -->
