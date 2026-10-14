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
`src/pravah/`: `config` (frozen dataclasses, theta kinds), `forward` (FiPy + `ForwardSolver`/`BackendInfo`), `inverse`, `attribution`, `design`, `winds`, `population`, `era5`, `runner` (shared study helpers),
`indoor`, `geo/` (fetch/prepare/regionio/scenarios; optional GIS extra), `studies/toy/` (`experiments`, `transboundary`, `windunc`), `studies/belauri/` (`belauri`, `geodesign`, `geodesign_robust`; the old top-level names still import, as aliases), `cli.py`
(`python -m pravah {simulate,inverse,benchmark,geo}`; `benchmark` dispatches on `cfg["experiment"]`). Configs in `configs/`, notebooks `01`–`06` in `notebooks/`.

### Commands
Env: `uv venv --python 3.12 .venv && uv pip install -e ".[dev,geo]" -c requirements-lock-geo.txt`. Tests `.venv/bin/pytest -q` (595 tests, ~22 min; geo/real-data
tests skip without `data/belauri/manifest.json`). Types `.venv/bin/pyright` (0 errors expected). Notebooks `.venv/bin/python scripts/verify_notebooks.py`.
FiPy regression gate: `scripts/capture_regression.py` + `scripts/compare_runs.py <baseline> <new>` must print `21 files compared, 0 differ`.
The gate is **architecture-scoped** (D-005): baseline `results/solver-regression/before` on arm64 (laptop), `results/solver-regression-x86_64/before` on the x86_64 cluster. Comparing across architectures gives 18 spurious differences (reassociation only, <=5e-08 relative in any physical quantity); never loosen a tolerance for it.
Geography: `python -m pravah geo fetch --config configs/belauri_geo.json` (network; Overpass main endpoint is flaky, a mirror fallback is built in) then `geo prepare` (offline).
Scientific runs go through `.research/bin/research run exec EXP-… -- <cmd>` (register the experiment first; synthesise before the 5-run budget).
Research OS is **1.0.0** (`research --version`; `research config check`, `research upgrade --dry-run` are clean). Project use of its features:
* **Evidence binding**: findings F-002/F-006/F-007/F-009 are `supported` but have `no_bindings` (`research evidence status` lists their legacy numbers). New findings: bind every number
  (`research evidence bind F-… --run RUN-… --metric NAME`, then `evidence verify`) before asking for `supported`. Do not bind retroactively without checking the run record.
* **Tuning** (alpha, layouts, noise): `research tuning declare|trial|select|test` (the held-out test is one-shot; test seed 304 is the only unspent one). Replaces the old `orchestrate tuning` form.
* **Hooks / strategy**: none active (`research hook list`, `research strategy show`). If wanted, *propose* (never enable or approve yourself) a `post_implementation` hook running the regression gate
  and a pyright check; the human approves via `research hook approve`. Delegated tasks: `research task context T-…`, then `research orchestrate verify T-… --tier LEVEL --record`.
* **MCP** covers the core loop only; use the CLI for HPC, orchestration, hooks, tuning.

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
was derived from the test ensemble (2.6% off the design value); no terrain-coupled flow. The G2 fixes (validation-scale test noise, chi2/dof, candidate flags) were applied in commit 0370f3e, but the geodesign results and their doc numbers have NOT been rerun with them.
HPC runs go through Research OS (`research hpc doctor|submit|status|sync`), NOT `sbatch slurm/phase3.sbatch` (superseded); the venv lives outside the bundle, artifacts must be written under `$RESEARCH_RUN_DIR`, datasets are staged on the cluster and linked by `scripts/hpc_link_datasets.sh`, and every reservation needs the user's explicit approval. Phase 3 state and what to run on HPC: `docs/hpc-runbook.md`; also `docs/transfer.md`, `docs/numerical-methodology.md`, `docs/winds.md`, `docs/population-data.md`.
**BASELINE FROZEN (2026-10-12).** `baseline/pre-refactor-2026-10` @ 4320f8b is the pre-migration, pre-refactor snapshot: see **D-007** for what is
established vs uncertain and the invariants a refactor must preserve, and `docs/baseline-qualification.md` for environment, config and dataset hashes plus
verification commands. It is a **working engineering baseline, NOT a scientifically validated model**: only F-002/F-006/F-007/F-009 are independently
reviewed, everything quantitative is synthetic, and there is no empirical validation.
**EXP-015 did NOT establish numerical adequacy for sensor-placement decisions**: F-035 records H1 as UNTESTED (the resampling baseline is not commensurable
with the mesh comparison, so the test could not fail; review F-035/R1 = needs_work). 500 m / dt 100 is PROVISIONALLY admissible only. D-006 records the dated
post-hoc criterion amendment. EXP-027 is the pre-registered fix (with a 2 km falsification control); EXP-026 is gated on a defensible mesh decision.
Open and consequential: **Q-013** — the design objective may not be identifiable at 6 design cases (eps_sd/J_ref 1.01-2.52), which would affect the
small-budget conclusions of EXP-013/014/020/026 too. Test seed 303 is burned; 304 is unspent.
Best next experiment: EXP-027 (resolve the mesh question properly) or Q-013 (resolve identifiability first — arguably the more fundamental of the two).


**Dynamic population-weighted program (PLAN-007, 2026-10-13/14).** Full account: `docs/DYNAMIC_POPULATION_SENSOR_RESEARCH_REPORT.md` (PROVISIONAL). Code: `pravah.dynamic` (bin expansion, truth scenarios, BVLS/ADMM solvers; the forward solver is untouched), `pravah.dynamic_design`, `pravah.studies.toy.dynamic_{study,history,design_study,robustness,sufficiency,separability,separability_threshold}`. Current findings: F-044, F-046, F-048..F-051, F-053, F-054 (all preliminary; earlier F-036..F-047, F-052 are superseded, do not cite them). Hard-won rules: ADMM must converge (it raises `NotConverged`; the old cap made results meaningless); prior scale uses fixed `REF` amplitudes, never layout-dependent norms; check (parameter, bin) sensitivity before scoring; regret needs an independent evaluation reference; Research OS collects only logs from cluster jobs (print results, see `scripts/hpc_cat.sh`); the cluster venv holds a stale `pravah`, launchers prepend the bundle `src/`. Default `pytest` now skips 15 tests >= 25 s (`pytest -m ""` runs everything; list in `tests/conftest.py`). Open: DYN-08/EXP-026 (not run), EXP-027b (needs approval), review of the final findings.

<!-- research:begin — managed by `research init`; edit outside these markers freely -->
## Research OS

This repo uses Research OS for research memory and task coordination. The core loop (current state, tasks,
experiments, runs, findings, review, search) is also exposed as MCP tools if the `research` MCP server is configured
(`research mcp install`). MCP does **not** cover orchestration (`orchestrate freeze|contract|attempt|delegate|verify`),
hooks, skills editing, strategy/configuration, tuning, or most `hpc` commands: use the CLI for those.
`research task context T-…` prints the effective task specification (objective, constraints, permitted files,
acceptance criteria, checks, hooks needing human approval, conflicts) and is the same on CLI, MCP and the editor.

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

### Scientific practice
- Read context in layers (`research current` → `task context` → `show <ID>`) and stop when you have enough. Search prior
  findings and docs before repeating work.
- Keep intent, implementation, observation, interpretation and supported conclusions distinct. Label findings
  preliminary, supported, contradicted or superseded.
- Approved experiment specs and evaluation criteria are binding. Never tune on held-out test results; never loosen a
  check or tolerance just to pass. Quote numbers from run/metric records, not memory.
- Bind every number in a finding to its record (`research evidence bind F-… --run RUN-… --metric NAME`, then `research evidence verify F-…`).
  A mismatch blocks `supported`; correct it by superseding the binding, never by explanation. Verified numbers are not a reviewed interpretation.
- Preserve the user's files, data, notebooks and unsaved work. Project-specific numerical/scientific rules belong in the
  user-authored part of this file, outside the markers.
- Unfamiliar with the science? Read `docs/SCIENTIST_GUIDE.md` (`research skill show scientist-guide`). After synthesizing
  an experiment, refresh it or leave it marked pending: `research guide status`. Hand-offs: `research skill show scientific-handoff`.

### Remote execution (optional; only when `hpc.clusters` is configured)
- `research hpc doctor` first: it says whether SSH/Slurm works and which modes are usable, and every failing
  check prints its own fix. Never `ssh` by hand — the `research hpc` commands enforce the approved paths,
  resource limits and provenance.
- Setting it up, or doctor reporting a problem? Read `research skill show hpc-setup` before changing config.
  Common trap: `sbatch=NO` while an interactive `ssh` finds it — a non-interactive shell skips the profile
  that loads Slurm. Ask the user for approved paths; never invent a storage location.
- **Always confirm the resource reservation with the user before submitting.** Run
  `research hpc submit EXP-001 --cluster NAME --dry-run`, show them the CPUs, GPUs, memory and wall-time it
  would reserve (and the CPU-/GPU-hours that costs at the limit), and get an explicit yes. Slurm bills the
  reservation, not the usage. Only then re-run with `--yes`. Never raise cpus/gpus/mem/time on your own
  initiative, and never submit twice.
- If a submission is reported uncertain, run `research hpc sync` — it adopts a job that is already queued. Do not resubmit.
- No direct execution available, or a human will run it? `research hpc bundle EXP-001` → they run it → `research hpc import <archive>`.
- `research hpc sync` before proposing more runs: it reconciles finished jobs, imports results and lists what now
  needs interpretation. Register the complete outcome (metrics, artifacts, failures) before drawing conclusions.
- Datasets live on the cluster; the filesystem is the source of truth (`research hpc datasets list|check`). Never
  re-download something that verifies, and never assume an undescribed directory is valid.
- Slurm keeps running while the laptop is off; agents do not. Nothing is interpreted until someone runs sync.

### Context hand-off and upgrades
- A complete hand-off (new session, another model, a human): `research report --full --check`. It is a deterministic
  dump of the records — **not** a synthesis. It interprets nothing; don't quote it as if it concluded something.
- Before migrating a project: `research upgrade --dry-run`, and `research agents-md --diff` to review the managed
  block. Everything outside the two marker comments below is the project's — never rewrite it.

### Tiered agents (optional)
If `orchestration.enabled`: `research strategy show` says which model level does what (or the legacy Opus/Sonnet/Haiku tiers when no
strategy is active). `research orchestrate route T-001`; protocol: `research skill show engineering-loop`. Never edit frozen tests.
A delegated task has a contract (`research task context T-001`); check your work with `research orchestrate verify T-001 --tier LEVEL --record`.
Project hooks (`research hook list`) are run by Research OS only after a human approves them; propose one with `research hook add`.

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
