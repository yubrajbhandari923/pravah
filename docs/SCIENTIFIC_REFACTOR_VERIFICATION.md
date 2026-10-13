# Scientific refactor verification (Phase 3)

Plan **PLAN-006** (T-044..T-052). Authority for scope: `docs/ARCHITECTURE_WORKFLOW_AUDIT.md`. This is an engineering record, not a scientific result.

## Revisions

| | Revision |
|---|---|
| Start (audit committed, clean tree except `tools/.DS_Store`) | `3fd5c53` (before it: `4d9d68b`; the audit's uncommitted changes were committed on their own, diff saved in the session scratchpad) |
| Batch A | `d5a38df` |
| Batch B | `47af272` |
| Batches E, F, architecture tests, docs | `e4695e1` |
| End | the commit that adds this file |

## What was done

| Batch | Result |
|---|---|
| **B0 baseline** (T-044) | Gate on unmodified code: `21 files compared, 0 differ` (2:16). 125 guard/study tests passed (8:28). |
| **A shared utilities** (T-045) | New `src/pravah/runner.py`: `say`, `save_fig`, `check_frozen`, `guard_out`, `Progress`. Replaced 4 identical `_say`, 3 identical `_save_fig`, and the identical tails of the freeze guard (3 population scripts) and run guard (2). Message text, exceptions and `run_guard.json` format unchanged. |
| **C protocol** (T-046) | **Re-scoped.** Line-by-line comparison showed the audit overstated D1/D2: the three `check_ensembles` differ in draw-key payload (`q`+full wind tuple / `theta`+list / `theta`+`astuple[:3]`), noise-seed arithmetic (`first+i` vs `first+i*noise_reps+r`), messages and signatures; `windunc` freezes into its own file with `AssertionError` and a 16-char digest. Only what is identical was shared (Batch A). Independent `scientific-reviewer` verdict: protocol preserved exactly; D1/D2 only partly sound. |
| **B study drivers** (T-047) | `experiments`, `transboundary`, `windunc` -> `pravah/studies/toy/`; `belauri`, `geodesign`, `geodesign_robust` -> `pravah/studies/belauri/`. The old modules are one-line aliases (`sys.modules[__name__] = <new module>`), so old imports return the *same module object* (tested). `cli.py` uses the new paths. No `studies/observed/` (no observations exist). |
| **D thin scripts** (T-048) | Not changed. Body comparison of same-named functions across the 28 scripts found ratios of 0.3-0.6 (same skeleton, different studies) except `prescribed_winds*` (0.65-0.76, ~25 lines) and `hpc_*` `sha256` (kept duplicated: cluster scripts stay standalone). Moving pre-registered study bodies into `src` would be code motion without a demonstrated duplicate. |
| **E data_provenance** (T-049) | `io.data_provenance(kind, datasets)` and `io.provenance(..., data=)`; optional, key omitted when not given. Wired into `cli simulate/inverse`, toy drivers (`synthetic`) and Belauri drivers (`geography_only` + manifest path and sha256). The population/EXP scripts' own provenance calls are not changed. |
| **F cleanup** (T-050) | Removed `scripts/register_research.py` (only docs and historical run snapshots mention it). `slurm/phase3.sbatch` marked DEPRECATED (one comment line). `winds.py` no longer imports `geo` (`eddy_streamfunction` moved into `winds`, re-exported from `geo.scenarios`). |

## Audit corrections found during the refactor

* `scripts/hpc_regression_selfcheck.sh` is **not** unused: RUN-0053 executed it. My grep had excluded `.research/runs`. Kept.
* `plotting.plot_layout` / `plot_error_vs_budget` are used by `tests/test_observations.py:181`. Kept.
* Audit D1/D2/D7/D8 (duplicated guards and script family) were largely name coincidences or deliberate per-study protocol; see Batch C and D above.
Both corrections are written into the audit report.

## Verification

| Check | Command | Result |
|---|---|---|
| Regression gate (before A, after A, after B, final) | `scripts/capture_regression.py` into the session scratchpad, then `scripts/compare_runs.py results/solver-regression/before <out>` | **21 files compared, 0 differ**, four times (arm64 baseline; no tolerance changed) |
| Types | `.venv/bin/pyright` | 0 errors, 0 warnings (after every batch) |
| Study/guard subset after A | `pytest -q tests/test_{experiments,geodesign,transboundary,windunc,population_*,convergence_stage2,geodesign_robust,belauri}.py` | 157 passed, 14:37 |
| Full suite after B | `.venv/bin/pytest -q` | 589 passed, 19:07 |
| **Full suite, final** | `.venv/bin/pytest -q` | **595 passed**, 23 warnings, 18:43 (585 baseline + 10 new tests) |
| Import graph | AST scan of `src/pravah` | no cycles; no module imports `research`; core does not import `geo`, `studies` or `research` (`tests/test_architecture.py`) |
| Old import paths / CLI | `from pravah import experiments, belauri, geodesign, geodesign_robust, transboundary, windunc`; `python -m pravah geo --help` | OK |
| HPC files | `git diff 3fd5c53 -- scripts/hpc_* slurm configs/hpc_*` | one comment line in `slurm/phase3.sbatch`; nothing else |

### Scientific equivalence

| Invariant or output | Before | After | Method | Result |
|---|---|---|---|---|
| FiPy deterministic outputs (simulate, inverse, benchmark, alpha-selection, mismatch smoke) | 21 files | same | exact SHA-256 compare | identical |
| Seeds, partition membership, frozen-hash payloads and values | `3fd5c53` sources | for each of the six moved drivers, `diff <(git show 3fd5c53:src/pravah/X.py) studies/.../X.py` minus import lines shows only: removed `Progress`/`_say`/`_save_fig` copies, one deferred import path, and the added `data=data_provenance(...)` argument; no RNG, config, payload or hash code differs | line diff + independent reviewer (Batch A) | unchanged |
| Freeze/run guard behaviour (messages, exceptions, file format) | in-script | `runner.*` | reviewer + `tests/test_runner.py` + the existing guard tests (about 60 references across 9 test files) | equivalent |
| Held-out test spending | none re-run | none re-run | no held-out evaluation was executed for this refactor | n/a |
| Study outputs | smoke profiles in the test-suite | same | full suite | pass (exact comparisons in the tests are unchanged) |

No tolerance was loosened and no baseline or golden file was regenerated.

## Not done / residual risk

* No formatter or linter ran: neither `ruff` nor `black` is installed in `.venv` (only pyright).
* The cluster (x86_64) gate was not re-run; nothing HPC-related changed, but code under `src/` did. Run `hpc_regression_selfcheck.sh` and the x86_64 gate before the next cluster job.
* Moved files appear to git as "modified + added" (the old path is now a shim), so `git log --follow` on the new files may need `-M`. History is intact at `4d9d68b..3fd5c53`.
* Aliases hide the new location from type checkers, so new code should import from `pravah.studies.*`; scripts, tests and notebooks still use the old names.
* `jupyter`/`ipykernel` remain hard dependencies (not changed: would alter the lockfiles); `slow` marker still unused.
* Notebooks and the four unreviewed findings (F-030..F-035) were not touched; `supported` findings F-002/6/7/9 still have no evidence bindings.
* Deliberately left as they were: `orchestration.enabled=false`; no hooks approved; `.claude/skills` and `tools/architecture-explorer` retained (Q-014 O-6/O-7); studies are inside the package (O-1 option "only into `pravah.studies`" taken).
* `scripts/population_*`, `prescribed_winds*`, `convergence_*` still contain their study logic (see Batch D).

## Outstanding

Hook proposals (audit section 5.2), `slow` markers, the `jupyter` dependency move, archiving superseded docs, `docs/hpc-runbook.md` trimming, `studies/observed/` when data exists, Q-014.
