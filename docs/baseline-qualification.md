# Baseline qualification — pre-migration, pre-refactor snapshot

**Scope of this document.** It holds ONLY what Research OS cannot represent: the exact environment,
configuration and dataset hashes needed to reproduce the baseline, and the commands that verify it.
Everything else — findings, their status and reviews, decisions, open questions, runs, artifacts,
blocked work, handoff state — lives in Research OS and **Research OS is authoritative**. This file is
not a second findings database; where it mentions a result it only points at the record.

Authoritative records for this baseline:

| What | Record |
|---|---|
| The baseline decision, invariants to protect, what is/isn't established | **D-007** |
| Architecture-scoped regression gate | **D-005** |
| EXP-015's dated post-hoc criterion amendment | **D-006** |
| EXP-015 outcome (H1 **untested**) and its 12 limitations | **F-035**, review **F-035/R1** (`needs_work`) |
| The only independently reviewed findings | **F-002, F-006, F-007, F-009** |
| HPC round trip verified (infrastructure, not science) | **EXP-024/S1** |
| Cluster environment + preflight | **EXP-025/S1** |
| Pre-registered fix for EXP-015, with a falsification control | **EXP-027** |
| Pre-registered WorldPop variant | **EXP-026** |
| Objective identifiability at 6 design cases | **Q-013** |
| Handoff snapshot | **CP-008** (and CP-007) |
| Complete deterministic export | `research report --full` → `.research/reports/project-full.md` |

Read `research current`, then `research show <ID>` for any of the above. Do not quote this file for a
scientific claim; quote the record.

---

## 1. Engineering baseline vs scientific validation

This snapshot is a **working engineering baseline**. It is **not** a scientifically validated model.

- **Engineering-verified:** the solver, inverse, design and HPC pipelines run reproducibly; legacy FiPy
  output is bit-locked by a regression gate; the HPC round trip is verified end to end; datasets carry
  source, licence and checksums.
- **Not validated:** every quantitative atmospheric result is **synthetic** (real Belauri geography;
  synthetic emissions, winds, noise). Only **4 of 35** findings have passed independent review. There
  is **no empirical validation** of any concentration prediction or sensor recommendation.
- **EXP-015 did not establish numerical adequacy for sensor-placement decisions.** F-035 records H1 as
  **untested** (the resampling baseline is not commensurable with the mesh comparison, so the test
  could not fail); 500 m / dt 100 is **provisionally admissible** — "no evidence against", not "shown
  adequate". Preserve this distinction in any future work.

## 2. Git revision

| | |
|---|---|
| Commit | `48e3f811a1db68d8b74043beff0cb734ae71625a` |
| Tag | `baseline/pre-refactor-2026-10` (annotated) |
| Branch | `master` |
| Tree state | clean for tracked paths (`tools/.DS_Store` untracked and ignored; `notebooks/` deliberately untouched) |

## 3. Environment

Local (`.venv`, Apple M2, arm64) — the environment that produced the committed arm64 regression baseline:

| Package | Local `.venv` | `requirements-lock-geo.txt` | Cluster (x86_64) |
|---|---|---|---|
| python | 3.12.7 | — | 3.12.12 |
| numpy | **2.5.4** | **2.5.3** | 2.5.3 |
| scipy | 1.18.1 | 1.18.1 | 1.18.1 |
| fipy | 4.0.3 | 4.0.3 | 4.0.3 |
| pandas | 3.0.6 | — | — |
| matplotlib | 3.11.2 | — | 3.11.2 |

> **Known reproducibility limitation (recorded in D-007).** The local `.venv` has drifted to
> numpy 2.5.4 while the lockfile pins 2.5.3, and `results/solver-regression/before` was captured with
> the drifted 2.5.4. The cluster is lock-correct. Both gates pass against their own baselines, so this
> does not invalidate the snapshot, but a rebuild from the lockfile on arm64 should be expected to
> reproduce `0 differ` as well (verified during EXP-025: a lock-exact arm64 venv gave `0 differ`).

Rebuild: `uv venv --python 3.12 .venv && uv pip install -e ".[dev,geo]" -c requirements-lock-geo.txt`

## 4. Frozen configuration and dataset hashes (sha256, first 16 hex)

| File | sha256 (16) | Role |
|---|---|---|
| `configs/reference.json` | `94be6b23cd5db0bf` | v0 baseline (D-002) |
| `configs/geodesign.json` | `b825fc8d21df94fa` | geographic design base |
| `configs/convergence_stage2.json` | `437959b829d986c6` | EXP-015 stage 2 |
| `configs/population_belauri.json` | `80a62fef2b0ef0e5` | EXP-026 |
| `configs/belauri_geo.json` | `47acdee626112050` | geography preparation |
| `results/geodesign/alpha_selected.json` | `721bb53a5104eddf` | **frozen alpha 0.01** (exact-equality assert) |
| `results/geodesign/noise_levels.csv` | `dff50249aa0d51ce` | **frozen noise scale** |
| `data/belauri/manifest.json` | `7ff8b8b9b1615a87` | Belauri geography, 500 m + 250 m |
| `data/population/belauri_worldpop_2025_CN.json` | `1fb184b9086c4d90` | WorldPop, CC BY 4.0 |
| `data/era5/belauri_era5_10m_wind.json` | — | ERA5 2023 10 m wind, DOI + licence recorded |

Cluster dataset: `belauri-geography` version `belauri-500m-250m-plus-raw-2026-10-10`, 39 files with
per-file sha256, at `/scratch/railabs/yb107/research-os/shared-data/belauri-geography`. Verify with
`research hpc datasets check belauri-geography --deep` (expect: *usable, 39 files*).

## 5. Reference outputs that must not be overwritten

| Path | Protect because |
|---|---|
| `results/solver-regression/before` | arm64 regression baseline (45 files) |
| `results/solver-regression-x86_64/before` | x86_64 cluster baseline (45 files), see D-005 |
| `results/reference-v0`, `results/inverse-v0` | v0 baseline (D-002) |
| `results/phase3-convergence/` | EXP-015 stage 1 + stage 2 evidence (RUN-0028, RUN-0058) |
| `results/geodesign`, `results/geodesign-robust` | Phase 2 design evidence (D-003) |
| `results/phase3-*` | Phase 3 toy evidence |

## 6. Verification commands and expected output

```bash
# regression gate — architecture-scoped (D-005); never loosen a tolerance to make this pass
.venv/bin/python scripts/compare_runs.py results/solver-regression/before results/solver-regression/after
#   -> 21 files compared, 0 differ            (arm64)
.venv/bin/python scripts/compare_runs.py results/solver-regression-x86_64/before <new-x86-capture>
#   -> 21 files compared, 0 differ            (x86_64)
# cross-architecture is EXPECTED to differ (18 files, reassociation only) — not a regression:
.venv/bin/python scripts/compare_runs.py results/solver-regression/before results/solver-regression-x86_64/before
#   -> 21 files compared, 18 differ

.venv/bin/pytest -q        # see section 7 for the measured result
.venv/bin/pyright          # -> 0 errors, 0 warnings, 0 informations (scope: src/pravah only)

.research/bin/research hpc doctor                              # -> ready: ssh-slurm, bundle
.research/bin/research hpc datasets check belauri-geography --deep
.research/bin/research report --full --check                   # deterministic export + coverage check
```

`scripts/` is outside pyright's configured `include` (`src/pravah`), so analysis scripts are not type-gated.

## 7. Measured test results at this revision

Measured on 2026-10-13 at this revision, local arm64 `.venv` (numpy 2.5.4):

| Check | Command | Result |
|---|---|---|
| Test suite | `.venv/bin/pytest -q` | **585 passed, 0 failed**, 23 warnings, 24:09 |
| Type check | `.venv/bin/pyright` | **0 errors**, 0 warnings (scope: `src/pravah`) |
| Regression gate, arm64 | `compare_runs.py results/solver-regression/{before,after}` | **21 files compared, 0 differ** |
| Regression gate, x86_64 baseline self-consistency | `compare_runs.py results/solver-regression-x86_64/before <same>` | **21 files compared, 0 differ** |
| Cross-architecture (expected to differ, D-005) | `compare_runs.py results/solver-regression/before results/solver-regression-x86_64/before` | **18 differ** — reassociation only, NOT a regression |
| Lock-exact arm64 venv (numpy 2.5.3) | rebuilt venv + `capture_regression.py` + compare | **21 files compared, 0 differ** — confirms the local 2.5.4 drift is benign |

Geo/real-data tests do not skip at this revision because `data/belauri/manifest.json` is present.
The suite includes an EXP-015 stage-2 smoke test, which is why it runs ~24 min rather than the ~13 min
recorded for the older 447-test suite.

**Reproducibility limitations.** (1) The numpy drift in section 3 — benign, but rebuild from the
lockfile for an exact match. (2) Cross-architecture bit-identity is not achievable (D-005); use the
baseline for your own architecture. (3) EXP-015 stage 2's 721 MB response-basis cache was deliberately
left on the cluster, so `stability.csv` cannot be re-derived locally from the layout JSONs — only its
internal consistency and its sha256 against the cluster copy were checked. (4) Physics and solver
internals were not re-verified during this baseline audit; the regression gate locks behaviour, it does
not prove correctness.

## 8. Invariants a future refactor must preserve

Full list and rationale in **D-007**. Summary:

1. Both architecture-scoped regression baselines and the `21 files compared, 0 differ` gate.
2. The frozen alpha exact-equality assert (`convergence_stage2.py:150-156`) against
   `results/geodesign/alpha_selected.json`.
3. The frozen noise scale (`:142-147`), passed to pool workers via `initargs` so no worker re-derives it.
4. Design / validation / test ensemble disjointness (machine-checked, `:139`).
5. **Layouts frozen before the test ensemble is built** (`:289` asserts all layout files exist before
   `:293` constructs test cases). This is the structural no-leakage guarantee.
6. The **unspent fresh test seed 304**. Seed 303 is **burned** by EXP-015 stage 2 and must not be
   reused as held-out.
7. All registered evidence under `results/` (section 5).
8. Checkpoint/resume guards hash **configuration but not code** — see the runbook's
   "Checkpoints and resume" section. Do not make resume look safer than it is.

## 9. What is pending, and what blocks it

Authoritative: `research current`. In brief — **EXP-027** (pre-registered fix for EXP-015, with a 2 km
falsification control that must fail) and **EXP-026** (WorldPop variant, gated on a defensible mesh
decision that does not yet exist). **Q-013** asks whether the design objective is identifiable at all
at 6 design cases; if it is not, small-budget conclusions in EXP-013/014/020/026 are affected too.
