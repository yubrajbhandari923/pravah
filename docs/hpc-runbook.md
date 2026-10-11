# HPC runbook: heavy Phase 3 experiments

Audience: an agent (or person) who moves this repository to an HPC cluster and runs the experiments that were too heavy for the laptop. Scope: set-up,
what to run in what order, what is NOT built yet, how to register results. Everything is synthetic (real Belauri geography; synthetic emissions,
winds and noise; the population fixtures are synthetic). Nothing here is empirical validation. Read `../AGENTS.md` and `.research/bin/research current`
first; this document does not repeat them. Numbers below come from the files named next to them; "derived" means arithmetic on a measured number, not a
measurement. Commands were checked against `--help` or the argparse code; none was executed for this document.

## How to run on the cluster (one job)

The code is CPU-only (FiPy/NumPy); GPUs are not used, do not request them. The whole heavy Phase 3 set runs as ONE Slurm job (`squeue` shows one
job): `scripts/hpc_run.py` runs the phases of `configs/hpc_phase3.json` strictly in sequence (preflight, stage-2 `--estimate`, stage-2 production,
Belauri population design if `scripts/population_belauri.py` exists, else that phase is skipped with a message); the parallelism is a Python process pool
inside each phase (BLAS threads = 1). Nothing here was run under Slurm; the Python side was tested locally (`tests/test_hpc_run.py`, smoke config).
```
# once: copy the repo (git clone or rsync) + data/belauri/raw (section 1.3), then
bash slurm/setup_env.sh                      # venv, locked deps, 125 m level (safe to repeat; never overwrites data/belauri/manifest.json)
edit slurm/phase3.sbatch                     # job name, partition, account, mem, module line; --time 24:00:00; --cpus-per-task 64..128
sbatch slurm/phase3.sbatch                   # CPUS up to 128 on one node: sbatch --cpus-per-task=128 slurm/phase3.sbatch
python scripts/hpc_run.py --config configs/hpc_phase3.json --dry-run     # prints the plan, runs nothing
```
Resume after a time limit or failure: resubmit the same job; a finished phase (done marker present and phase-config hash unchanged) is skipped, and stage 2
resumes from `results/phase3-convergence/stage2_work`. `--only NAME` runs one phase; `--max-workers N` caps the pool. Logs: `results/hpc_logs/<phase>.log`.
Stage 2 option `stage2.replicates_on_reference: true` (default false; pre-register it as an EXP-015 variant first, section 5) also selects the replicate layouts on
the reference mesh and writes `replicate_baseline_refmesh.csv`; it is not set in `configs/convergence_stage2.json` (it would change the config hash).

## How results come back

No Research OS runs on the cluster. At the end the job writes `results/hpc_manifest.json` (per phase: argv, start/end, wall seconds, exit code, host, Slurm job id,
CPU count, package versions, git commit + dirty flag + `git diff` sha256, config hashes, every output file with sha256 and size, work-dir sizes, peak child RSS)
and `results/phase3-hpc-bundle.tar.gz` (+ `.sha256`; outputs and logs, without caches and `.npz` over 50 MB). Copy both to the laptop repo:
```
scp CLUSTER:<repo>/results/phase3-hpc-bundle.tar.gz CLUSTER:<repo>/results/hpc_manifest.json results/
sha256sum -c results/phase3-hpc-bundle.tar.gz.sha256   # after copying the .sha256 too;  tar -xzf results/phase3-hpc-bundle.tar.gz
```
The laptop agent then registers the runs from the manifest with `research run attach EXP-015 ...` (one per phase; use the manifest's argv, times, exit code, commit
and file hashes as the provenance). The manifest, not the cluster, is the record: the commit may be dirty on the cluster (check `git.dirty`/`diff_sha256`).

## 0. Rules that must not be broken

* **Freeze before test.** Hyperparameters (alpha, budgets, layouts, tolerances tau, beta, c, prior ranges, reference mesh) are fixed from design or
  validation ensembles only, written down (config hash), and only then are test cases built or opened. Never reselect after seeing test results.
* **Disjoint seeds.** Design, validation and test scenario seeds, and their noise-seed ranges, must be pairwise disjoint. The guards are
  `geodesign.check_ensembles` (Belauri) and `population_design.check_ensemble_seeds` / `windunc.assert_disjoint_cases` (toys). Seeds already used:
  101/202/303 (Phase 2 design/validation/test; noise 1000/2000/3000), 304 (held for EXP-015), 305 (EXP-016, noise 3500), 1101-1108 (EXP-015 replicates;
  noise 4000 + 100 r), 4101/4202/4305 (EXP-017 toy), 5101/5202/5303 (EXP-019 toy production). Pilot and smoke profiles have their own seeds.
* **Regression gate.** The FiPy outputs of the legacy workflows must stay bit-identical: capture and compare, and the compare must print
  `21 files compared, 0 differ` (section 1.5). Never loosen a tolerance or a check to make anything pass. If the gate shows differences on a new
  machine (different CPU, BLAS, SciPy build), record the environment and report it as a finding; do not edit the comparison.
* **Findings** start as `preliminary`; only an independent review makes them `supported`. Never review your own finding. Supersede, never rewrite history.
* **Notebooks are the user's.** Do not touch, clear, commit or run `notebooks/`; never kill VS Code kernels. Commit with
  `git add -A -- . ':!notebooks'` or explicit paths. Never commit `data/*/raw`, `.venv`, `node_modules`.
* Commit after every finished stage, with a message saying what and why. Terms are exact: local = emitted inside the study polygon; external =
  outside it; the polygon is a mask, never a PDE wall.
* Do not write into `results/` directories that hold registered evidence: use a new `--out` (several scripts have `--overwrite`; do not pass it there).

## 1. Environment

### 1.1 Python environment
```
uv venv --python 3.12 .venv
uv pip install -e ".[dev,geo]" -c requirements-lock-geo.txt
# equivalent without uv:  python3.12 -m venv .venv && .venv/bin/pip install -e ".[dev,geo]" -c requirements-lock-geo.txt
.venv/bin/pyright          # 0 errors expected
.venv/bin/pytest -q        # 447 tests, about 13 min on the laptop; geo/real-data tests skip without data/belauri/manifest.json
```
The lock file `requirements-lock-geo.txt` pins the versions the results were produced with; do not upgrade. The laptop used 8 GB RAM. FiPy uses its default
(SciPy) LU solver (`SolverOptions`, tolerance 1e-12); there is no MPI or PETSc path, so each solve is single-core. Set `OMP_NUM_THREADS=1`
`OPENBLAS_NUM_THREADS=1` per process when running many processes per node (a precaution, not measured here).

### 1.2 Research OS on the cluster
`.research/bin/` is git-ignored, so a fresh clone has no CLI. Ask the human to run `research bin install` (or copy `.research/bin/research` from the
laptop); the CLI needs only Python 3. `.research/` itself (records, runs) is tracked. Run ids (RUN-xxxx) are allocated per checkout: a cluster checkout and
the laptop checkout will both allocate the next number. Work on a branch, hand the branch back, and let the human merge; after a merge run
`research sync` (and `research rebuild` if the database looks stale). I could not verify how Research OS resolves duplicate ids; avoid the situation.

### 1.3 Data to copy
| Item | Tracked in git? | Needed for | Action |
|---|---|---|---|
| `data/belauri/manifest.json`, `data/belauri/processed/*.npy` (500 m and 250 m levels, 2.0 MB) | **yes** (comes with the clone) | everything Belauri | nothing |
| `data/belauri/raw/` (96 MB: boundaries, OSM extracts, DEM window, fetch records, osmnx cache) | **no** (git-ignored) | only to re-prepare a geography level (125 m) | `rsync -a laptop:.../data/belauri/raw/ data/belauri/raw/`; do not re-fetch (Overpass is flaky and the OSM state would differ) |
| `results/phase3-bench/geo125/` (8.5 MB, a 125 m level) | no (git-ignored) | stage 1 reruns, 125 m benchmarks, any 125 m work | copy, or regenerate (1.4) |
| `results/geodesign/alpha_selected.json`, `noise_levels.csv` | yes | frozen alpha and noise scale for stage 2 | nothing |
| `results/phase3-convergence/stage2_work/` | no | resuming a started stage 2 | copy only if continuing a laptop run |

No network is needed for any run below. `geo fetch` is the only network command and must not be rerun.

### 1.4 The 125 m geography level (re-prepare offline)
The repository only carries 500 m and 250 m (`data/belauri/processed`). The 125 m level used by `scripts/bench_resolution.py` and stage 1 is a throw-away
re-preparation of the SAME raw data with one extra resolution; see `results/phase3-bench/README.md`. The only difference between the geography config
and the one that produced `geo125/` is `grid.resolutions_m: [500, 250, 125]` (verified by diffing `results/phase3-bench/geo125/belauri_geo_125.json`
against `configs/belauri_geo.json`). To rebuild:
```
.venv/bin/python - <<'EOF'
import json
c = json.load(open("configs/belauri_geo.json")); c["grid"]["resolutions_m"] = [500, 250, 125]
json.dump(c, open("results/phase3-bench/geo125/belauri_geo_125.json", "w"), indent=1)   # mkdir -p results/phase3-bench/geo125 first
EOF
.venv/bin/python -m pravah geo prepare --config results/phase3-bench/geo125/belauri_geo_125.json \
    --cache-dir data/belauri/raw --out results/phase3-bench/geo125/processed
```
**Pass `--out` explicitly.** The default `--out data/belauri/processed` would overwrite the tracked `data/belauri/manifest.json` (the manifest goes in the
parent of `--out`). After preparing, check that the 500 m and 250 m arrays equal the tracked ones
(`np.array_equal` on `study_fraction_500m.npy`, `..._250m.npy`, `w_road_*`); if they differ the raw data is not the same and every 125 m number is
suspect. Smoke test: `.venv/bin/python scripts/bench_resolution.py --cases 125:100` (skips rows already in the CSV; writes to
`results/phase3-bench/resolution_cost.csv`; use `--out` to keep the committed CSV untouched, and expect 71 s, 0.38 GB).

### 1.5 Regression gate (run once after setting up, and after any code change)
```
.venv/bin/python scripts/capture_regression.py --out results/solver-regression/hpc-check
.venv/bin/python scripts/compare_runs.py results/solver-regression/before results/solver-regression/hpc-check
# required last line: 21 files compared, 0 differ
```
(commands from the scripts' `--help`; `before` is the committed capture.) Do not commit `hpc-check`.

## 2. Cost model (sources)

`results/phase3-bench/resolution_cost.csv` (script `scripts/bench_resolution.py`; one production-like Belauri case, 8 GB laptop, fresh subprocess per row):

| mode | resolution | dt | cells | wall | peak RSS |
|---|---|---|---|---|---|
| forward (1 solve) | 500 m | 100 s | 4290 | 4.84 s | 0.17 GB |
| forward | 500 m | 50 s | 4290 | 8.75 s | 0.17 GB |
| forward | 250 m | 100 s | 17160 | 15.1 s | 0.21 GB |
| forward | 125 m | 100 s | 68640 | 71.08 s | 0.38 GB |
| basis (5 solves = one cache unit) | 500 m | 100 s | 4290 | 22.3 s | 0.17 GB |
| basis | 250 m | 100 s | 17160 | 76.1 s | 0.21 GB |
| basis | 125 m | 100 s | 68640 | 344 s | 0.43 GB |

Scaling used below (derived): cost grows about 4x per halving of dx at fixed dt (observed 3.1x then 4.7x for forward solves) and about 1/dt at fixed cells
(500 m dt 100 -> 50 costs 1.8x). So a basis at 250 m dt 50 is about 140 s, at 125 m dt 50 about 600 s (derived, not measured). One Belauri case = one
basis (distinct scenario = distinct cache key). A cluster core may be faster or slower than the laptop: measure with `bench_resolution.py --out` first.
The 125 m exponential scheme at dt 100 has Courant up to 2 at 2.5 m/s: stable (implicit) but its accuracy at that dt is untested (section 8).

## 3. The ResponseCache and parallelism

`inverse.ResponseCache(directory)` stores one `<sha256>.npz` per response basis. The key hashes scenario (grid, dt, wind, sources, physics), solver options,
schema version, package version, `q_ref` and the forward backend info. Consequences:

* **Keys never collide across mesh, dt, wind or source set**, so processes working on DIFFERENT configs/cases write different files. This is the safe
  case: one process per config, each with its own `--work`/`--out`, or sharing a directory only when their keys differ.
* **Concurrent writers of the SAME key are now safe**: `ResponseCache.get` writes a temp file in the same directory and `os.replace`s it (keys unchanged), so a
  reader never sees a half-written `.npz`. (Older caches written before this change may still hold a truncated file from a crashed run.) A crashed writer can leave a truncated `.npz`: if a resume fails with a zip/EOF error, delete that one file (its name is the key
  in the traceback) and rerun.
* `population_design.py` writes unit results through `.tmp` + `replace` (safe); `convergence_stage2.py` writes `layouts_*.json` and `rep*.json` with plain
  `write_text` (an interrupted write leaves a corrupt JSON that the resume then trusts: if json decoding fails, delete that file).
* The cache is on the node's filesystem: put `--work` on node-local scratch for speed, copy back the small JSON results, and keep the big cache out of git.
* `convection="vanleer"` is nonlinear: the cache refuses it (`ValueError`), so it can never appear in cached-response work (section 8).

## 4. Order of work

1. Section 1 set-up and the regression gate.
2. (a) EXP-015 stage 2. Its verdict selects the mesh for (b).
3. (b) EXP-016 Belauri population design (needs new code).
4. (c) EXP-019 follow-ups at Belauri scale (needs new code; optional).
5. (d) conditional S5a (T-031), only if EXP-019 shows a design-relevant wind effect.
After any batch: tests, commit, `research run exec` record, synthesis (section 9).

## 5. (a) EXP-015 stage 2: decision stability across mesh and dt

**Status.** Script `scripts/convergence_stage2.py` (commit 2eb96e2, tests 38872b0 `tests/test_convergence_stage2.py`) and `configs/convergence_stage2.json` exist
and pass their smoke test; the production run has NOT been run. Stage 1 (RUN-0028, F-022) is done.

**Purpose.** Do greedy_total sensor layouts selected on 500 m/dt 100 (and 500 m dt 50, 250 m dt 100) differ from those on the reference 250 m/dt 50
more than layouts differ when the design scenarios are merely resampled?

**Pre-registered (EXP-015).** H1: mesh-induced displacement and regret are no larger than the resampling baseline. H0: they are larger (numerically unstable
at 500 m). H2: a flux-limited scheme reduces field error and K_eff/K but not necessarily regret (vanleer is NOT a design arm, see section 8). Frozen:
alpha 0.01 (`results/geodesign/alpha_selected.json`), noise sigma = fraction (0.05, 0.15) x the frozen validation signal scale (0.25096 ug/m3,
`noise_levels.csv`), budgets 3, 5, 8, design seed 101 (6 cases), validation 202, test 303 (6 cases; 304 held for a final fresh test), candidate sites fixed in UTM
metres once. Replicate baseline: 8 design ensembles of 6 cases (seeds 1101-1108, noise seeds 4000 + 100 (r-1)) on 500 m/dt 100. Success (no universal
threshold): per budget, mesh-induced displacement and regret vs the replicate distribution (median, IQR); stability claimed only if mesh-induced values lie within
the replicate IQR AND decrease monotonically 500 -> 250 m; report tau (500 m, 1 km) match fractions and the epsilon-tie fraction. Stop: abort a config if one
solve exceeds 3x its benchmark; drop the 125 m design stage if the 250 m stage exceeds 4 h; do not change alpha, budgets, tau or reference after seeing test results.
(Metric definitions: `numerical-methodology.md` section 5.)

**Commands.**
```
.venv/bin/python scripts/convergence_stage2.py --help
.venv/bin/python scripts/convergence_stage2.py --config configs/convergence_stage2.json --estimate        # times one reference-mesh design case, then exits
.venv/bin/python scripts/convergence_stage2.py --config configs/convergence_stage2_smoke.json --out /scratch/s2smoke   # plumbing check, cropped domain
.research/bin/research run exec EXP-015 -- .venv/bin/python scripts/convergence_stage2.py --config configs/convergence_stage2.json
# options: --out DIR (default results/phase3-convergence)  --work DIR (default OUT/stage2_work)
```
Use `--estimate` first; compare with the benchmark scaling (derived: 250 m dt 50 basis about 140 s; design set of 6 about 14 min).

**Order inside the script** (resumable, one process): candidate table once; `layouts_<mesh>.json` per mesh, reference first; 8 replicate files `rep<r>.json`;
only then the test ensemble on the reference mesh (the script asserts all layout files exist, so layouts are frozen before the test ensemble is built);
tables, figures, `README_stage2.md`. Meshes: (500,100), (500,50), (250,100), (250,50); reference (250,50). The 125 m level is NOT in this config.

**Time and memory.** The laptop estimate is about 2 h (measured earlier by the owner; pre-registered upper bound 4-5 h). Derived from the benchmark: design sets
about 14 + 8 + 4 + 2 min (250/50, 250/100, 500/50, 500/100), replicates 8 x 6 x 22 s about 18 min, test about 14 min, plus greedy/evaluation overhead.
The script asserts peak RSS < 2 GB (`RSS_LIMIT_GB`); measured per-basis RSS is 0.2 GB. Request 4 GB and a 6 h limit; the cluster core may be faster.

**Resume / checkpoints.** Re-running the same command skips existing `layouts_*.json`, `rep*.json` and cached bases. **The script hashes the config (`config_hash.txt` in the work dir) and refuses a changed config; note it does not hash the code:**
if you change the config (or the code) after a work dir exists, delete `stage2_work` or use a new `--work`, otherwise stale layouts are silently reused.
A killed run during the test ensemble loses nothing but time (bases are cached).

**Parallelism.** `--workers N` (default 1 = sequential, identical outputs) runs the 4 mesh layout sets (reference submitted first, alongside the others: no
other mesh needs the reference design cases) and then the 8 replicate ensembles (they score the reference layout, so they start when it exists) in a spawn
process pool; the test ensemble and tables stay in the main process afterwards. Critical path = reference design set, then the test ensemble.

**Outputs** (in `--out`, default `results/phase3-convergence/`, committed unless large): `stability.csv`, `replicate_baseline.csv`, `layouts.csv`,
`layouts.geojson`, `test_scores.csv`, `epsilon_swaps.csv`, `candidate_mapping.csv`, `README_stage2.md` (numbers only), `figures/layout_maps.png`,
`figures/displacement_regret_vs_baseline_f0.05.png` and `_f0.15.png`, `figures/objective_ties.png`. `stage2_work/` is git-ignored. Register the directory and the
README as artifacts (section 9). The pre-registered artifact names `stage1_fields.csv` were renamed in stage 1 (`stage1_summary.csv`); `stability.csv` and
`replicate_baseline.csv` keep their names.

**Optional HPC extensions (1 implemented as `stage2.replicates_on_reference`, files `refrep<r>.json`; 2 not implemented).**
1. *Reference-mesh replicate baseline* (about 3.5 h on the laptop, planning estimate; derived check: 8 replicates x 6 cases x 140 s = 1.9 h at 250 m/dt 50). The
   current baseline resamples design scenarios at 500 m only, so it answers "is 500 m movement larger than resampling at 500 m", not "at the reference
   mesh". Implementation: a second replicate loop with `candidate_mesh` = the reference mesh and a separate `rep250_<r>.json` family; compare mesh-induced
   displacement against both baselines. Pre-register it as a variant of EXP-015 (`research experiment variant`) before running.
2. *125 m design stage.* Add (125, 100) or (125, 50) to `stage2.meshes` and point `region.manifest` at `results/phase3-bench/geo125/manifest.json` (1.4;
   `mapping_table` needs the 125 m level in the manifest; `load_config` merges `extends`). Cost (derived): 125 m basis 344 s at dt 100, about 600 s at dt 50;
   6 design cases about 35 min (dt 100) to 1 h (dt 50); the test ensemble stays on the reference mesh unless the reference changes (changing the reference after
   seeing results is forbidden; if 125 m is to be the reference, pre-register that first). The stop rule drops it if the 250 m stage exceeded 4 h. Check Courant
   (section 8) before choosing dt.

**Decision hand-off.** After the run, write the verdict as a finding and a synthesis (section 9). EXP-016 (b) uses 500 m/dt 100 if decisions are stable relative
to resampling, else the finer mesh named in the verdict.

## 6. (b) EXP-016: Belauri-scale population-aware design — IMPLEMENTED (code ready, production not run)

**Update (2026-10-11).** The driver now exists: `scripts/population_belauri.py` (commit 61f8ae4), `configs/population_belauri.json` (+ `_smoke`), `tests/test_population_belauri.py`.
It is already phase `population_belauri` of `configs/hpc_phase3.json`, so the one-job run (top of this file) executes it. It uses exact enumeration at budgets 2-4, swap search at 5 and 8,
scores greedy layouts on test, includes the uniform fixture and a sixth real fixture (WorldPop, `data/population/`), a two-way cluster bootstrap and a design-split control. Pilot estimate:
about 1.2 h on 1 worker, about 9 min on 64 (the floor is three sequential case rounds). Mesh 500 m / dt 100 is a config default until EXP-015 stage 2 gives its verdict. The paragraphs below are the original
specification and gap analysis, kept for context where they do not contradict this update (the run command is
`.research/bin/research run exec EXP-016 -- .venv/bin/python scripts/population_belauri.py --profile production`, and the 'build it' text no longer applies).

**Original status.** Pre-registered, `proposed`, 0 runs. The toy version EXP-017 is finished (F-023, preliminary: benefit only at 3 sensors on one concentrated town, about 3x area
error, vanishes by 5 sensors; layouts matter more than objectives). Library support exists: `src/pravah/population.py` (fixtures, `mixed_weight`, `pop_weighted_rmse`,
regridding), `design.DesignOptions.weight`, and the toy driver `scripts/population_design.py` (objective weights, low-population mask, per-cell error forms
`CellForm`, freeze guard, resumable unit files, bootstrap, uniform==area sanity). What is missing is a Belauri driver. Build `scripts/population_design_belauri.py`
(suggested name) + `configs/population_belauri.json` (+ `_smoke.json`) + `tests/test_population_belauri.py`.

**Pre-registered hypotheses (EXP-016).** H1: on concentrated fixtures (one_town, separated_settlements, dense_core_sparse_surround, mixed_rural_urban) population-weighted
greedy layouts lower test population-weighted RMSE vs the area layout at equal budget; for the uniform fixture layouts must be identical (sanity). H2: area RMSE rises, more in
low-population cells. H3: local/external attribution error is not degraded beyond between-case variability. H4: the benefit shrinks or reverses when the test population
differs from the design one. Stop: uniform-fixture layouts differ from the area layouts (bug); population weight zero over the whole candidate neighbourhood; no
reselection of beta, c, alpha or fixtures after test results. Null results are valid.

**What to build, precisely.**
1. *Mesh.* Read the EXP-015 stage 2 verdict (section 5): 500 m/dt 100 if stable relative to resampling, otherwise the finer configuration it names. Write the chosen mesh in the
   config and in the finding/decision before any EXP-016 run. If the mesh is not 500/100, bases cost more (section 2).
2. *Setup.* Reuse `geodesign.Setup` as `convergence_stage2.make_setup` does (deep-copy the cfg, set `mesh`, share one `ResponseCache`). Config `extends: geodesign.json`;
   ensembles design `{n_cases 6, seed 101, noise_seed 1000}`, validation `{6, 202, 2000}`, test `{12, 305, 3500}` (EXP-016: 12 fresh scenarios, test noise seed 3500; never
   used for any choice). Validate with `geodesign.check_ensembles` over design/validation/test (and against 303/304/1101-1108 so no draw is shared with EXP-015).
   Noise: sigma = fraction (0.05, 0.15) x the FROZEN validation scale from `results/geodesign/noise_levels.csv` (as stage 2 does via `frozen_noise_scale`); alpha 0.01 frozen from
   `results/geodesign/alpha_selected.json` for ALL objectives (check it per noise fraction like `check_alpha`). A validation-selected-alpha rerun is allowed only as a
   clearly labelled SECONDARY analysis.
3. *Population on the Belauri grid.* Grid = `geo.region.level(res).grid` (UTM 32644). `population.FIXTURES[name](grid, total, seed)` places centres by fractions of the whole
   padded grid (it includes the 8 km buffer), so the raw fixtures put mass outside the study polygon. Required: counts **inside the study polygon only**, i.e. mask with
   `study_fraction` (arrays["study_fraction"], values in [0,1]) and restore a declared total over the study (a small new function `population.mask_to_study(counts, study_fraction, total)`
   with tests: total conserved, zero outside, nonnegative; fractional people per cell are acceptable if documented). Declare `total`, the fixture seeds (0..4 as in the toy
   config) and the study-population total in the config; use all five fixtures. Real WorldPop/GHS-POP rasters are NOT part of EXP-016 (see `population-data.md`).
4. *Weights.* Objectives all weighted MSE through `DesignOptions.weight`: area `w = A_i * study_i / sum`; population `w = N_i / sum N` (N already masked); mixed
   `(1-beta) w_area + beta w_pop`, beta in {0.25, 0.5, 0.75}; capped `min(N_i, c * mean N over study cells) / sum`, c = 3 (an equity proxy, not personal exposure). Do NOT multiply by
   `study_fraction` twice. Check that the area objective equals the existing `Setup.weight` (study_fraction) up to normalisation, so it reproduces the Phase 2 greedy_total layouts
   (a regression check against `results/geodesign/layouts.csv`; the toy driver's `objective_weights` assumes uniform area and no mask, so it must be generalised, not copied).
   `DesignOptions.weight` is baked into the case at build time (design.py docstring), so each objective needs its own cases (or `design.py` weight handling reused as in the toy
   via per-cell quadratic forms); check the cost: bases are cached, case construction from cached bases is cheap.
5. *Methods and budgets.* Objectives {area, mixed 0.25/0.5/0.75, population, capped}, plus baselines random (n = 20, `layout_seed` 7) and spread, budgets 2, 3, 5, 8; greedy on the
   DESIGN ensemble. Interior-only setting as in Phase 2 (`SETTING_A`, standard regime).
6. *Evaluation (test only).* Per layout x test case x noise rep: pop-weighted RMSE, area RMSE, RMSE in the lowest-population quartile of study cells (`lowpop_mask` with q = 0.25,
   support_min >= 1 person), worst-cell RMSE, held-out-sensor RMSE (block and random, from the Phase 2 held-out set), local/external attribution error (`D.fit_layout` rows already
   carry attribution in Phase 2). Per-cell time-averaged squared errors via `CellForm` (move it from the script into `src/pravah/` with tests, or import the script as the tests
   do) give any weighted RMSE without new solves. Cross-fixture matrix: layouts designed on fixture X evaluated on Y (H4). Pareto: beta sweep 0, 0.25, 0.5, 0.75, 1.
7. *Freeze.* Write the frozen config hash (objectives, betas, c, alpha, sigma, budgets, mesh, layouts, seeds) to `provenance.json` BEFORE building the test cases; re-derive and assert it
   after (`assert_frozen`, as `population_design.py`). Unit files per (fixture, objective, budget) written atomically (`.tmp` + `replace`), resumable.
8. *Sanity stops (implement as exceptions).* Uniform fixture: every objective's layout equals the area layout. Population weight must be > 0 within one candidate spacing of some
   candidate. Mass-closure and positivity as for any forward run.
9. *Statistics.* Paired per-case differences (objective minus area layout) with a percentile bootstrap interval over the 12 test cases (`bootstrap_ci`), labelled "percentile bootstrap
   interval over test cases (not a confidence interval for new scenarios)"; effect called resolved only if the interval excludes 0 for pop-RMSE AND the same direction holds in >= 75 % of
   cases; SD reported separately. 12 cases is small: say so.
10. *Outputs.* `results/phase3-population/{test_raw.csv, paired.csv, layouts.csv, figures/}` (expected artifacts of EXP-016), plus `provenance.json`, `cross_fixture.csv`, `pareto.csv`,
    `timing.json`, `README.md`. `test_raw.csv` is large: if over a few MB add it to `.gitignore` with a checksum file as `results/geodesign/RAW_FILES.json` does.
11. *Profiles.* `smoke` (cropped domain via `geodesign_smoke.json`, 2 cases, one fixture) and `pilot` (own seeds, labelled EXPLORATORY, 6 test cases) before `production`.

**Cost.** Design/validation/test = 6 + 6 + 12 = 24 bases. At 500 m/dt 100: 24 x 22 s about 9 min; at 250 m/dt 50: 24 x 140 s about 56 min (derived). Greedy over 5 fixtures x 7 objectives
x 4 budgets reuses cached bases; the pre-registered estimate is 1-2 h at 500 m. Memory below 1 GB per process. Parallelism: one process per fixture (separate `--out`/`--work`, same
mesh): objectives and fixtures do not change the bases, so ALL processes would need the same 24 bases; compute them once in a single pass first (a `--bases-only` step) and then copy the cache
directory read-only to each process, to avoid concurrent writers of the same key (section 3).

**Registration.** `research run exec EXP-016 -- .venv/bin/python scripts/population_design_belauri.py --profile production ...` (planned runs: 2; synthesise after the pilot and the production).

## 7. (c) EXP-019 follow-ups at Belauri scale — NOT IMPLEMENTED, NOT PRE-REGISTERED

**State of EXP-019 (toy).** Code: `src/pravah/windunc.py` (edc2cb9), `scripts/wind_error.py` (885d88b), configs `wind_error{,_pilot,_smoke}.json`. The pilot (6 test cases, EXPLORATORY,
467 s total) is committed (d26be8b). Production (RUN-0031, `n_test` 32, seeds 5101/5202/5303) existed in Research OS when this document was written, with its results still uncommitted in
`results/phase3-wind-error-toy/production/` (another process owned it). Check `research show EXP-019` for its outcome and any finding before starting; **do not start follow-ups until the toy production result is
synthesised and its finding independently reviewed.**

**Pre-registered toy hypotheses (EXP-019).** H1 source/reconstruction error rises monotonically with |direction error| and speed error; H2 per unit vector error, direction error is more damaging
than speed error; H3 strategy C (discrete wind-ensemble marginalisation, prior range declared a priori, no access to the true wind) beats B (fixed wrong wind) inside the prior range
and loses its advantage or becomes overconfident outside it; H4 chi-square per dof flags the wrong wind when noise is known, rank/identifiability flags do not; H5 external attribution degrades
more than local. Stop: strategy C given the true wind in any code path invalidates the run. Prior range and alpha are fixed on validation scenarios only.

**Follow-ups that would need HPC** (propose, pre-register as a variant of EXP-019 with `research experiment variant`, get the plan reviewed before running):
1. *Belauri port.* Same A/B/C strategies on the Belauri geodesign case: truth wind = Phase 2 draws rotated/scaled (`windunc.rotate_wind`), inference sees nominal, truth (oracle) or a declared prior
   grid. Needs a Belauri `Study` analogue to `windunc.Study` (the toy builds a Phase-1 `Domain` scenario; the Belauri scenario comes from `geodesign.Setup.scenario`), Belauri wind-condition banks,
   fresh seeds disjoint from 101/202/303/304/305/1101-1108/4xxx/5xxx (propose 6101/6202/6303 with noise 61000/62000/63000; unverified that these are free of other uses), and a layout set
   (greedy under nominal wind, spread, random). Wind conditions: the toy grid has 17 test conditions + 32 total winds; each distinct wind needs one basis. Cost (derived): 32 winds x 22 s = 12 min
   at 500 m/dt 100; 32 x 76 s = 41 min at 250 m/dt 100; 32 x 344 s = 3 h at 125 m/dt 100 (not recommended before section 8 is resolved). Gram quadratic forms keep the row cost small (toy: rows 10 s).
2. *Layouts designed under wind uncertainty are S5a (section 8).*
3. *ERA5-driven winds* are out of scope: `docs/winds.md` holds only a feasibility note.

## 8. (d) Conditional S5a (T-031): wind-robust and population-robust placement

Acceptance (Research OS): "Only if S4b shows a design-relevant wind effect". S4b = the EXP-019 wind-error response. Decide with the independent reviewer after EXP-019 is synthesised: is the wind error
larger than the sensor-count effect and the layout effect in a way that changes design? F-021 (preliminary) already reports that an unknown wind error dominates the sensor-count effect at 500 m.
If yes: pre-register an experiment (it does not exist yet) with: objective = expected design objective averaged over the declared wind prior (the Gram quadratic forms make this cheap: the objective is a
mixture of per-wind quadratic forms), optional population weight from EXP-016's frozen objectives, baselines = nominal-wind greedy, spread, random, evaluation on fresh test cases and on winds
OUTSIDE the prior range, freeze rules as in section 0. If no: record it as blocked/not needed with `research task block T-031 "reason"` or complete it with the justification; do not build it speculatively.

## 9. Recording results in Research OS

```
.research/bin/research current                     # state; read "Needs attention"
.research/bin/research show EXP-015                # pre-registration text: do not edit it after seeing results
.research/bin/research run exec EXP-015 -- <command>      # makes RUN-xxxx with provenance (git, command, params); exit 3 = policy block: read the hint
.research/bin/research run exec --help             # --param K=V --label --notes --detach --override
.research/bin/research artifact add --run RUN-xxxx --experiment EXP-015 -d "stage 2 outputs" results/phase3-convergence   # paths, never copied; --hash for checksums
.research/bin/research experiment synthesize EXP-015 --happened ... --worked ... --failed ... --interpretation ... --limitations ... --unresolved ... --next ...
.research/bin/research finding create "title" --statement "..." --status preliminary --confidence medium --supports EXP-015 RUN-xxxx --questions Q-011 --limitations "..."
.research/bin/research finding review F-0xx --verdict supported|contradicted|needs_work --notes "..."   # by someone else
.research/bin/research experiment variant ...      # derive a new pre-registered experiment from an old one (see --help)
```
* Everything runs through `run exec` of a REGISTERED, pre-registered experiment. Register new experiments (hypothesis, method, success, stop, parameters, expected artifacts) BEFORE running.
* **Synthesis before more than 5 runs** (`max_runs_without_synthesis` = 5). Write a synthesis after every production run in any case. Failed runs: max 3 without review.
* Findings are `preliminary` until reviewed. **No self-review.** Ask the human for a reviewer, or use `research dispatch verifier F-0xx` / the read-only `scientific-reviewer` agent; their verdict
  (`supported`, `contradicted`, `needs_work`) is recorded by the reviewer. Report numbers from the files, not from memory. Supersede wrong findings with `research finding supersede`.
* Stage 2 and the toy scripts already stamp their figures and READMEs with the experiment id and the label "Real geography; synthetic emissions and meteorology".
* After the batch: `research checkpoint create`, then commit `.research/` and the result directories (explicit paths).

## 10. State of the code at handoff

Commit ids are the last commits touching each item (`git log -1 -- <path>`), master at 3f8c8b2 when written. "Run" = production result exists and is registered.

| Item | Code (commit) | Tests | Run? | Notes |
|---|---|---|---|---|
| Solver interface, FiPy routing, regression gate | `forward.py`, `inverse.py` (Phase 2) | yes | gate captured (21 files) | rerun the gate on the cluster |
| Population layer (counts, conservative regrid, fixtures, weights, metrics) | `src/pravah/population.py` (fe9e1da) | `tests/test_population.py` | n/a | no real raster used yet |
| Resolution benchmark (T-025) | `scripts/bench_resolution.py` (96cb93f) | smoke only | yes, `results/phase3-bench/resolution_cost.csv` | 125 m level is throw-away `geo125/` |
| vanleer convection (T-026) | `forward.py` (706d1d2) | `tests/test_scheme_vanleer.py` | `results/phase3-scheme/` (42ee6ab) | nonlinear, not usable with cached responses |
| EXP-015 stage 1 | `scripts/convergence_stage1.py` (fd6c19a) | `tests/test_convergence_stage1.py` | yes RUN-0028 (228163e), F-022 superseded by F-025 | needs `geo125/` to rerun |
| EXP-015 stage 2 | `scripts/convergence_stage2.py` (2eb96e2), config | `tests/test_convergence_stage2.py` (38872b0) | **no** | section 5; about 2 h |
| EXP-017 toy population design | `scripts/population_design.py` (b6d93b5) | `tests/test_population_design.py` | yes RUN-0030 (3f8c8b2), F-023 superseded by F-026 | alpha 0.1, 16 test cases |
| EXP-018 prescribed winds | `src/pravah/winds.py` (887abbd), `scripts/prescribed_winds.py` (c000620), `docs/winds.md` (5644935) | `tests/test_winds.py` | yes RUN-0029 (3f8c8b2), F-024 superseded by F-027, F-031 | ridge amplitude too weak; finer grid is an HPC follow-up |
| EXP-019 toy wind error | `src/pravah/windunc.py` (edc2cb9), `scripts/wind_error.py` (885d88b) | `tests/test_windunc.py` | pilot committed (d26be8b); production RUN-0031 uncommitted when written | verify before relying |
| EXP-016 Belauri population design | `scripts/population_belauri.py` (61f8ae4) | `tests/test_population_belauri.py` | **no** (smoke + pilot only) | section 6; in the one-job config |
| EXP-020, EXP-021, EXP-022, EXP-023 toys | `scripts/population_design_exact.py`, `prescribed_winds_converged.py`, `population_design_greedy_test.py`, `wind_error.py` controls | yes | yes (RUN-0033/0034/0035/0036-0047) | findings F-030, F-031, F-032, F-034 (F-022/23/24/28/33 are superseded) |
| Belauri wind-error port, S5a | not written | none | no | sections 7-8 |
| Reference-mesh replicate baseline, 125 m design stage | not written | none | no | section 5 extensions |

## 11. Known gaps and cautions

* **vanleer is nonlinear.** `ResponseCache` raises for it; cached-response (superposition) design cannot use it. In EXP-015 it is used only in stage-1 field verification and, if ever, as a
  less-diffusive TRUTH generator through direct solves (inverse-crime check; not implemented). Its effective dt is smaller than nominal (sub-steps to Courant <= 0.5), so scheme and dt are confounded:
  report `dt_eff`. A vanleer design arm would need direct-solve responses per layout (not planned).
* **Stage-1 finding F-022 is preliminary** (not independently reviewed): one case (d0), one wind segment, references themselves discretised (two 125 m references differ by 0.8 %, so errors below
  about 1 % are unresolved), vanleer orders and the Richardson estimate are not asymptotic (uninterpreted), the in-study fraction of each source varies with mesh. It supports "500 m is mesh-limited, not
  time-limited" but says nothing yet about layouts.
* **Courant number and dt at 125 m.** The benchmark uses dt 100 s; at 125 m and 2.5 m/s the Courant number is up to 2 (exponential scheme: implicit, stable, accuracy untested at that ratio; stage 1
  used dt 25 s for its 125 m references). `numerical-methodology.md` shows the exponential error growing with Courant number at fixed dx. Do not use 125 m/dt 100 as a reference without a dt check.
* **Stash `stash@{0}` (G2-unshipped).** AGENTS.md calls these fixes unshipped; `git log` shows commit 0370f3e ("Apply G2 fixes: validation-scale test noise, chi2/dof, candidate border flags, git
  patch archive, stability scripts") already applied the code changes. The stash still exists and no longer applies cleanly in either direction (`git apply --check`, forward and reverse, both fail);
  I did not reconcile it. Leave it; ask the human before dropping.
* **Geodesign docs numbers predate the noise correction.** 0370f3e says the results of `geodesign` / `geodesign_robust` were not rerun, so the numbers in `docs/geodesign.md` and
  `docs/phase2-validation.md` come from the old (test-ensemble-derived) noise level. To refresh (about 1 h compute; laptop timing of `geodesign` was 708 s total, `results/geodesign/timings.json`):
  ```
  .venv/bin/python -m pravah benchmark --config configs/geodesign.json        --out results/geodesign        --overwrite
  .venv/bin/python -m pravah benchmark --config configs/geodesign_robust.json --out results/geodesign-robust --overwrite
  .venv/bin/python scripts/geodesign_tables.py        # fills the result slots in docs/geodesign.md and docs/phase2-validation.md
  ```
  Treat this as a registered rerun (an experiment record, a new finding that supersedes F-020/F-021 if numbers change); it overwrites evidence, so do it only with the human's agreement. Note that
  EXP-015/016 freeze alpha and sigma from the CURRENT `results/geodesign/alpha_selected.json` and `noise_levels.csv`: if that rerun changes them, stage 2 and EXP-016 must use the values that
  were frozen when they were pre-registered (alpha 0.01, validation scale 0.25096 ug/m3), not silently the new ones.
* Stage 2 has no config hash on its work files (section 5). Toy scripts (EXP-017/019) do check a config hash.
* Small ensembles: 6 test cases (Phase 2, EXP-015), 12 (EXP-016), so intervals are percentile bootstrap intervals over cases and say so.
* No SLURM template is tracked and none was tested. A minimal job (adapt the module/partition lines) is: `#SBATCH -c 1 --mem=4G -t 06:00:00`, `cd` to the repo, `source .venv/bin/activate`,
  `export OMP_NUM_THREADS=1`, then the `research run exec ...` command. Run only inside a job or a tmux session; `--detach` of `run exec` exists but its behaviour was not checked.
