# PRAVAH reproducibility contract (T-003)

Implements, and never overrides, `preregistered-protocol.md` (still DRAFT pending human approval; its numeric choices are the config defaults). Changing a default in `src/pravah/experiment_config.py` is a scientific amendment.

## What exists
- `src/pravah/experiment_config.py` — stdlib-only typed config (`ExperimentConfig`), JSON files (`dump_config`/`load_config`; unknown keys and invalid values are rejected), `config_hash()`.
- Units: length m, time s, concentration ug m^-3, source strength ug s^-1, areal flux ug m^-2 s^-1 (`ModelConfig.units`, recorded in every provenance file).
- Seeds: replicate s in 0..19 → truth 1000+s, noise 2000+s, placement 3000+s; bootstrap 4000. Out-of-range replicates raise.
- `truth_id(cfg, s, K)` — deterministic `TRUTH-…` hash of truth-side config, truth seed and K_s only (no noise or estimator settings).
- Metrics: `MetricDef` list with name, unit, definition; exactly one primary (`forecast_rmse_1h_preassim`), enforced by `validate()`.
- No-leak split: `Split` (train 1–30, validation 31–42, test 43–60; must be contiguous/disjoint) and `check_no_leak(split, purpose, days, wind_regimes)` raising `LeakError` when test days or held-out wind regimes feed tuning/placement/model selection, or when headline evaluation uses non-test days.
- Provenance: `write_provenance(run_dir, cfg, command, replicate, n_sources, run_id, manifest)` writes `outputs/provenance.json` (command, git SHA + dirty flag, seeds, full config + hash, truth ID, run ID, units, metric definitions, data manifest, machine, Python) and registers it. `DataManifest` hashes input files (SHA-256) and `verify()` reports changed/missing ones.
- Resource profiles: `local` (4 CPU, 8 GB, 2 h) and `hpc` (slurm, 8 CPU, 16 GB, 6 h). These are declared ceilings, **not** submissions: any `research hpc submit` still needs a dry-run and explicit human approval, and the protocol keeps the cluster off until E01 verifies.
- Output registration (HPC-correct): write under `$RESEARCH_RUN_DIR/outputs/`; `log_metric` appends to `metrics.jsonl`, `register_artifact` appends the absolute path to `artifacts.jsonl`; unregistered files stay on the cluster. Missing files and non-finite metrics raise.
- `reports/quarto/` — comprehensive report (book, one chapter per experiment E00–E03; extend as experiments land). `build.py` refuses to render unless every file in `data/` is listed in `data/MANIFEST.json` with a matching hash. Chapters read their tables from `data/eNN_summary.csv`; no values are typed in. This is separate from the curated Scientist Guide (`docs/scientist-guide/`), which is untouched, as are Research OS–generated files.

## Tests
`PYTHONPATH=src python -m pytest -q tests/test_experiment_config.py` (frozen) covers defaults vs protocol, one-primary rule, split ordering failures, round-trip/hash, unknown-key and bad-value rejection, truth-ID determinism, leak guard, wind disjointness, resource validation, manifest tampering, provenance and registration.

## Limitations
- Defaults inherit the protocol's unapproved judgement calls. Config validation checks internal consistency, not scientific adequacy.
- `check_no_leak` is a guard callers must invoke; it cannot see what an estimator actually reads. The frozen estimator-level leakage test (future observation changes nothing) belongs to E03.
- `truth_id` hashes configuration, not generated truth fields; E02 must also hash the generated arrays into the manifest.
- Git SHA is read from the current directory's repo; `git_dirty` flags uncommitted changes but does not store a diff.
- The Quarto build was verified for manifest checking only; no experiment tables exist yet and no render was run.
- Metrics definitions are text; metric code is not implemented here.
