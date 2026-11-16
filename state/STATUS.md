# PRAVAH — current status

**Status recorded:** 2026-11-14 (after E19 synthesis and gate-12-17)

**Working plan:** `plans/plan-03-robustness-synthesis.yaml` (E12–E20). Plans 1 and 2 are complete as executed.

## Current phase

- **Plan 3 is nearly complete.**
  - E12–E17 have results (32² pilots; E17 also at 64²).
  - E18 was not conducted: no data (user, 2026-11-14).
  - E19 synthesis is done: `reports/evidence-matrix.csv`, `scripts/validate_report.py` (0 problems) and the
    conclusions chapter.
- **Next: E20, independent replication and release review.**
- **Heavy 64² sweeps are deferred** (user, 2026-11-13). Each design note documents its cluster plan. Done at 64²: E07,
  E08 (and stage-2 reruns), E11 and E17.

## Key results (details in the journals and `reports/quarto`)

- **E07 confirmatory:** dopt beats random by ≥ 10 % in the 8 cells run (scope user-approved).
- **The strategy ordering depends on estimator memory** (E07, gate-09, E08 rerun, E16):
  - with a 24 h window, source_near beats dopt;
  - with all past data, dopt beats source_near.
- **Sufficiency:** k*(0.2) ≈ K for location-aware designs (E08 64²). With all past data, k* = 2 for every strategy:
  a genuine floor, but it depends on constant sources and a perfect model.
- **Model error dominates placement.** Wind errors ≥ 30° or a timing lag, and wrong source shapes, cancel placement and
  memory gains (E12, E13, E17). E17's wrong-wind collapse is a transport floor.
- **Self-diagnosis is moderate** (AUC 0.71 within configuration, 0.74 pooled). A common-mode bias is a structural
  silent failure.
- **Robust placement:** E15's tail gain is indistinguishable from chance.

## Audits
- gate-07, b3, gate-09 and gate-10-11: all pass or pass with conditions.
- gate-12-17 covers E12–E17 and the 64² runs of E08 (stage-2 reruns) and E11; all pass or pass with conditions.
- Unaudited: the E16 VanLeer cells (added after gate-12-17) and the main E08 64² sweep tests (H1–H5).

## Open items
- **Bugs:** the gate-12-17 bugs and E10 kf_qsel coverage were fixed 2026-11-16 (old → new in journals e08, e10,
  e12–e17). Main change: E14's omitted source is no longer detected by informed networks (AUC 0.84–0.89 → ≈ 0.64).
- **E03:** remains rank deficient for structural reasons.
- **Missing run records:** the E06, E08 pilot, E11 v2 pilot, E15 pilot and E17 32² pilot runs (see the evidence-matrix
  caveats).
- **Citations:** ERA5 and WorldPop are cited only in chapter text, not as bib entries.
- **HPC:** verified 2026-11-12. Limits are in settings.yaml; the current policy is in memory and settings (one 64-CPU job,
  heavy runs deferred).

## Experiments

| Exp | Title | State | Journal |
|---|---|---|---|
| e00 | Protocol and evaluation contract | approved 2026-11-12 (amendable baseline); amendments A1, A2 | [e00](experiments/e00.md) |
| e01 | Forward solver verification | done at 16²–64²; conditionally cleared | [e01](experiments/e01.md) |
| e02 | Synthetic benchmark | smoke pipeline works; not frozen | [e02](experiments/e02.md) |
| e03 | Static NNLS inverse | rerun at 1 g/s; still rank-deficient (structural) | [e03](experiments/e03.md) |
| e04 | Footprints / Fisher information | rerun at 1 g/s, 1 replicate; D-opt design 4/4 practical rank | [e04](experiments/e04.md) |
| e05 | Temporal observability | smoke, 20 seeds, audited (gate-05) and rerun; exploratory support | [e05](experiments/e05.md) |
| e06 | Placement algorithms | smoke done, audited (gate-06); 3 conditions before E07 reuse | [e06](experiments/e06.md) |
| e07 | Cross-objective placement | closed: confirmatory supported in cells run; source_near beats dopt; gain shrinks with in-footprint sites excluded | [e07](experiments/e07.md) |
| e08 | Sensor sufficiency | 64² sweep + reruns done; H5 reverses with all past data; stage-2 audited (gate-12-17) | [e08](experiments/e08.md) |
| e09 | Causal Kalman filter | 32² replay + gate-09 (pass w/ conditions): all-past-data MAP ≫ 24 h MAP; placement ordering flips | [e09](experiments/e09.md) |
| e10 | Joint source–field assimilation | 32² pilot: q=0 best for constant sources; random-walk filter wins and is calibrated for time-varying ones | [e10](experiments/e10.md) |
| e11 | Forecast placement | 64² (20 clusters): out-of-sample wind-robust design beats dopt under map24 (+0.34 at k = 8, 19/20); loses to source_near; worse under map_grow | [e11](experiments/e11.md) |
| e12 | Wind mismatch | 32² pilot: direction/timing error destroys forecast skill; estimators compensate by distorting sources (good fit, wrong strengths) | [e12](experiments/e12.md) |
| e13 | Model mismatch | 32² pilot: background offset fixable by estimating it (but indistinguishable from common-mode bias → silent failure); source-basis errors unfixable | [e13](experiments/e13.md) |
| e14 | Reliability diagnostics | 32² pilot: truth-free warnings only moderate (AUC 0.71 within config, 0.74 pooled); speed errors and common bias undetectable; missing source missed by random networks | [e14](experiments/e14.md) |
| e15 | Robust placement | 32² pilot: robust ≈ source_near under map24; small tail gain vs dopt only at k = 2; no gain under map_full; 64² deferred (low priority) | [e15](experiments/e15.md) |
| e16 | Generalization/ablation | 32² pilot (8/8 cells): no conclusion reversed; memory flip in every cell; 64² deferred (≈2,760 CPU-h) | [e16](experiments/e16.md) |
| e17 | Regional case (Belauri) | 64² done: placement beats random; dopt≈source_near; only map_grow beats persistence; a nearest-point wind cancels all gains | [e17](experiments/e17.md) |
| e18 | Observational validation | gap assessment done: only 2 DoEnv PM2.5 stations within 100 km, sparse record; held-out test not defensible as validation; **not conducted** (user, 2026-11-14: no data available); gap assessment is the final artefact | [doc](../docs/research/observational-validation-audit.md) |
| e19 | Synthesis | done: evidence matrix (82 claims), validator, conclusions chapter, manuscript docs | [matrix](../reports/evidence-matrix.csv) |
| e20 | Replication/release review | not started | — |

## Verified facts vs questions

Verified (smoke scale, synthetic; audited where a gate file exists):
- Upwind/backward-Euler FiPy solver converges at first order and conserves mass to ~1e-14 (gate-01).
- Benchmark generator is deterministic, the leakage guard works, and the frozen manifest matches at a_ref = 1 g/s.
- At a_ref = 100 µg/s sensors saw no plume (SNR ~1e-3); at 1 g/s peak SNR is 20–30.
- Source-informed placement (D-opt, source_near) reaches full practical rank where random/coverage designs do not (E04, E06; in-sample).
- Under an exactly constant wind a longer window adds no rank; wind changes do (E05 v2, exploratory).

Open:
- VanLeer is ~1 % non-linear (one test); its effect on G-based inference at 64² is unmeasured.
- Everything at 1 g/s is exploratory (scale chosen after seeing data); confirmatory tests start with E07/E08 designed under the approved protocol.

## Pointers

- [Scientific plans](../plans/) · [Settings](../settings.yaml)
- [Gate audits](../docs/research/gates/) — the most critical reading of each experiment
- [Quarto report, tables, figures](../reports/quarto/)
- [Run records](../reports/runs/) · [Decisions](decisions/) · [History](history/)
