# PRAVAH — current status

**Status recorded:** 2026-11-12 (after amendment A1 reruns of E02–E05, E06, and gate-05/06 audits)

**Working plan:** `plans/plan-02-identifiability-placement-forecast.yaml` (E04–E11).
Plan-01 (E00–E03) has been executed at smoke scale; plan-03 (E12–E20) is not started.

## Current phase

Plan 2. E04–E06 done at smoke scale (32², local, exploratory), each with an independent audit. E07 64² sweep done and audited; follow-up controls open.
Amendment A1 (a_ref = 1 g/s, `decisions/2026-11-12-source-scale.md`) was approved by the user and applied on 2026-11-12.
Protocol and `configs/benchmark-v1.yaml` APPROVED 2026-11-12 as an amendable baseline (`decisions/2026-11-12-protocol-approval.md`). HPC verified (smoke job only).

## Next action

1. **B3 resolved by amendment A2** (`decisions/2026-11-12-advection-scheme.md`): 64² runs use split VanLeer, dt 30 s
   (K_eff ≈ 50 vs 350–2900 for upwind; gates met). Before the 64² sweep, measure VanLeer's superposition error on benchmark footprints.
2. **E07 closed** (sweep + gate-07 + follow-ups + gate-09). Confirmatory supported in the cells run (user-approved scope).
   source_near beats dopt (115/120) ONLY for the 24 h-window estimator; with all past data dopt wins (103/120, gate-09), and with in-footprint sites excluded dopt's gain over random drops to 0.20 (0/8 cells).
   dopt's loss is concentrated in the unseen southerly regime. See `experiments/e07.md`.
3. **E08 64² sweep done** (job 205039, 18.8 h). Holm family: H1, H3, H4 and H5 supported; H2 contradicted. k*(0.2) ≈ K
   for location-aware designs. Stage-2 reruns done (job 205394): with all past data every strategy reaches τ = 0.2
   at k = 2 and H5 reverses (dopt beats source_near 36/40). With 1200 m exclusion, about twice the sensors are needed.
   Pending: gate-08.
4. HPC verified 2026-11-12 (smoke job 204775: 175 tests passed on node004; limits 64 CPU, 4 GPU, 2 days in settings.yaml).
5. E09 done and audited. E10 and E11 32² pilots done (see journals). Gate-10/11 done (pass w/ conditions; E10 diurnal gain is 1 h only; ~40 % of E11's gain was in-sample).
   E11 64² done (job 205282): the out-of-sample wind-robust design beats dopt under map24 (19/20 clusters at k = 8), loses to source_near, and is worse under map_grow; E08 report and exclusion rerun. Plan 3 started: E12 pilot done. From now on, heavy 64² sweeps are deferred (user, 2026-11-13): pilots plus documented submission plans.

Known open items: E03 (fixed-sensor static inverse) remains rank deficient for structural reasons (sensor placement, 24 h window).

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
| e08 | Sensor sufficiency | 64² sweep + reruns done; H5 reverses with all past data; audit pending | [e08](experiments/e08.md) |
| e09 | Causal Kalman filter | 32² replay + gate-09 (pass w/ conditions): all-past-data MAP ≫ 24 h MAP; placement ordering flips | [e09](experiments/e09.md) |
| e10 | Joint source–field assimilation | 32² pilot: q=0 best for constant sources; random-walk filter wins and is calibrated for time-varying ones | [e10](experiments/e10.md) |
| e11 | Forecast placement | 64² (20 clusters): out-of-sample wind-robust design beats dopt under map24 (+0.34 at k = 8, 19/20); loses to source_near; worse under map_grow | [e11](experiments/e11.md) |
| e12 | Wind mismatch | 32² pilot: direction/timing error destroys forecast skill; estimators compensate by distorting sources (good fit, wrong strengths) | [e12](experiments/e12.md) |
| e13 | Model mismatch | 32² pilot: background offset fixable by estimating it (but indistinguishable from common-mode bias → silent failure); source-basis errors unfixable | [e13](experiments/e13.md) |
| e14 | Reliability diagnostics | 32² pilot: truth-free warnings only moderate (AUC ≤ 0.71 overall); slow-wind and missing-source failures undetectable | [e14](experiments/e14.md) |
| e15 | Robust placement | pilot running (agent) | — |
| e16 | Generalization/ablation | 32² pilot running (agent) | — |
| e17 | Regional case (Belauri) | 64² done: placement beats random; dopt≈source_near; only map_grow beats persistence; a nearest-point wind cancels all gains | [e17](experiments/e17.md) |
| e18 | Observational validation | gap assessment done: only 2 DoEnv PM2.5 stations within 100 km, sparse record; held-out test not defensible as validation; **not conducted** (user, 2026-11-14: no data available); gap assessment is the final artefact | [doc](../docs/research/observational-validation-audit.md) |
| e19–e20 | synthesis, audit/release | not started | — |

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
- Held-out (forecast / off-sensor) evaluation has not been run anywhere yet.

## Pointers

- [Scientific plans](../plans/) · [Settings](../settings.yaml)
- [Gate audits](../docs/research/gates/) — the most critical reading of each experiment
- [Quarto report, tables, figures](../reports/quarto/)
- [Run records](../reports/runs/) · [Decisions](decisions/) · [History](history/)
