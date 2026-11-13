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
3. **E08 running:** design committed, 32² pilot done, 64² sweep is job 205039 (~19 h; see `experiments/e08.md`). Then the
   1200 m exclusion rerun on cached fields, the report, and a gate-08 audit.
4. HPC verified 2026-11-12 (smoke job 204775: 175 tests passed on node004; limits 64 CPU, 4 GPU, 2 days in settings.yaml).
5. E09 done and audited. E10 and E11 32² pilots done (see journals). Next: joint gate-10/11 audit; E11 64² run after E08
   (one job); E08 report and exclusion rerun. Plan 3 (E12–E20) after that.

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
| e08 | Sensor sufficiency | design committed; pilot done; 64² sweep running (job 205039) | [e08](experiments/e08.md) |
| e09 | Causal Kalman filter | 32² replay + gate-09 (pass w/ conditions): all-past-data MAP ≫ 24 h MAP; placement ordering flips | [e09](experiments/e09.md) |
| e10 | Joint source–field assimilation | 32² pilot: q=0 best for constant sources; random-walk filter wins and is calibrated for time-varying ones | [e10](experiments/e10.md) |
| e11 | Forecast placement | 32² pilot: no design best under both memories; wind-robust 24 h objective fixes dopt's map24 failure | [e11](experiments/e11.md) |
| e12–e20 | see plans (plan 3) | not started | — |

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
