# EXP-027b: redesigned mesh-adequacy test (task T-079; design only, nothing submitted)

Status: DESIGN, not yet registered as an experiment record, not approved for the cluster. All numbers below come from local runs of 2026-10-15 (workstation, 16 cores) or
from F-048 / RUN-0081; "derived" means arithmetic on those. Everything is synthetic emissions/winds/noise on real geography; this tests numerical adequacy of the mesh only.

## 1. Why EXP-027 was void and what the local pilots show

F-048: the 2 km control passed the paired displacement test because layouts from 6-case draws move 3-4 km between independent draws. D-012/F-048 hoped that 48-96 cases per draw would
shrink that below the control. It does not. Local pilot A (500 m cases, 4 draws x 48 cases = 192 cases; 2 km control on the same cases; `convergence_paired_padded_pilot.json`, run
with the unchanged driver, 12 workers, 26 min):

| statistic (greedy_total, 4 cells) | N = 6 (RUN-0081, 250 m) | N = 12 | N = 24 | N = 48 |
|---|---|---|---|---|
| median replicate-pair displacement, ref layouts | 3.1-4.0 km | 2.9 km | 2.8 km | 2.4-3.5 km (4 draws, 6 pairs/cell) |
| median 2 km control displacement on the same cases | 2.3-3.2 km (vs 250 m, RUN-0081) | 2.7 km | 2.5 km | 1.0-3.0 km |
| median out-of-sample replicate-pair relative regret (500 m pool resampling) | 165 % | 118 % | 49 % | 21 % |
| 2 km control passes the registered D-012 test | 4 of 4 cells | | | 3 of 4 cells (displacement), 2 of 4 (regret) |

(rows 2-4 N = 12-48: resampled disjoint draws from the 192-case pool, `power_rows`; pilot A verdict from `paired_verdict.csv`.) Pair displacement falls only from ~3.3 to ~2.4 km
over a 8x increase in N (about N^-0.15); the layouts are near-ties of a flat objective. A replicate-pair floor below the control's ~2.5 km would need N far above 48 per draw
and >= 8 draws (>= 400 250-m cases, ~190 core-h even at best packing). The D-012 design (displacement and |regret| of the mesh vs a replicate-pair floor) is therefore not
repairable inside the budget, and it is dropped, not rescaled.

Cases: bootstrapping cases for the layout-displacement floor with RUN-0081 data was not possible (only the per-draw displacement tables are in `results/exp027/`; the per-case
objects stayed on the cluster), so the scaling above is from the new local 500 m pool, which has the same case generator.

## 2. Design (the cheapest one found that can fail)

Idea: do not estimate a sampling floor at all. Evaluate the mesh on the SAME ensemble of cases at every mesh, with a fixed menu of candidate layouts. The mesh effect is then deterministic
given the cases (no design-selection noise); case-to-case variation enters only through a bootstrap over cases. Falsifiability comes from an absolute margin and from the control.

* **Cases (the only expensive part).** K = 3 independent draws x N = 24 cases = 72 cases, seeds 1201-1203 (fresh; amended from 1101-1103 after pilot B, disclosure retained) (nested prefixes of the 48-case draws; the first 6 cases of each equal EXP-027's),
  noise seeds 4000/4100/4200, each simulated once at 250 m (reference, dt 50), 500 m (main, dt 100) and 2 km (control, dt 400; D-016 origin tolerance inherited). Config
  `configs/convergence_paired_padded_exp027b.json`; the driver is unchanged. Each case is pickled and reused for layout selection and scoring (response-cache reuse across draws and
  across evaluation: every case is simulated once; draws are just groupings).
* **Cells.** noise fraction {0.05, 0.15} x budget {3, 5} (same as EXP-027; frozen alpha and noise scale unchanged; budget 8 excluded as before). Objective greedy_total.
* **Menu (per cell).** Layouts = the driver's per-draw greedy layouts at each of the 3 meshes (9) + greedy on the pooled 72 cases at each mesh (3) + greedy on 8 random 24-case sub-ensembles
  of the pool at each mesh (24, seed 27) = up to 36 distinct layouts. Fixed before any scoring.
* **Scoring.** Every menu layout is scored case by case at every mesh (design objective, `D.score_layout`, ~1 ms per layout-case).
* **Statistic.** For mesh m vs reference: shape error `E_m` = median over the menu of | (J_m(L)/median J_m) / (J_ref(L)/median J_ref) - 1 |, i.e. the ranking-relevant error of the objective with
  the common scale removed (the 2 km mesh also shifts J by 15-37 %, which alone would not change a decision). Descriptive: decision regret `R_m` = J_ref(argmin J_m)/min J_ref - 1, Spearman rho,
  median J_m/J_ref. 95 % intervals: bootstrap over the 72 cases (400 resamples, paired across meshes).
* **Staging (cost control).** Stage 1 = draws 1-2 (`..._exp027b_s1.json`, 48 cases). Evaluate the gate on them. Stage 2 (draw 3) only if the gate passes, in a fresh work dir seeded with stage-1
  case pickles (`scripts/hpc_exp027b.sh ... SEED_CASES_DIR`), so nothing is simulated twice.

Why this can fail (power/falsifiability): the control's effect is large and N-insensitive. Local proxy (500 m reference vs 2 km control on the same pool; `scripts/exp027b.py power`, menus
built inside random N-subsets, 4 partitions each, 300 bootstraps): median E_control = 0.09-0.25 at N = 12-48 (0.15-0.21 at N = 24 in the 0.05-noise cells, 0.09-0.16 at 0.15), E lower 95 % bound
above 0.05 in 4/4 cells in every partition at N = 12, in 4/4 cells in 3 of 4 partitions at N = 24 (the fourth gave 2/4), and in 3/4 cells in every partition at N = 48. That is P(gate) ~ 0.9 at 24 cases
(derived 10 of 11 partitions); the real pool is 72 cases (three times more), so the interval is narrower. Local pilot B (250 m, 500 m, 2 km on 2 x 12 cases): E_control = 0.14-0.31 with
lower bounds 0.095-0.18, E_main(500 m) = 0.04-0.10; J_500/J_250 median 0.87-0.95 (500 m underestimates J by 5-13 %). So the pilot data say the 2 km control is detectably bad and 500 m is
close to, perhaps above, the 5 % margin: the experiment can return INADEQUATE, and passing (E_hi95 <= 5 %) is the unlikely outcome with 72 cases (a 24-case interval is 0.08-0.22 wide).

## 3. Registered criteria (ready to paste; margins are open decision D1)

> EXP-027b registered before any 250 m case is built. Meshes: 250 m / dt 50 (reference), 500 m / dt 100 (main), 2 km / dt 400 (falsification control, D-016 tolerance). Ensemble: K=3 independent
> design draws (seeds 1201-1203 (fresh; amended from 1101-1103 after pilot B, disclosure retained)) x N=24 cases = 72 cases, each simulated once per mesh; stage 1 uses draws 1-2 only. Cells: noise fraction {0.05, 0.15} x budget {3, 5}; greedy_total; alpha and noise
> scale frozen as in EXP-027. Per cell a menu of candidate layouts is built from greedy selections at each mesh (per-draw layouts, pooled-ensemble layouts, 8 random 24-case sub-ensembles per mesh,
> seed 27) and scored case by case on every mesh. Shape error E_m = median over the menu of |(J_m(L)/median J_m)/(J_ref(L)/median J_ref) - 1|; its 95 % interval is a case bootstrap (400 resamples).
> (G) FALSIFIABILITY GATE: the 2 km control must have E_lo95 > 0.05 in at least 3 of the 4 cells. If not, the experiment is VOID and no statement about the 500 m mesh is made. (If the gate fails on
> the stage-1 cases, stage 2 is not run.)
> (A) ADEQUACY OF 500 m: ADEQUATE iff E_hi95 <= 0.05 in all 4 cells; INADEQUATE iff E_lo95 > 0.05 in any cell; otherwise INCONCLUSIVE. Adequacy is claimed only for this synthetic domain, these cells, these
> margins.
> Descriptive, not decisive: decision regret R, Spearman rho, median J ratio, mean site displacement. No replicate-pair displacement test (shown non-estimable at N<=48). No criterion, margin, menu rule or
> cell may be changed after the stage-1 evaluation. Pre-run disclosures: the margin 0.05 follows the DYN-05 convention (F-062) and was fixed after local pilots showed E_control 0.09-0.31; local pilot
> B (24 of the 72 cases: the first 12 cases of draws 1-2 at the three meshes) was looked at and gave E_main 0.04-0.10, so ties to the margin are expected.

Outcome table: gate fail -> VOID (second time; then stop mesh work, keep 500 m provisional). Gate pass and ADEQUATE -> 500 m / dt 100 supported for the E<=5 % sense. INADEQUATE or INCONCLUSIVE ->
500 m stays provisional; INADEQUATE would motivate a 250 m check inside DYN-08.

## 4. Cost model (CPU-hours; hard cap 35 billed, total remaining ~88 shared)

Measured locally (single core unless noted): 500 m case 43 s, 250 m case 288 s, 2 km case 5.8 s. With 11-12 workers busy (memory bandwidth) 500 m takes 78 s (x1.8) and 250 m 595 s (x2.1).
RUN-0081: 32 cores x 44 min = 23.5 core-h billed for 48 case triples = 0.49 core-h per triple, but its 250 m phase was one unit per draw (8 draws on 32 cores), so ~60 % of the reservation idled; the
driver now schedules one unit per (mesh, draw, case).

| | cases / mesh | compute (serial) | with contention | billed, 70 % efficiency | billed, RUN-0081 rate (pessimistic) |
|---|---|---|---|---|---|
| stage 1 (K2 x N24) | 48 | 4.5 h | 8.1 h | 11.6 h | 23.5 h |
| stage 2 (draw 3 only) | +24 | 2.2 h | 4.0 h | 5.7 h | 11.7 h |
| total (K3 x N24) | 72 | 6.7 h | 12.1 h | 17.3 h | 35.2 h |
| evaluation (`exp027b.py evaluate`) | | ~0.02 h | | | |

Wall time on 32 cores: 44 min (stage 1) / 66 min (total) at the pessimistic rate. The pessimistic total just meets the 35 core-h target; the expected total is ~17 core-h. Reservation
(billed = reserved, not used): suggest 24 cores x 1.5 h = 36 core-h as the ceiling for stage 1 only if the user accepts it; 16 cores x 1.5 h = 24 core-h is enough at the expected rate and caps stage 1 at 24.
`python scripts/exp027b.py plan [--config CFG]` prints this table (dry run, no data needed).

What is lost under the cap: (i) no replicate-pair sampling floor, so no statement "mesh effect is below sampling noise" (it would need >= 400 250-m cases and, per section 1, still may not be estimable);
(ii) only 3 draws: variation of the case set is covered by the bootstrap, not by independent re-selection of layouts; (iii) with 72 cases a pass (E_hi95 <= 5 %) is unlikely even if 500 m is fine,
so the likely best outcome is INCONCLUSIVE, not ADEQUATE; (iv) single seed family and a menu limited to greedy-derived layouts; (v) no 250 m replication of the 500 m budget-8 case. Cutting to stage 1
only (48 cases, <= 23.5 core-h) keeps the gate test and a wider interval.

## 5. Instrumentation

`scripts/hpc_usage_report.sh` (new; none existed besides `campaign_usage.py`, which needs hand-fed job ids): sums sacct billed (AllocCPUS x ElapsedRaw) and used (TotalCPU) CPU-hours of jobs whose
name starts with `EXP-` since `--since`, with `--budget 88` printing the remainder. Run on the login node or as a 1-CPU job; `--selftest` and `--input FILE` (parsable2 dump,
`--print-format` shows the fields) work offline. Tested only on canned input here (no cluster access from this task); the `sacct -X --parsable2 --noheader` call itself is untested.

## 6. Pilot evidence (local, small; scratch outputs not committed)

* Pilot A (4 x 48 cases at 500 m + 2 km, driver `--paired --workers 12`, 26 min wall, ~5 core-h): driver end-to-end OK; control passes the D-012 test in 3/4 cells (displacement) -> D-012 is non-falsifiable
  at N=48; layouts exist for all (draw, mesh); resampling power table in section 2.
* Pilot B (250 / 500 / 2000 m, 2 draws x 12 cases, `convergence_paired_padded_pilot250.json`, 25 min wall): driver `--paired` then `exp027b.py evaluate`: 8 cell rows, gate passes in 4/4 cells
  (E_control lower bounds 0.095-0.18), main status INADEQUATE in 2 cells / not adequate in any (E_main 0.04-0.10; intervals 0.08-0.22 at 24 cases). So the 2 km control CAN fail on a small
  configuration and the evaluation pipeline runs end to end. Not a test of anything: 24 cases.
* `exp027b.py power` on pilot A: section 2.

## 7. Files

New: `scripts/exp027b.py` (plan / power / evaluate), `scripts/hpc_exp027b.sh` (cluster launcher, submits nothing), `scripts/hpc_usage_report.sh`, `configs/convergence_paired_padded_exp027b.json`,
`..._exp027b_s1.json`, `..._pilot250.json`, this file. Unchanged: `scripts/convergence_stage2.py` (case-level units already exist), `src/`. The earlier `..._padded_48.json` / `..._pilot.json` configs are not used by the registered design.

## 8. Open decisions (user)

D1. Margin: 0.05 (above) vs 0.10. At 0.10 the 2 km control's lower bounds fail the gate in the 0.15-noise cells (pilot A proxy E_control 0.09-0.16 at N=24), so 0.10 makes the gate weaker; at 0.05 the 500 m mesh may be
   called INADEQUATE or INCONCLUSIVE. Decide before any 250 m case runs; do not tune after stage 1.
D2. Approve stage 1 only (<= 23.5 core-h pessimistic, ~12 expected) or both stages; reservation size (16-24 cores x 1.5 h suggested).
D3. Whether to make a stronger control (e.g. 1 km) in addition: not added (a new padded domain/mesh level would need re-preparation and exceed the cap).
D4. Whether D-006/D-012/F-048 records get a pointer to this design; I did not touch `.research/` state.
D5. Pilot B cases (24 of 72) were seen before registration; either accept that disclosure or move the production seeds (e.g. 1201-1203; changing seed0 changes every case).
