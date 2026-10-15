# DYN-03 causality and detection fix (review F-053/R1)

Code: `src/pravah/studies/toy/dynamic_history.py`. Protocol: D-024 (supersedes D-014 for EXP-031). Fix committed in `dfa3988`; this note documents and verifies it.

## What was wrong
1. CT and CAUSAL were not causal. `run_episode` fitted the M0 static prior on the whole 6 h window and passed it as `theta0` to both. Bin-b estimates therefore used observations from after bin b. CT was in practice the full-window static fit.
2. `detect()` never checked that the estimate was on the pre-change side before the change bin. A constant estimate already past the pre/post midpoint was scored as "detected, delay 0".

## What changed
1. `estimate()` (CT, CAUSAL): for each bin k the prior `pk` is the M0 ridge fit (frozen alpha `a0`) on observations with `bin_index <= k` and columns of bins `<= k` only. CT then fits bin k from observations in bin k and bin-k columns. CAUSAL fits M2 on observations `<= k` and columns `<= k` and keeps only bin k. The `prior` argument is still accepted but is ignored by CT/CAUSAL.
   Lag: bin k owns times in (t_k, t_{k+1}]; the estimate for bin k is available at t_{k+1}, i.e. at the END of the bin (no look-ahead beyond the bin). Transport delay (about 3 h over 12 km) means a bin's emission is mostly seen in later bins. A causal estimator cannot see that, which is why CT needs a huge alpha. That is a physical consequence, not a leak.
   RETRO (full-window M2) and M1 (full-window independent bins) keep the whole-window M0 prior. They are explicitly NON-causal (retrospective).
2. `detect(series, pre, post, c, e)`: let `s = sign(post-pre)`, `mid = (pre+post)/2`, `crossed[b] = s*(series[b]-mid) > 0`. If `c >= 1` and `crossed[c-1]`, the result is a miss (a constant or already-crossed estimator is not a detection). Otherwise the delay is the first `b` in `[c, e)` with `crossed[b:e].all()`; miss if none. `run_episode` scores a change only if `c >= 1` and the parameter is identifiable at bins `c-1..e-1`. Changes at bin 0 are not scored.
3. Seeds: `TEST_SEED0 = 8600` (test 8600-8695). 7600-7695 are consumed.

## Tests (`tests/test_dynamic_history.py`, 10 tests, about 1 s)
- `test_causal_estimates_ignore_future_data_including_the_prior`: perturb observations in bins >= 4; CT and CAUSAL estimates for bins 0-2 must be unchanged (atol 1e-9).
- `test_causal_uses_no_future_data`: same for CAUSAL with an explicit prior; perturbation at each b leaves bins <= b unchanged to 1e-12.
- `test_constant_estimate_on_post_side_is_not_a_detection`: constant post-side series gives a miss; a genuine step at c gives delay 0.
- `test_test_episodes`: seeds 8600-8695.
Checked against the pre-fix module (`dfa3988^`): `test_causal_estimates_ignore_future_data_including_the_prior`, `test_constant_estimate_on_post_side_is_not_a_detection` and `test_test_episodes` FAIL there; all 10 pass now.

## Validity of old results
F-053, RUN-0068 and RUN-0084 (test seeds 7600-7695, RUN-0092) are INVALID for CT and CAUSAL numbers (leaked prior), for the CT/CAUSAL miss rates and delays (detection artefact), and for the "all 12 primary contrasts meet the criterion" headline (CAUSAL-CT and the steady-wind negative control were contaminated). RETRO and M1 numbers and the M1/RETRO versus CAUSAL contrasts are affected only through CAUSAL. The reviewer reproduced CAUSAL/RETRO detection numbers (miss 1.7-5.2%).

## Rerun proposal
Validation (tune) hyperparameters were already regenerated under v2: `results/dyn03v2/frozen_hparams.json`, hash `3c86b5e668181a3d` (commit `05049a5`). Test stage only, through the registry:

    .research/bin/research run exec EXP-031 -- .venv/bin/python -m pravah.studies.toy.dynamic_history --stage test --freeze results/dyn03v2/frozen_hparams.json --out <RUN_DIR> --workers 16

Episodes: `episodes_test()`, 96 episodes (6 winds x 8 families x 2 seeds), seeds 8600-8695, never viewed. Do not reuse 7600-7695 (viewed).
Cost, measured locally (1 worker, 2 episodes, both with all 5 models, 2 layouts x 2 noise): 14.1 s and 12.2 s per episode, so about 13 s x 96 = about 21 CPU-min = 0.35 CPU-h (about 2 min wall on 16 workers, plus pool start-up). Re-tuning is not needed and would cost much more (48 validation episodes x a 42-candidate grid); do it only if the freeze is rejected.

## Decisions the human/registry must make
- D-024 retired the steady-wind negative control, but the reviewer's point stands that the stop rule (negative-control gain CI excluding 0 means invalid analysis) fired in RUN-0092 and was not reported. Confirm that retiring it after seeing the result is acceptable, and that F-053 must state the firing. The v2 replacement prediction is relative (gain larger on variable winds in relative terms); the reviewer found it already holds relatively (e.g. L4 2%: -49.6% steady vs -59.9% variable) on the old, leaky run. Decide whether relative or absolute is the decisive reading, since D-014 did not say. The code reports both.
- The v2 frozen hyperparameters have many edge selections (CT alpha at the 1e9 grid edge in L4, CAUSAL alpha at 0.01 in every cell, M0 alpha at 0.01). Decide whether to widen the grid (needs a re-tune and a new freeze hash) or to accept and report them.
- "Gain comes from carry-over" was never tested; a tested claim needs its own registered check.
- Seeds 8600-8695: confirm they are still unviewed before the run; any look at them forces new seeds.
