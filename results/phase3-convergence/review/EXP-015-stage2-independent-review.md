# Independent review — EXP-015 stage 2 (RUN-0058)

Reviewer: `scientific-reviewer` agent (read-only, not the author of the result).
Author: the agent that ran RUN-0058. The author did not review its own finding.
Date: 2026-10-12. Finding under review: **F-035**.

The reviewer recomputed every number from the CSVs independently and read the driver source.
Its conclusions were then re-verified by the author in the data before being accepted
(48/48 positive `vs_main` regret; signed mesh regret -6.8% to -11.5% of J_ref;
`eps_sd/J_ref` 1.01-2.52). **The review is authoritative over the author's first reading.**

## Verdict

- **H1 is UNTESTED, not supported.** The mesh comparison and the resampling baseline are not
  commensurable: `scripts/convergence_stage2.py:244` and `:258` build the design cases for every
  mesh from the same seed-101 draws (zero case-draw variability), while `:274` builds each
  replicate from a different draw (full case-draw variability). A one-sided `mesh <= baseline`
  test therefore cannot fail — a deliberately broken 2 km mesh would pass too.
- **The registered criterion yields no verdict.** It tests `IQR_low <= x <= IQR_high` while the
  hypothesis says "no larger than". All six failures are below the floor, never above the ceiling.
  The one-sided reading is a **post-hoc repair**, not a reinterpretation, and must be registered
  as a dated amendment before it is reused.
- **500 m / dt 100: provisionally admissible** for EXP-026 — "no evidence against", not
  "shown adequate". If EXP-026's conclusions prove mesh-sensitive, this evidence will not protect them.

## Author claims the review overturned

| Author's first claim | Review verdict |
|---|---|
| "H1 is supported" | **Overstated** — H1 is untested (non-falsifiable baseline) |
| "Held-out test: 500 m is 6-35% better, never worse" | **Retracted** — one of six cases; LOO -4.0%/-8.7%; Wilcoxon p 0.22-0.56; no detectable difference |
| "The objective is very flat" | **Mis-diagnosed** — `eps_sd/J_ref` = 1.01-2.52, so the tie threshold exceeds the objective; the real problem is the objective is not identifiable at 6 cases |
| "dt has no effect" | **Over-scoped** — dt 100 vs 50 only; dt 25 never run |
| "failed in all 6 cells for the same reason" | **Wrong in one cell** — at f0.05 b5 displacement was inside the IQR; only regret failed low |
| "6 agreeing cells" | **Effective n = 2** — budgets 3/5/8 are nested prefixes of one greedy trace (`:176-177`) |

## Findings the author had missed

- **Mesh regret is NEGATIVE** at noise 0.05 (-6.8% to -11.5% of J_ref), falsifying
  `docs/numerical-methodology.md` section 5's non-negativity expectation and bounding greedy
  sub-optimality at >=7-12% at the reference mesh. "Regret vs the reference layout" therefore
  partly measures greedy path-dependence — and benignly explains the apparent test advantage.
- **Noise 0.15 is uninformative** for mesh: sampling noise ~1.3 x J_ref swamps the mesh difference,
  so its three zeros are one uninformative observation, not three confirmations.
- **Test seed 303 is now burned** and cannot serve as held-out for EXP-026. Seed 304 is unspent.
- The 125 m arm was dropped although its registered stop condition (250 m stage > 4 h) was **not
  met** (the run took 1h10m) — an undeclared deviation, not an invoked stop rule.
- 1 of 142 interior candidates (`int_c009015`) changes study category between meshes.

## Verified clean

No test-set leakage: `:289` asserts every layout file exists before `:293` builds the test cases;
alpha is an exact-equality **assert** against `results/geodesign/alpha_selected.json` (`:150-156`),
stronger than a read; the noise scale is frozen (`:142-147`) and passed to workers by `initargs`;
ensemble disjointness is machine-checked (`:139`). The epsilon-swap scan runs after the test build
but scores on design cases only and feeds no selection.

## What the reviewer could NOT verify

Physics and solver internals (no forward solves re-run), `greedy_trace`/`fit_layout` correctness,
re-derivation of the CSVs from the layouts JSONs (the 721 MB response-basis cache was deliberately
left on the cluster), the figures, stage-1 numbers, and any empirical validity.

## Required before this finding can advance beyond preliminary

1. The paired baseline: select at 500 m **and** 250 m on each of the 8 replicate draws, so the two
   distributions are commensurable. Pre-register as an EXP-015 variant first.
2. Register the two-sided -> one-sided criterion repair as an explicit dated post-hoc amendment.
3. Keep the retractions above in any downstream write-up.
