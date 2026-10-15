# Consolidated HPC batch proposal (wrap-up sprint) — 2026-10-15, NOT YET APPROVED

Budget: Research OS `EXP-*` jobs have billed ~423.8 core-h (sacct, AllocCPUS x elapsed) of the 512 ceiling -> ~88 left. 319 of those core-h were on jobs that
requested a GPU the CPU-only code never used. This batch requests **no GPUs**.

| # | Job | Purpose | Cores x wall (reservation) | Expected core-h (billed) | Depends on |
|---|-----|---------|------------------|------------|-----------|
| 1 | DYN-08 `--stage time` | measure cluster core speed, design-search time at 12 winds | 1 x 20 min | 0.33 | staging of inputs |
| 2 | EXP-026 (WorldPop real layouts, amendment filed first) | reconcile EXP-026; feeds DYN-08 layouts | ~4 x 20 min | ~1.2 | amendment |
| 3 | DYN-08 `--stage all` (500 m padded, 12/6/12 winds, 192 test episodes, 5 model-error conditions, controls) | Belauri-scale synthetic deployment, H8a-d | 16 x 1.5 h | 10-15 (24 reserved) | 1, 2, DYN-08 registration |
| 4 | EXP-027b stage 1 (48 cases x 250/500/2000 m, common cases) | mesh adequacy with gate | 16 x 1.5 h | 11.6-23.5 (24 reserved) | registration |
| 5 | EXP-027b stage 2 (+24 cases), only if the 2 km gate passes | tighten intervals | 16 x 1 h | 5.7-11.7 | 4 gate passes |

Worst-case reservation ~61.6 core-h (leaves ~26); expected ~29-52. Jobs 3 and 4 run concurrently (32 cores total, well under 64); case-level
parallelism, response sets cached once. Bounded retries: 1 per job, only for infrastructure failure (never a changed seed/parameter), within the reservation.
Not proposed: 250 m DYN-08 inverse (+12), replicate design draws, GPU, the displacement/regret floor test (needs >=400 cases at 250 m, ~190 core-h).
