# E16 64² generalization run: cluster plan (not submitted)

Status: documented 2026-11-14; **not submitted** (heavy 64² compute deferred by the user, 2026-11-13). Design:
`docs/research/generalization-ablations.md`. Code: `scripts/e16_generalization.py --sweep` (stage 1 field solves with
content-hashed caches, then stage 2 per scenario; reruns resume). Placeholders only: `$R` = PRAVAH_HPC_ROOT and `$H` = the
host, both from `settings.local.yaml`.

## Matrix

64² VanLeer dt 30 s (amendment A2), full protocol calendar, training winds `mixed`, held-out regimes as in E07/E08,
K ∈ {2, 4}, seeds 0–19 (n = 20 clusters per cell), ks 2/4/8/16, strategies random (10 networks)/coverage/source_near/dopt,
estimators map24 and map_grow, white noise, R = 5 paired draws.

| cell | change | new solves | note |
|---|---|---|---|
| nominal | – | 0 | identical cache names to the E07/E08 sweep fields (`test_sweep_nominal_reuses_e07_fields`) |
| bg_hi | c_bg 20 µg m⁻³ | 0 | reuses nominal fields, truth shifted (exact; `test_background_share_is_exact`) |
| diff_hi | K 1000 m² s⁻¹ | 157 | at 64² VanLeer K_eff ≈ K, so this is a 20× diffusivity change (only ≈ 1.5× at 32² upwind) |
| loss_hi | λ = 1/(2 h) | 157 | |
| src_wide | σ_s = 800 m | 157 | |
| src_sparse | min separation 3000 m | 157 | layouts change; strengths and winds paired |

`diff_lo` and `diff_lo×loss_hi` are not run at 64²: every 64² cell already uses VanLeer, so they coincide with nominal
and loss_hi. The **resolution contrast** (32² vs 64² with the physical parameters, hence the physical Pe, Da and σ_s/L,
fixed and the scheme fixed) is 32² `diff_lo` (pilot) against 64² nominal; the 32² upwind nominal against 64² nominal
mixes resolution and scheme and is reported only as context.

## Cost (projected)

- Per 64² solve (1 truth or 1 unit response over the full calendar): **≈ 4.4 CPU-h** under 64 concurrent workers
  (E07 sweep: 156 solves + stage 2 in 12.05 h on 64 CPUs = 4.9 CPU-h per solve, upper bound; E13 plan 4.4; A2 serial
  timing 7.8 s per simulated hour → 3.3 CPU-h).
- New solves: 628 (4 cells × 157; duplicates removed by content hash) → **≈ 2 760 CPU-h ≈ 43 h on 64 CPUs**
  (range 2 100–3 100 CPU-h, 33–48 h).
- Stage 2 (evaluation, 6 cells × 40 scenarios = 240 cell-scenarios): pilot ≈ 1–2 s per scenario at 32²; 64² has
  16× the cells and 5.6× the hours → ≲ 3 min per scenario → ≲ 12 CPU-h, negligible.
- Memory: ≤ 1 GB per solver worker; stage 2 holds ≤ 0.3 GB of fields per scenario (K = 4). `--mem=128G` as for E07/E08.
- Disk: ≈ 50 MB per field → ≈ 31 GB new under `$R/results/e16-sweep-<rev>/fields`.

Because 43 h is close to the 2-day wall-time limit, the recommended plan is **two sequential submissions of the same
resumable command**: first `--n-seeds 10` (312 new solves, ≈ 1 380 CPU-h ≈ 22 h), then the full 20 seeds (the first
half is cached). This keeps one 64-CPU job at a time and gives a usable n = 10 result if the second half is not run.
**[LEAD REVIEW]**: stopping after the 10-seed stage is allowed only if decided before looking at its results (design
note §6); otherwise the preregistered n is 20.

## Commands

```bash
REV=$(git rev-parse --short HEAD)       # a commit containing scripts/e16_generalization.py and this note
OUT=$R/results/e16-sweep-$REV
git archive --format=tar "$REV" | ssh "$H" "mkdir -p $R/code/snapshots/$REV && tar -x -C $R/code/snapshots/$REV"
ssh "$H" "cd $R/code/snapshots/$REV && UV_PROJECT_ENVIRONMENT=$R/envs/main ~/.local/bin/uv sync --frozen"
# 0) reuse the E07/E08 nominal fields (the E08 cache holds K 2/4/8, seeds 0-19, mixed; symlinks are followed)
ssh "$H" "mkdir -p $OUT/fields && cd $OUT/fields && for f in $R/results/e08-sweep-cd7afd9/fields/*; do ln -sf \"\$f\" .; done"
# 1) reuse check (1 CPU, minutes): must print e16sweep_nominal absent (0) and 157 for each other cell (78 with --n-seeds 10)
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-gen -c 1 --mem=4G -t 00:20:00 --export=ALL,PRAVAH_HPC_ROOT=$R \
  -o $R/logs/%x-%j.out scripts/hpc/run.sbatch $R/envs/main/bin/python scripts/e16_generalization.py --sweep \
  --out $OUT --dry-run"
# 2) first half (check squeue first; one 64-CPU job at a time)
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-gen -c 64 --mem=128G -t 2-00:00:00 --export=ALL,PRAVAH_HPC_ROOT=$R \
  -o $R/logs/%x-%j.out scripts/hpc/run.sbatch $R/envs/main/bin/python scripts/e16_generalization.py --sweep \
  --out $OUT --workers 64 --n-seeds 10"
# 3) after 2) has finished: the full matrix (resumes; same command without --n-seeds)
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-gen -c 64 --mem=128G -t 2-00:00:00 --export=ALL,PRAVAH_HPC_ROOT=$R \
  -o $R/logs/%x-%j.out scripts/hpc/run.sbatch $R/envs/main/bin/python scripts/e16_generalization.py --sweep \
  --out $OUT --workers 64"
# 4) copy $OUT/e16-sweep-{rows,compute,solves,cells}.csv and e16-sweep-meta.json back, then report locally:
uv run python scripts/e16_generalization.py --sweep --report --out <local copy dir>
```

If the dry run reports nominal solves > 0, the E08 cache was not linked: stop (157 extra solves ≈ 690 CPU-h otherwise).
A time-out is resumed by resubmitting the same command (atomic caches; finished solves and scenarios are skipped).
`meta.git_rev` is empty in a `git archive` snapshot; record `$REV` and the job id in `reports/runs/runs.jsonl`.

## Pilot timings used (32², local)

From `reports/quarto/tables/e16-local-solves.csv` / `-compute.csv` (6 upwind cells, 195 solves, 6 workers on a shared
16-core workstation): 46–72 s per 276-h upwind solve (median 62 s; 3.2 CPU-h total), stage 2 ≈ 0.45 s per scenario.
VanLeer dt 75 s solves took 290–330 s under load (≈ 1.1 s per simulated hour, dt 60 s ≈ 1.4). Scaling to 64²
(4× cells, ~2× steps relative to 32² VanLeer dt 60, 5.6× hours) gives ≈ 3–4 CPU-h per solve, consistent with the
cluster-measured 4.4–4.9 used above; the cluster figure is the one used for the projection.

What the 64² run is for (pilot outcome): C2, C4 held in every pilot cell; C1 weakened under `src_wide`, C3 under
`diff_hi`, C5 conditional under `bg_hi`. By design note §6 those three cells are run once at n = 20 and reported as
unresolved if still borderline. `diff_hi` matters most at 64², where K_eff changes 20× instead of ≈ 1.5×.
