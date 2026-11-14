# E17 regional case: 64² cluster plan (not submitted)

Status: plan only, 2026-11-14. Heavy 64² sweeps are deferred (user, 2026-11-13); submit only after the lead approves.
Design: `docs/research/regional-case-design.md`; driver `scripts/e17_regional.py --sweep`. Label: **synthetic tracer under
ERA5 winds, not validation**. Placeholders only: `<hpc-host>`, `$PRAVAH_HPC_ROOT` (= `$R`; real values in the untracked
`settings.local.yaml`).

## What runs

- 3 cases (coolA Nov–Dec 2021, coolB Jan–Feb 2022, monsoon: Nov train / Jul 2021 test), K ∈ {2, 4}, seeds 0–9 →
  60 scenarios, clusters (K, seed) = 20 per case. 64², split VanLeer, dt 240 s (per-axis Courant ≤ 0.74), K = 500 m²/s,
  48 h spin-up, 1440 (coolB 1416) model hours + spin-up per solve; sensors k ∈ {4, 8, 16}; R = 5 noise draws;
  estimators map24 and map_grow; wrong-wind specs (point, dir30) on coolA.
- Stage 1: **210 solves** (60 truths + 150 unit responses after content-hash de-duplication; counted with
  `field_tasks`). Stage 2: 60 scenario evaluations. Both in the same process pool; atomic per-solve caches in
  `OUT/fields`, so a resubmitted job resumes.

## Projected cost (from local measurements; not measured on the cluster)

| item | measurement | projection |
|---|---|---|
| one 64² VanLeer model hour, dt 240 s | 0.98 s (local, single thread, shared machine under load; 140 MB RSS) | — |
| one solve (~1488 model hours) | pilot 32² upwind solve (576 h): 131 s mean over 83 solves on 6 workers | ≈ 24 min |
| stage 1 | 210 solves | ≈ 85 CPU-h; 4 waves on 64 workers ≈ 1.7 h wall |
| stage 2 | pilot: 18 scenarios in < 1 min on 6 workers after the solves (32², 528 h, R = 3) | ×20–40 at 64² (4× cells, 2.7× hours, R = 5, larger k) ⇒ ≈ 10 CPU-h, ≈ 15 min wall |
| memory | per worker ≤ ~1 GB in stage 2 (3 specs × K = 4 × 1440 × 64² float64 ≈ 0.57 GB + truth) | `--mem=96G` |

Total ≈ 95 CPU-h, ≈ 2 h wall. Request: `-c 64 --mem=96G -t 06:00:00` (≈ 3× the projection). If stage 2 memory is the issue, rerun the same command
with `--workers 32` (stage 1 is cached).

## Commands (from a committed revision containing the driver and this note)

```bash
H=<hpc-host>; R=$PRAVAH_HPC_ROOT
REV=$(git rev-parse --short HEAD)
git archive --format=tar "$REV" | ssh "$H" "mkdir -p $R/code/snapshots/$REV && tar -x -C $R/code/snapshots/$REV"
ssh "$H" "cd $R/code/snapshots/$REV && UV_PROJECT_ENVIRONMENT=$R/envs/main ~/.local/bin/uv sync --frozen"
# raw inputs (2 MB, immutable): copy once, then verify against docs/research/data/e17-manifest.json
rsync -a .cache/e17/raw/ "$H:$R/data/e17/raw/"
ssh "$H" "cd $R/data/e17/raw && sha256sum era5_sl_belauri_*.nc worldpop/*.tif"   # compare with the manifest
# dry run: lists the work only (no solves, seconds): expect '60 scenarios, 210 solves, 210 to do'
ssh "$H" "cd $R/code/snapshots/$REV && PRAVAH_E17_RAW=$R/data/e17/raw $R/envs/main/bin/python scripts/e17_regional.py --sweep --out $R/results/e17-sweep-$REV --dry-run"
# the run (check squeue first; one job, <= 64 CPUs)
ssh "$H" "cd $R/code/snapshots/$REV && sbatch -J pv-rg -c 64 --mem=96G -t 06:00:00 \
  --export=ALL,PRAVAH_HPC_ROOT=$R,PRAVAH_E17_RAW=$R/data/e17/raw -o $R/logs/%x-%j.out scripts/hpc/run.sbatch \
  $R/envs/main/bin/python scripts/e17_regional.py --sweep --out $R/results/e17-sweep-$REV --workers 64"
# report locally after copying OUT/e17-sweep-*.csv/json into a local dir D:
uv run python scripts/e17_regional.py --sweep --report --out D
```

Before submitting: the driver prints `N scenarios, 210 solves, M to do` first; a resubmission should show M < 210.
Record the job id, revision and result path in `reports/runs/runs.jsonl` (lead).

## Risks

- Runtime per model hour on the cluster nodes is not measured for dt 240 s; the E07 64² VanLeer runs (dt 30 s) are the
  only cluster reference. A 1-CPU 24-model-hour smoke test (minutes) before the full job would remove the guess.
- ERA5 winds are weak (box-mean median 1.2 m/s); designs may tie. That is a result, not a failure.
