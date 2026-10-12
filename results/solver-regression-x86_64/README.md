# x86_64 regression baseline (cluster)

Captured on the Duke cluster (`plp-drcc-login2`, compute node, AMD EPYC 7513) by
`scripts/hpc_regression_selfcheck.sh` under Research OS RUN-0053 / Slurm job 201814,
with the locked environment (Python 3.12.12, numpy 2.5.3, scipy 1.18.1, FiPy 4.0.3).

## Why a second baseline exists

See decision **D-005**. `results/solver-regression/before` was captured on Apple M2
(arm64/NEON). The cluster is x86_64 with OpenBLAS `DYNAMIC_ARCH` (AVX-512 capable).
The two architectures reassociate floating-point sums differently, so the arm64
baseline compares as `21 files compared, 18 differ` on the cluster even though the
code is unchanged and every difference is at floating-point-reassociation scale
(largest physical quantity: 5.0e-08 relative; `max_concentration`: 1 ulp).

Evidence that this is architectural and not a regression:

- two captures on the same cluster node: `21 files compared, 0 differ` (deterministic)
- Intel Xeon Silver 4310 and AMD EPYC 7513 produce bit-identical values to each other
- a local venv at the lockfile's pinned numpy 2.5.3 reproduces `0 differ` on arm64,
  so dependency drift is not the cause

## Use

On the cluster, compare against this baseline:

    python scripts/compare_runs.py results/solver-regression-x86_64/before <new-capture>

Expect `21 files compared, 0 differ`. On a laptop (arm64) keep using
`results/solver-regression/before`. The gate is binding within an architecture;
cross-architecture bit-identity is not achievable and is not required.
