# Methodological decisions and open questions (draft before E00 protocol freeze)

**Status: proposed, NOT an E00-approved or frozen protocol.** This file is researcher context. E00 must record decisions, units, numerical tolerances and a human sign-off if scientific changes to the uploaded study are required.

| Dimension | Proposal from uploaded plan | To establish in E00 |
|---|---|---|
| Physics | 2-D depth-averaged passive tracer; wind, diffusion, removal | exact PDE, conserved units, source-height convention, boundary conditions |
| Solver | FiPy finite-volume on 16²–64² local verification | time stepping, reference solutions, CFL/Peclet regimes, numerical tolerances |
| Source and background | nonnegative low-dimensional source coefficients; background nuisance | normalization, source basis, rank deficiencies |
| Observations | candidate sensors, noisy scalar readings, correlated variants | sampling operator, instrument noise covariance, fixed admissible sites |
| Data splits | separate scenario, wind, and calendar runs | precise random seed IDs, train/validation/test separation and no leakage |
| Main endpoint | held-out one-step pre-assimilation forecast field RMSE | exact time origin and forecast horizon; scored before updates |
| Secondary metrics | source RMSE, analysis field error, population-weighted error, calibration | precise formulas and zero-denominator conventions |
| Compute | local-first, HPC after smoke and review, <=16 CPU cores total, <=32 GB per job, no GPU | full factorial design and core-hour cap before submitting |
| Human gates | no unauthorized scientific modifications | E00 protocol gate and subsequent change-control process |

The plan's author and scientific reviewer are responsible for the eventual choices; agents must not silently decide scientific questions on their behalf.
