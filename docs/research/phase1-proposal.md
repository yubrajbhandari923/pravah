# PRAVAH Phase 1 — proposal (starter context; not a completed protocol)

## Research question
To what extent can sparse sensor networks reconstruct a pollutant concentration field and population exposure when winds and emission sources are uncertain, and what sensor placement maximizes reliability?

## Model and scientific distinction
Start with a controlled two-dimensional, depth-averaged passive tracer, not directly observed PM2.5. FiPy finite-volume forward transport provides simulated concentration fields. Inversion infers a parameterized nonnegative emission source and possibly background from simulated sensor readings. Sensor designs are optimized for potentially different goals: source identification, field reconstruction and future forecast. Separate perfect-model benchmarks from uncertain wind/removal/boundary-condition cases.

## Sequence and stakes
- E00 Protocol: freeze metrics, seeds, splits, numerical oracle conditions and primary pre-assimilation forecast endpoint.
- E01 Forward verification: diffusion/transport/decay/forcing analytical or manufactured cases; conservation, convergence, positivity and units.
- E02 Benchmark: hidden synthetic truth, stable candidate sensors, noise, wind scenarios and train/validation/test leakage controls.
- E03 Static inverse: response matrix and NNLS/weighted inverse; identifiable and underdetermined examples.
- E04–E05: footprints, Fisher information and the value of time history.
- E06–E08: placement objectives, cross-objective comparisons and conditional sensor sufficiency.
- E09–E11: causal Kalman/augmented-state assimilation and forecast-informed design.
- E12–E16: uncertain meteorology, structural mismatch, calibration, robust design and held-out generalization.
- E17–E18: conditional regional application / real observational check ONLY with genuinely available licensed data.
- E19–E20: evidence-grounded Quarto synthesis, independent audit and researcher final sign-off.

## Pre-analysis commitments from the uploaded plan
The proposed primary outcome is a one-step-ahead **pre-assimilation forecast RMSE** on held-out truth at a declared forecast issue time. Secondary outcomes include source recovery, post-analysis field error, uncertainty calibration and optionally population-weighted field error. Candidate sensor counts are 2/4/8/16/32, source numbers 2/4/8, with snapshot/6h/24h/7d windows and paired seeds, subject to explicit preregistration and compute-budget review in E00.

## Constraints
No extrapolation from synthetic results to actual air quality or public health validity; no fabricating regional observational performance; no using test truth for model selection or sensor placement; no unreviewed scientific changes. Plan tasks passing their automated checks is engineering progress, not a scientific finding.
