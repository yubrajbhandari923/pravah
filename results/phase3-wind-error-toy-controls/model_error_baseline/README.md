# EXP-019 wind-error toy (production)

Synthetic Phase-1 Domain (configs/reference.json), 3 local sources plus one external (inflow) basis. Raw rows: `raw/` (one CSV per truth wind
condition); derived: `tables/`; figures: `figures/` (stamped EXP-023); frozen choices: `freeze.json`; timing: `timing.json`.
Vector error |du|/|u| of a rotation by dtheta at constant speed is 2 sin(dtheta/2) (5/10/15/30 deg = 0.087/0.174/0.261/0.518); a speed error e
is |e|; a joint error is sqrt(1 + f^2 - 2 f cos(dtheta)).

## Identifiability: when amplitude and wind compensate

A rotation of the wind moves a plume centreline at downwind distance L sideways by about L*dtheta, i.e. it acts like a displaced source, and
changes the travel time and dilution, i.e. like a rescaled source. With source positions known and only amplitudes free, the only way the
nominal-wind model (strategy B) can imitate the rotated truth is to change the amplitudes (and the inflow amplitude). The rank/effective-rank/
condition/confounded-pair flags only look at the columns of the NOMINAL operator: they do not depend on the data, and therefore cannot see a wrong
wind. chi2/dof with known sigma can, once the part of the signal change that amplitudes cannot absorb exceeds the noise.

Numerical demonstration (noise-free, from the cached response bases, base amplitudes, frozen layouts). For each layout and rotation: the signal change
(A(phi) - A(0)) theta in units of sigma, the part absorbed by refitting amplitudes on the nominal operator, the leftover (its square is the chi2
excess a noiseless fit would show; the expected noise chi2 is about dof), and the amplitude shift that does the absorbing (relative; emission 1..3,
then inflow). The refit is unconstrained least squares: the q >= 0 bound is NOT applied, so huge negative shifts (spread layouts leave one
direction, essentially source S3, almost unobserved: smallest singular value ~0.01) overstate what a bounded fit can absorb.

| layout | rotation (deg) | signal change (sigma) | absorbed (sigma) | leftover (sigma) | chi2 excess / dof | relative amplitude shift |
|---|---|---|---|---|---|---|
| greedy-3 | 10 | 27.4 | 24.0 | 13.3 | 176 / 68 | -0.18, -0.11, -0.29, +0.52 |
| greedy-3 | 30 | 64.4 | 59.8 | 24.0 | 578 / 68 | -0.86, -0.07, -0.55, +1.55 |
| spread-3 | 10 | 22.2 | 22.1 | 1.7 | 3 / 68 | +0.65, +0.70, -181.41, +0.12 |
| spread-3 | 30 | 37.5 | 37.4 | 3.6 | 13 / 68 | +1.56, +0.89, -493.51, +0.28 |
| greedy-5 | 10 | 28.1 | 24.5 | 13.8 | 190 / 116 | -0.09, -0.05, -0.29, +0.06 |
| greedy-5 | 30 | 75.6 | 70.7 | 26.8 | 717 / 116 | -0.63, +0.12, -0.51, +0.18 |
| spread-5 | 10 | 22.2 | 22.0 | 2.2 | 5 / 116 | +0.73, +0.44, -51.95, +0.02 |
| spread-5 | 30 | 37.5 | 37.2 | 5.0 | 25 / 116 | +1.75, +0.29, -188.73, +0.05 |
| greedy-8 | 10 | 41.3 | 36.6 | 19.2 | 367 / 188 | -0.10, -0.02, -0.36, -0.00 |
| greedy-8 | 30 | 99.3 | 90.9 | 39.8 | 1585 / 188 | -0.65, -0.08, -0.63, +0.27 |
| spread-8 | 10 | 22.3 | 22.1 | 3.1 | 9 / 188 | +0.72, +0.53, -92.41, +0.03 |
| spread-8 | 30 | 51.3 | 39.5 | 32.7 | 1068 / 188 | +1.46, +2.18, -1101.59, +0.29 |

Singular values of the whitened, scaled nominal operator, then of the operator augmented with the wind-direction tangent column (+-5 deg central
difference), and the sine of the angle between that tangent and the span of the nominal columns (1 = wind direction fully separable from the
amplitudes, 0 = fully confounded):

| layout | singular values (nominal) | singular values (augmented) | sin(angle) |
|---|---|---|---|
| greedy-3 | 101, 65.9, 25.1, 11.2 | 101, 65.9, 25.1, 11.2, 0.693 | 0.313 |
| spread-3 | 58.2, 26, 16.3, 0.00821 | 58.2, 26.1, 16.4, 0.176, 0.00782 | 0.061 |
| greedy-5 | 112, 73.3, 52.7, 37.6 | 112, 73.3, 52.7, 37.6, 0.818 | 0.366 |
| spread-5 | 87.8, 27, 19.1, 0.01 | 87.8, 27.1, 19.2, 0.2, 0.01 | 0.073 |
| greedy-8 | 134, 108, 67.2, 46.8 | 134, 108, 67.2, 46.8, 1.93 | 0.542 |
| spread-8 | 112, 27.3, 19.7, 0.0106 | 112, 27.4, 19.7, 0.239, 0.0106 | 0.087 |

Posterior member weights of strategy C (mean over test cases and layouts): the weights concentrate when the data can tell winds apart; the effective
number of members 1/sum(w^2) of the 15-member declared ensemble measures how spread the posterior is.

| condition | vector error | mean max weight | mean effective members |
|---|---|---|---|
| d+000.0_s+000.0 | 0.000 | 0.87 | 1.53 |
| d+010.0_s+000.0 | 0.174 | 0.89 | 1.42 |
| d-010.0_s+000.0 | 0.174 | 0.87 | 1.57 |
| d+030.0_s+000.0 | 0.518 | 0.92 | 1.25 |
| d-030.0_s+000.0 | 0.518 | 0.91 | 1.48 |
