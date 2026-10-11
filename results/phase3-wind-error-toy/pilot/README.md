# EXP-019 wind-error toy (EXPLORATORY pilot)

Synthetic Phase-1 Domain (configs/reference.json), 3 local sources plus one external (inflow) basis. Raw rows: `raw/` (one CSV per truth wind
condition); derived: `tables/`; figures: `figures/` (stamped EXP-019); frozen choices: `freeze.json`; timing: `timing.json`.
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
then inflow).

| layout | rotation (deg) | signal change (sigma) | absorbed (sigma) | leftover (sigma) | chi2 excess / dof | relative amplitude shift |
|---|---|---|---|---|---|---|
| greedy-3 | 5 | 7.2 | 6.1 | 3.9 | 15 / 68 | -0.21, -0.07, -0.01, +0.59 |
| greedy-3 | 10 | 16.8 | 14.5 | 8.4 | 71 / 68 | -0.47, -0.01, -0.03, +1.13 |
| greedy-3 | 15 | 26.6 | 23.7 | 12.0 | 145 / 68 | -0.68, +0.09, -0.05, +1.35 |
| greedy-3 | 30 | 44.9 | 40.1 | 20.3 | 414 / 68 | -0.82, +0.18, -0.08, +0.49 |
| spread-3 | 5 | 12.9 | 12.9 | 0.8 | 1 / 68 | +0.30, +0.35, -52.94, +0.05 |
| spread-3 | 10 | 22.2 | 22.1 | 1.7 | 3 / 68 | +0.65, +0.70, -181.41, +0.12 |
| spread-3 | 15 | 28.7 | 28.6 | 2.8 | 8 / 68 | +0.99, +1.09, -395.08, +0.22 |
| spread-3 | 30 | 37.5 | 37.4 | 3.6 | 13 / 68 | +1.56, +0.89, -493.51, +0.28 |
| greedy-5 | 5 | 10.8 | 4.5 | 9.8 | 96 / 116 | -0.24, -0.05, +0.09, +0.60 |
| greedy-5 | 10 | 20.1 | 9.6 | 17.6 | 311 / 116 | -0.49, -0.12, +0.13, +1.23 |
| greedy-5 | 15 | 29.8 | 16.9 | 24.5 | 602 / 116 | -0.73, -0.18, +0.14, +1.66 |
| greedy-5 | 30 | 53.0 | 41.0 | 33.6 | 1131 / 116 | -0.96, -0.29, +0.13, +1.17 |
| spread-5 | 5 | 12.9 | 12.9 | 1.0 | 1 / 116 | +0.33, +0.24, -0.66, +0.01 |
| spread-5 | 10 | 22.2 | 22.0 | 2.2 | 5 / 116 | +0.73, +0.44, -51.95, +0.02 |
| spread-5 | 15 | 28.7 | 28.5 | 3.9 | 15 / 116 | +1.13, +0.62, -159.37, +0.04 |
| spread-5 | 30 | 37.5 | 37.2 | 5.0 | 25 / 116 | +1.75, +0.29, -188.73, +0.05 |
| greedy-8 | 5 | 17.4 | 8.6 | 15.1 | 229 / 188 | +0.01, -0.04, -0.08, +0.05 |
| greedy-8 | 10 | 35.6 | 22.0 | 28.0 | 786 / 188 | -0.02, -0.10, -0.18, +0.12 |
| greedy-8 | 15 | 53.2 | 37.2 | 38.1 | 1448 / 188 | -0.10, -0.15, -0.27, +0.19 |
| greedy-8 | 30 | 92.6 | 78.1 | 49.7 | 2474 / 188 | -0.36, -0.34, -0.39, +0.15 |
| spread-8 | 5 | 13.0 | 12.9 | 1.2 | 2 / 188 | +0.33, +0.26, -9.29, +0.01 |
| spread-8 | 10 | 22.3 | 22.1 | 3.1 | 9 / 188 | +0.72, +0.53, -92.41, +0.03 |
| spread-8 | 15 | 29.3 | 28.6 | 6.3 | 40 / 188 | +1.10, +0.86, -273.24, +0.06 |
| spread-8 | 30 | 51.3 | 39.5 | 32.7 | 1068 / 188 | +1.46, +2.18, -1101.59, +0.29 |

Singular values of the whitened, scaled nominal operator, then of the operator augmented with the wind-direction tangent column (+-5 deg central
difference), and the sine of the angle between that tangent and the span of the nominal columns (1 = wind direction fully separable from the
amplitudes, 0 = fully confounded):

| layout | singular values (nominal) | singular values (augmented) | sin(angle) |
|---|---|---|---|
| greedy-3 | 93.9, 37.6, 20.2, 9.21 | 93.9, 37.6, 20.2, 9.23, 0.493 | 0.418 |
| spread-3 | 58.2, 26, 16.3, 0.00821 | 58.2, 26.1, 16.4, 0.176, 0.00782 | 0.061 |
| greedy-5 | 118, 58.5, 29.7, 12.2 | 118, 58.5, 29.7, 12.3, 2.2 | 0.907 |
| spread-5 | 87.8, 27, 19.1, 0.01 | 87.8, 27.1, 19.2, 0.2, 0.01 | 0.073 |
| greedy-8 | 154, 92.6, 51.5, 28.9 | 154, 92.6, 51.5, 28.9, 3.08 | 0.956 |
| spread-8 | 112, 27.3, 19.7, 0.0106 | 112, 27.4, 19.7, 0.239, 0.0106 | 0.087 |

Posterior member weights of strategy C (mean over test cases and layouts): the weights concentrate when the data can tell winds apart; the effective
number of members 1/sum(w^2) of the 15-member declared ensemble measures how spread the posterior is.

| condition | vector error | mean max weight | mean effective members |
|---|---|---|---|
| d+000.0_s+000.0 | 0.000 | 0.85 | 1.48 |
| d+005.0_s+000.0 | 0.087 | 0.83 | 1.56 |
| d-005.0_s+000.0 | 0.087 | 0.83 | 1.55 |
| d+000.0_s+015.0 | 0.150 | 0.82 | 1.66 |
| d+000.0_s-015.0 | 0.150 | 0.86 | 1.48 |
| d+010.0_s+000.0 | 0.174 | 0.86 | 1.46 |
| d-010.0_s+000.0 | 0.174 | 0.86 | 1.54 |
| d+015.0_s+000.0 | 0.261 | 0.90 | 1.30 |
| d-015.0_s+000.0 | 0.261 | 0.86 | 1.45 |
| d-015.0_s-015.0 | 0.284 | 0.89 | 1.36 |
| d+015.0_s-015.0 | 0.284 | 0.90 | 1.28 |
| d+000.0_s-030.0 | 0.300 | 0.92 | 1.21 |
| d+000.0_s+030.0 | 0.300 | 0.90 | 1.35 |
| d-015.0_s+015.0 | 0.318 | 0.87 | 1.52 |
| d+015.0_s+015.0 | 0.318 | 0.84 | 1.51 |
| d+030.0_s+000.0 | 0.518 | 0.92 | 1.20 |
| d-030.0_s+000.0 | 0.518 | 0.93 | 1.32 |
