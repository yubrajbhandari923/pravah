# EXP-019 wind-error toy (production)

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
then inflow). The refit is unconstrained least squares: the q >= 0 bound is NOT applied, so huge negative shifts (spread layouts leave one
direction, essentially source S3, almost unobserved: smallest singular value ~0.01) overstate what a bounded fit can absorb.

| layout | rotation (deg) | signal change (sigma) | absorbed (sigma) | leftover (sigma) | chi2 excess / dof | relative amplitude shift |
|---|---|---|---|---|---|---|
| greedy-3 | 5 | 18.1 | 17.9 | 3.0 | 9 / 68 | -0.25, +0.16, -0.07, +0.26 |
| greedy-3 | 10 | 35.2 | 34.6 | 6.7 | 45 / 68 | -0.42, +0.29, -0.19, +0.50 |
| greedy-3 | 15 | 48.8 | 47.7 | 10.5 | 110 / 68 | -0.56, +0.36, -0.31, +0.71 |
| greedy-3 | 30 | 66.9 | 64.0 | 19.3 | 374 / 68 | -0.88, +0.21, -0.51, +1.19 |
| spread-3 | 5 | 12.9 | 12.9 | 0.8 | 1 / 68 | +0.30, +0.35, -52.94, +0.05 |
| spread-3 | 10 | 22.2 | 22.1 | 1.7 | 3 / 68 | +0.65, +0.70, -181.41, +0.12 |
| spread-3 | 15 | 28.7 | 28.6 | 2.8 | 8 / 68 | +0.99, +1.09, -395.08, +0.22 |
| spread-3 | 30 | 37.5 | 37.4 | 3.6 | 13 / 68 | +1.56, +0.89, -493.51, +0.28 |
| greedy-5 | 5 | 18.9 | 18.5 | 4.0 | 16 / 116 | -0.07, +0.19, -0.13, +0.00 |
| greedy-5 | 10 | 37.1 | 36.3 | 7.8 | 61 / 116 | -0.14, +0.35, -0.27, +0.01 |
| greedy-5 | 15 | 52.0 | 50.7 | 11.6 | 135 / 116 | -0.21, +0.44, -0.40, +0.03 |
| greedy-5 | 30 | 76.3 | 73.5 | 20.7 | 429 / 116 | -0.42, +0.36, -0.61, +0.06 |
| spread-5 | 5 | 12.9 | 12.9 | 1.0 | 1 / 116 | +0.33, +0.24, -0.66, +0.01 |
| spread-5 | 10 | 22.2 | 22.0 | 2.2 | 5 / 116 | +0.73, +0.44, -51.95, +0.02 |
| spread-5 | 15 | 28.7 | 28.5 | 3.9 | 15 / 116 | +1.13, +0.62, -159.37, +0.04 |
| spread-5 | 30 | 37.5 | 37.2 | 5.0 | 25 / 116 | +1.75, +0.29, -188.73, +0.05 |
| greedy-8 | 5 | 24.3 | 16.2 | 18.1 | 327 / 188 | -0.09, +0.05, -0.10, +0.22 |
| greedy-8 | 10 | 47.4 | 31.9 | 35.2 | 1236 / 188 | -0.19, +0.08, -0.22, +0.46 |
| greedy-8 | 15 | 67.5 | 46.4 | 49.1 | 2407 / 188 | -0.30, +0.09, -0.35, +0.67 |
| greedy-8 | 30 | 110.2 | 79.2 | 76.7 | 5880 / 188 | -0.61, +0.01, -0.63, +1.17 |
| spread-8 | 5 | 13.0 | 12.9 | 1.2 | 2 / 188 | +0.33, +0.26, -9.29, +0.01 |
| spread-8 | 10 | 22.3 | 22.1 | 3.1 | 9 / 188 | +0.72, +0.53, -92.41, +0.03 |
| spread-8 | 15 | 29.3 | 28.6 | 6.3 | 40 / 188 | +1.10, +0.86, -273.24, +0.06 |
| spread-8 | 30 | 51.3 | 39.5 | 32.7 | 1068 / 188 | +1.46, +2.18, -1101.59, +0.29 |

Singular values of the whitened, scaled nominal operator, then of the operator augmented with the wind-direction tangent column (+-5 deg central
difference), and the sine of the angle between that tangent and the span of the nominal columns (1 = wind direction fully separable from the
amplitudes, 0 = fully confounded):

| layout | singular values (nominal) | singular values (augmented) | sin(angle) |
|---|---|---|---|
| greedy-3 | 101, 56.7, 24.5, 10.2 | 101, 56.7, 24.5, 10.2, 0.776 | 0.225 |
| spread-3 | 58.2, 26, 16.3, 0.00821 | 58.2, 26.1, 16.4, 0.176, 0.00782 | 0.061 |
| greedy-5 | 121, 77.8, 59.4, 37.4 | 121, 77.8, 59.5, 37.4, 1.05 | 0.293 |
| spread-5 | 87.8, 27, 19.1, 0.01 | 87.8, 27.1, 19.2, 0.2, 0.01 | 0.073 |
| greedy-8 | 134, 106, 78.3, 48 | 134, 106, 78.3, 48, 3.33 | 0.724 |
| spread-8 | 112, 27.3, 19.7, 0.0106 | 112, 27.4, 19.7, 0.239, 0.0106 | 0.087 |

Posterior member weights of strategy C (mean over test cases and layouts): the weights concentrate when the data can tell winds apart; the effective
number of members 1/sum(w^2) of the 15-member declared ensemble measures how spread the posterior is.

| condition | vector error | mean max weight | mean effective members |
|---|---|---|---|
| d+000.0_s+000.0 | 0.000 | 0.84 | 1.77 |
| d+005.0_s+000.0 | 0.087 | 0.82 | 1.79 |
| d-005.0_s+000.0 | 0.087 | 0.82 | 1.86 |
| d+000.0_s+015.0 | 0.150 | 0.82 | 1.78 |
| d+000.0_s-015.0 | 0.150 | 0.85 | 1.69 |
| d+010.0_s+000.0 | 0.174 | 0.87 | 1.58 |
| d-010.0_s+000.0 | 0.174 | 0.84 | 1.81 |
| d+015.0_s+000.0 | 0.261 | 0.85 | 1.56 |
| d-015.0_s+000.0 | 0.261 | 0.84 | 1.83 |
| d-015.0_s-015.0 | 0.284 | 0.85 | 1.77 |
| d+015.0_s-015.0 | 0.284 | 0.86 | 1.47 |
| d+000.0_s-030.0 | 0.300 | 0.90 | 1.40 |
| d+000.0_s+030.0 | 0.300 | 0.86 | 1.58 |
| d-015.0_s+015.0 | 0.318 | 0.82 | 1.79 |
| d+015.0_s+015.0 | 0.318 | 0.84 | 1.61 |
| d+030.0_s+000.0 | 0.518 | 0.92 | 1.24 |
| d-030.0_s+000.0 | 0.518 | 0.89 | 1.57 |
