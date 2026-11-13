# Green (2006): technical reading notes for PRAVAH agents

**Source:** Kary Green, *Optimal Sensor Placement for Parameter Identification*, Rice University master's thesis, July 2006. The user supplied a 63-page scanned PDF in `docs/papers/Green_2006_Optimal_Sensor_Placement.pdf`. This PDF has no extractable text in ordinary PDF parsers; a text-only agent may not be able to read it. These notes are a navigation aid, NOT a substitute for verifying original pages and figures.

## What the thesis actually studies

It estimates unknown parameters in a dynamical system (after spatial discretization of a PDE) from sensor readings, then chooses the sensor locations in advance to improve precision of those parameter estimates. The illustrative 1D advection-diffusion problem estimates diffusivity and advection coefficient; it does **not** recover an unconstrained concentration field from many unknown sources, nor design an atmospheric network for real PM2.5.

## Page pointers (PDF page number, not printed thesis numbering)

- PDF pp. 10–15: problem setup and prior sensor-placement literature; parameter identification vs state estimation.
- PDF pp. 17–20: system derivation and finite-element semidiscretization of parameterized transport PDE.
- PDF pp. 21–29: weighted least-squares inverse problem, parameter sensitivities, estimator covariance and Fisher information.
- PDF pp. 29–34: 1D time-dependent advection-diffusion parameter-estimation test with time series and perturbations.
- PDF pp. 34–46: design objectives, source-sensor sensitivity matrices, approximate design weights and D-opt algorithm.
- PDF pp. 47–54: numerical sensor-placement comparison and confidence ellipse figures.
- PDF pp. 56–59: Appendix A (maximum likelihood, weighted least squares and Fisher information).

## Reusable mathematics

For sensor observation map `y(theta,S)` and Gaussian observation noise covariance `R`, the sensitivity/Jacobian `G = d y / d theta` produces the local Fisher information matrix `M(S) = G(S)^T R^-1 G(S)` when covariance does not depend on theta; for independent times, stacked Jacobians add information. With nonsingular `M`, the local covariance approximation is `M^-1`. D-optimality maximizes `log det M` / minimizes volume of the local parameter confidence ellipsoid. A-optimality minimizes `trace(M^-1)` and E-type criteria evaluate the worst informed direction. A singular or nearly singular `M` cannot safely be treated with a naive inverse/determinant.

The thesis also formulates a weighted approximate experimental design over discrete candidate sensor locations, with nonnegative weights summing to 1, and compares to equally spaced and center-biased placements.

## A reproducible *illustration*, not a universal result

On the author's 1D example, the reported optimized locations were approximately x=0.025 and 0.975 on [0,1]. The 95% confidence ellipse area table (PDF p.54, printed p.45) gives center placement ~1.91e-3, equally spaced placement ~4.66e-4 and optimized placement ~2.97e-4, hence ~36% lower than equal spacing. This result is specific to that PDE, unknown parameter vector, noise and design objective; no general rule says to deploy pollution sensors on boundaries.

## What PRAVAH should **not** conflate

1. Parameter identifiability (the target of Green's 2006 design) vs reconstructed full-field accuracy vs future forecast skill.
2. A source-receptor footprint is a model-derived sensitivity, not an independently measured spatial pollutant field.
3. Using 20 time samples may improve information about fixed parameters, but not if every time sample also introduces completely unrestricted independent unknown source strengths.
4. Green's asymptotic FIM confidence interval assumes the selected parameterization and noise model; under wind/source misspecification its coverage may be poor.

Agents should cite Green 2006 with specific section/page when adapting its formulas and mark every 2D/source-strength extension as a PRAVAH adaptation.
