# PRAVAH E03 — source-response operator and weighted nonnegative inverse

**Task:** T-013 · **Experiment:** EXP-004 · **Author:** claude-balanced / claude-sonnet-5-5 · **Date:** 2026-11-10
**Status:** DRAFT derivation plus a smoke-scale check. Not frozen, not human-approved, not independently reviewed. It reports no scientific finding.
**Evidence:** `scripts/e03_derivation_checks.py`, run RUN-0012 of EXP-004 (code commit `d053ff7`, the script itself was
untracked at run time and is captured in the run's snapshot ref; backend local, host RAD-88300596, 102 s). RUN-0011 ran the
same script earlier and **failed** at a sensor-selection bug in the script (two chosen sites closer than the minimum
separation) after the first six metrics; its output was discarded and the script fixed. Every number quoted below is a
metric of RUN-0012 (names in `code font`); none is typed from memory.
**Sources read:** `forward.py`, `benchmarks.py`, `observations.py`, `pde-and-discretization.md`, `truth-design.md`,
`preregistered-protocol.md` (§4, identifiability lines), `green-2006-reading-notes.md`, `gate-02-benchmark.md`.
**Not read:** Green (2006) and Bhardwaj & Subramanian (2026) themselves. The scanned Green PDF has no extractable text and the
Bhardwaj item is an unreviewed preprint. Chapter/page pointers below come from the project's reading notes, are unverified,
and every use of them is marked as a PRAVAH adaptation.

This is a synthetic 2-D passive tracer. Nothing here is evidence about ambient PM2.5.

## 1. What is being inverted

Unknown: source strengths a ∈ ℝ^K_{≥0} (total rates, µg s⁻¹) at **known** locations, and optionally the uniform
background b (µg m⁻³). Data: noisy hourly readings at chosen candidate sites. The estimator receives only
`Scenario.view(...)`: readings, nominal winds, known source locations, noise σ and nominal model parameters.

## 2. Forward operator from the actual transient solver

### 2.1 Anomaly form (why the background drops out)

The solver advances `c_t + u·∇c = ∇·(K∇c) − λ(c − c_bg) + S`, with Dirichlet `c = c_bg` on inflow faces and free
outflow. Write e = c − c_bg. The inflow value, the removal term and the initial value of a uniform field all cancel
against c_bg, so e solves the same equation with `c_bg = 0`:

```
e_t + u·∇e = ∇·(K∇e) − λ e + S,      e = 0 on inflow faces,      S = (1/H) Σ_k a_k g_k(t) φ_k(x)
```

This is exact in the continuum and discretely (oracle v of `pde-and-discretization.md`: uniform c_bg is a discrete steady
state). `benchmarks.generate_scenario` already solves this anomaly form and adds c_bg afterwards, because the FiPy linear
solve at c_bg = 5 dropped sources of this size (gate-02). The response matrix must be built the same way.
Check: a raw c_bg = 5 run with the source amplified ×10⁴ matches 5 + anomaly to relative `affine_background_rel_err`.

### 2.2 One hour, one wind

Let L_h be the spatial operator for hour h's uniform wind (central diffusion, upwind convection, removal, inflow
Dirichlet and outflow loss), M_h = I/Δt − L_h, Δt = 300 s. Backward Euler gives, per sub-step,

```
e_{n+1} = A_h e_n + P_h S,        A_h = M_h⁻¹/Δt,   P_h = M_h⁻¹
```

and 12 sub-steps per hour with the same wind give e^{(h)} = Φ_h e^{(h−1)} + Ψ_h S, with
Φ_h = A_h¹² and Ψ_h = (Σ_{j=0}^{11} A_h^j) P_h. Both are fixed linear maps of the wind u_h, K, λ and the grid.
The wind is piecewise constant per hour; reading hour h is the state at the **end** of hour h under wind u_h
(hour 0 is the first hour after spin-up).

### 2.3 Unrolled response, release history and transport lag

The release starts at the start of spin-up (hour −T_s, e = 0; T_s = 48 h at the preregistered scale, 12 h in the smoke
runs). With g_k ≡ 1 (the only temporal factor implemented):

```
e^{(h)} = Σ_k a_k  Σ_{τ=−T_s}^{h}  [ Φ_h Φ_{h−1} ⋯ Φ_{τ+1} ]  Ψ_τ φ_k / H
```

Reading i at hour h: `y_{i,h} = H_i e^{(h)} + c_bg + ε_{i,h}`, with H_i the bilinear sampling operator of
`observations.bilinear_sample` (exact at cell centres, convex weights ≥ 0). Hence

```
y = G a + b·1 + ε,       G[(i,h), k] = H_i Σ_τ [Φ_h ⋯ Φ_{τ+1}] Ψ_τ φ_k / H     [µg m⁻³ per µg s⁻¹]
```

- The product of Φ's is the **transport lag**: mass released at τ reaches site i at h only through every intermediate wind.
  An instantaneous footprint would keep only the τ = h term and would be wrong whenever the wind changes or the plume
  has not arrived. A time-varying release g_k(t) keeps linearity and replaces Ψ_τ φ_k by g_k(τ) Ψ_τ φ_k; G then depends
  on the assumed g_k. The diurnal variant is **not** implemented in the benchmark, so it is not covered here.
- The product of Φ's also contains the start-up transient: if spin-up is short, G includes a not-yet-steady plume.
  The residual of the slowest mode is of order e^{−λT_s} (≈ 1.8 % at 48 h for λ = 1/12 h⁻¹). It is part of the truth and
  therefore part of G, as long as G is built with the same spin-up and wind replay (the replay of the first spin-up
  hours of the train stream is an unreviewed [decision], W5 of gate-02).

### 2.4 Linearity: when it holds and when it does not

- **Linear in a** (exactly, for the discrete scheme): superposition and homogeneity, for fixed winds, K, λ, grid,
  source locations and shapes, release history and spin-up. The cost is K forward solves (one per source, unit amplitude)
  over spin-up + window.
- **Affine in b** (uniform background): enter b as the all-ones column, so θ = (a, b), G̃ = [G, 1]. In the base case
  c_bg = 5 is known and is subtracted. In the unknown-background variant b ~ N(5, 1²) is estimated; the all-ones column is
  nearly collinear with the plume columns when only a few far-field sensors see no plume (and exactly so at benchmark
  SNR, §5).
- **Not linear in**: source locations, widths, wind, K, λ, mixing depth. These make G a nonlinear function of the nuisance
  parameters. This task treats them as known (perfect-model truth).
- **Non-negativity of G.** A_h and P_h are inverse M-matrices (≥ 0), φ_k ≥ 0 and the sampling weights are convex, so every
  entry of G is ≥ 0 and a ≥ 0 gives non-negative anomalies. This follows from the M-matrix argument of
  `pde-and-discretization.md` §5.5 and was **not** separately tested here.

### 2.5 Sensitivity propagation and normalisation

- ∂y/∂a = G exactly (no linearisation). ∂y/∂b = 1.
- For a nuisance parameter p ∈ {u, K, λ} (needed for mismatch studies, **not implemented**), differentiate the recursion:
  δ^{(h)} = Φ_h δ^{(h−1)} + (∂Φ_h/∂p) e^{(h−1)} + (∂Ψ_h/∂p) S, with δ = ∂e/∂p. The same sparse factorisations apply.
- Normalisation fixed by the code: φ_k is discretely normalised (Σ φ_k ΔA = 1 after truncation and wall clipping), so a_k is
  the total rate in µg s⁻¹ on any grid. For conditioning, use ã_k = a_k / a_ref (a_ref = 100 µg s⁻¹) so columns are in
  µg m⁻³ per reference source, and whiten rows by the noise (§3). Rank decisions use the relative tolerance 1e-8 of the
  protocol (identifiability lines).

## 3. Noise covariance and the estimators

ε ~ N(0, R). From `observations.noise_field`: sites independent; per site either white (R = σ² I) or stationary AR(1) in
hours (R_site = σ² [ρ^{|h−h'|}], ρ = 0.8), with σ one scalar per scenario (0.05 × median train truth) or per site.
Let W = R^{-1/2} (a Cholesky solve for AR(1)).

**Primary baseline — weighted NNLS** (PRAVAH adaptation of the weighted least-squares/maximum-likelihood setting of Green
2006, ch. 2.2 and App. A, per the reading notes; page numbers unverified):

```
â = argmin_{a ≥ 0}  ½ ‖ W (y − c_bg 1 − G a) ‖²
```

Solved with `scipy.optimize.nnls` on the whitened system (WG, W(y − c_bg 1)).

**Regularised variant — ridge NNLS:**

```
â_λ = argmin_{a ≥ 0}  ½ ‖ W (y − c_bg 1 − G a) ‖² + ½ λ_r ‖ a ‖²
```

as an augmented NNLS (stack √λ_r I under WG and zeros under the data). λ_r must be fixed a priori or chosen on
**validation** data of the train/validation regimes only, never on test. In RUN-0012 it is fixed a priori at
λ_r = 10⁻³ ‖WG‖₂², chosen before seeing any result, and **not** tuned.
Unknown background: append the 1-column with a sign-free (or ≥ 0) coefficient.

**Information and uncertainty (Green 2006 adaptation).** M = GᵀR⁻¹G; if M is nonsingular the local covariance of the
unconstrained estimator is M⁻¹. Singular or near-singular M must not be inverted naively. For the NNLS estimator this is only an
approximation, because active constraints a_k = 0 change the distribution (truncated, non-Gaussian).
Bhardwaj & Subramanian (2026) is the planned reference for low-dimensional source fingerprints (columns of G as
source signatures); it was **not opened**, so no claim here depends on it.

## 4. Inferred sources versus reconstructed field

Two different targets, to be scored separately (as the protocol requires):

- **Source recovery:** â versus a (relative RMSE, bias). Needs G of full column rank on the network.
- **Field reconstruction:** ĉ = c_bg + F â, where F[:, k] is the unit-response field of source k on the whole grid (not just at sensors).
  The field error is ‖F(â − a)‖. A null direction n of the sensor operator (G n = 0) leaves the readings unchanged, but F n ≠ 0
  in general, so a non-identifiable source set does **not** give a determined field either; it gives a family of fields
  that agree at the sensors. A small field error at the sensors does not imply a small error elsewhere.

## 5. Results of the check (RUN-0012; smoke scale only)

Setting: 32² grid, spin-up 12 h, 24 h window, hand-placed examples from `scripts/e02_examples.py`, perfect-model truth. All numbers are
metrics of RUN-0012. This is **not** the preregistered 64² / 48 h-spin-up setting.

| Check | Metric | Value | Reading |
|---|---|---|---|
| Superposition vs hidden truth (2 sources, all 256 candidate sites, 24 h) | `superposition_rel_err` | 3.5e-6 | G a matches the truth anomaly to 3.5e-6, **not machine precision**. The cause was not isolated (joint vs separate solves, solver tolerance at a 1e-4-scale anomaly are candidates). |
| Homogeneity, ×3 source | `homogeneity_rel_err_x3` | 3.8e-16 | linear in amplitude at machine precision |
| Determinism of G | `G_rebuild_max_abs_diff` | 0 | identical on rebuild |
| Affine background (source ×10⁴) | `affine_background_rel_err` | 4.4e-12 | c_bg = 5 run equals 5 + anomaly |
| Identifiable case: K = 2, 8 sites, 24 h | `identifiable_rank`, `identifiable_cond` | 2, 2.7e7 | full rank at 1e-8, but badly conditioned |
| Noiseless NNLS, same case | `noiseless_nnls_rel_err` | 4.3e-10 | exact recovery without noise |
| Noisy NNLS at **benchmark scale** | `benchmark_scale_snr_peak`, `benchmark_scale_nnls_rel_err` | 2.1e-5, 1.0 | peak plume ≪ σ = 0.25 µg m⁻³; the estimate is all zeros (relative error 1) |
| Amplified emissions ×10², ×10³, ×10⁴ (**not** the preregistered scale; one noise seed) | `amp*_snr_peak` | 2.1e-3, 2.1e-2, 0.21 | still far below SNR 1 |
| NNLS error at those amplitudes | `amp*_nnls_rel_err` | 1.0, 1.1e6, 5.4e6 | all-zero at ×10²; noise amplified by cond 2.7e7 at ×10³ and ×10⁴ |
| Ridge NNLS error | `amp*_ridgennls_rel_err` | 1.0, 1.0, 0.51 | regularisation avoids the blow-up but recovers little even at ×10⁴ |
| Underdetermined: K = 4, 2 sites, snapshot | `snapshot_K4_k2_rank` | 2 | rank ≤ 2 < K |
| Non-unique alternative | `nonunique_alt_min_coefficient`, `nonunique_alt_vs_true_distance_rel`, `nonunique_alt_data_misfit_rel` | 80, 0.53, 1.2e-17 | a nonnegative a' that differs from a by 53 % fits the data to round-off |
| Same 2 sites, 24 h window | `window24_K4_k2_rank`, `window24_K4_k2_cond` | 2, 4.3e27 | time history from 2 sites adds **no** rank here: still < K = 4 |
| Transport lag | `lag_arrival_hours_max_reached`, `lag_sites_not_reached_in_12h` | 1, 65 | see limits: arrival is within 1 h wherever the plume exceeds the 1e-3 threshold, so this coarse check cannot resolve lag at the smoke wind speed |

Reading of the hypothesis (stated, not concluded): the **noiseless, identifiable** part is supported at smoke scale (recovery
4e-10). The **noisy** part is *unresolved to contradicted at the preregistered source scale*: SNR ≈ 10⁻⁵ means weighted NNLS returns
zeros; even ×10⁴ emissions leave NNLS unstable and ridge NNLS only partly useful. The non-uniqueness part is demonstrated for one K = 4, 2-sensor case. This
agrees in direction with F-002 (SNR ~2e-3 at the three E02 examples) but uses a different sensor set.

## 6. Failure cases and limits (do not read as validation)

1. **Benchmark scale carries no signal** (§5). A source-recovery comparison at SNR ≈ 10⁻⁵–10⁻³ measures noise. A human decision on source
   scale or noise level is required (gate-02) and was not made here; no metric was changed.
2. **Superposition error 3.5e-6** is unexplained and larger than the 1e-8 tolerance the discrete-exact oracles use. Until it is
   isolated, treat G as accurate to ≈ 4 digits at the plume level.
3. **Weak lag check.** A global 1e-3 threshold over all sites cannot show lag; the sampled wind moves the plume across the
   domain within an hour for most sites. A dedicated lag oracle (single source, steady slow wind, known travel time) is not written.
4. **Window definitions not implemented.** snapshot/6 h/24 h/7 d with the 7-day window ending at the forecast issue time
   (protocol) are not in `benchmarks.py`; only a 24 h block is used here. The case with 7 d (168 h) of data is untested.
5. **Single noise seed, 1–2 hand-placed examples, 8 deterministic sensors**, 32² grid. No uncertainty over replicates, no K = 8, no 64² run. No claim
   about generality.
6. **Perfect model only.** Wrong wind/K/λ/background are mismatch (E12–E14) and not covered; G is built from the truth's own parameters.
7. **Known background only.** The unknown-b variant is derived (§2.4) but not run.
8. **Constrained-estimator uncertainty** (CRB under active constraints) not computed; M⁻¹ is only a local unconstrained approximation.
9. **Ridge strength** was one fixed value; no validation-based selection exists yet.
10. **Citations unverified** (see top). Green ch. 2.2 and Bhardwaj 2026 are used for the formulation only through project notes.

## 7. Provenance

- Code and environment: commit `d053ff7ba9c8…`; numpy 2.5.4; scipy 1.18.1; FiPy LU solver; local host.
- Seeds: example replicate seeds as in `scripts/e02_examples.py` (0 and 2); noise seed for the amplified checks `default_rng(0)`.
- Command: `research run exec EXP-004 -- python3 scripts/e03_derivation_checks.py`. Metrics in the run record; stdout in the run directory.
- Datasets: none external; truth from the E02 pipeline (truth ids are produced by the scenario objects, not recorded here).
- Task T-013, runs RUN-0011 (failed, script bug) and RUN-0012 (completed). Findings: none created; no number here has been bound or independently reviewed.
