# Placement objectives and constraints (E06 step 1)

Status: written 2026-11-12 on branch `experiment/e06-placement` together with the first implementation
(`src/pravah/placement.py`, contract `tests/test_placement.py`). It defines what each design maximises; it is NOT evidence
that any objective is good for any downstream task (that is E07). Not yet independently reviewed (E06 step 4).
Synthetic 2-D passive tracer only.

## 1. Common setting

- **Sites.** The benchmark's 16×16 candidate lattice (`Scenario.site_xy`, 625 m spacing on the 10 km domain). A design is a
  set S of global site ids.
- **Exclusion mask (candidates).** `Scenario.admissible_sites()`: not on the outer lattice ring and farther than one source
  footprint σ (400 m) from every true source centre. This uses the true source locations, consistent with the protocol rule
  that source locations are known to the estimator. Experiments may pass a further restricted candidate list (E06's "small"
  set: admissible sites on every third lattice node).
- **Minimum spacing.** Every pair in S is ≥ `cfg.design.min_sensor_separation_m` (1250 m) apart. Every design applies it
  as a hard feasibility rule while building S; no design is repaired afterwards.
- **Budget.** k sensors, all of equal cost (no cost weights yet; cost-constrained QR is out of scope). Designs are not
  required to be nested in k; greedy ones are nested by construction.
- **Determinism and ties.** Candidates are scanned in the order of `Problem.cand` (ascending id for the admissible set). A
  candidate replaces the incumbent only if it is better by more than `TIE_TOL = 1e-12` (absolute), so the earliest
  candidate wins ties. `random` is the only stochastic design; its seed is the scenario's `placement` seed stream.
- **Training-only inputs.** Every design input comes from the training split: sensitivities G over the first training hours
  (E06 uses all 96 training hours; in the first 24 h its steady and veering streams are identical) from the truth's solver and winds; POD snapshots are the noise-free field at all sites over the train hours
  (`placement.training_snapshots`, which indexes `split_hours("train")` only). No design reads validation or test data.
  E06 uses validation snapshots only to *score* field reconstruction.
- **Scales.** a_ref = `SourceConfig().reference_rate_ug_s` (1e6 µg/s since amendment A1), σ = the scenario's preregistered
  noise sd (`sc.noise_sigma`, 5 % of the median training truth at candidate sites). The whitened, scaled sensitivity of site
  i is A_i = a_ref · G_i / σ_i (rows = hours, columns = K sources), i.e. the response to a reference-sized source change in
  noise sd units. Noise is assumed independent across sites and hours (the benchmark also has an AR(1) noise variant; under
  it, hours of one site would not add information independently).

Which design uses what:

| design | true source locations | G (train) | σ, a_ref | training field snapshots |
|---|---|---|---|---|
| uniform, random, coverage | only through the exclusion mask | no | no | no |
| source_near | yes (targets) | no | no | no |
| dopt, aopt | yes (G columns are the known sources) | yes | yes | no |
| diversity | yes (via G) | yes | yes (eligibility) | no |
| podqr | indirectly (the training field is produced by the true sources and strengths) | no | σ (truncation only) | yes |
| podqr_num (exploratory) | indirectly, as podqr | no | no | yes |

POD snapshots assume a perfect, noise-free archive of the training-period field at every candidate site. That is an
optimistic assumption (a real network would have a model or reanalysis there, with error) and it gives podqr access to the
training-period source strengths, which no other design sees.

## 2. Objectives

### A. Bayesian D-optimality (and A, E)

With prior θ ~ N(0, a_ref² I) on the K source strengths and data y_S = G_S θ + ε, ε ~ N(0, σ² I), the posterior precision in
units of the prior is M(S) = I + A_Sᵀ A_S = I + Σ_{i∈S} A_iᵀ A_i.

- **D:** f_D(S) = log det M(S). This equals 2 × the mutual information (nats) between θ and y_S. Maximised greedily
  (`dopt`). The identity prior keeps it finite when G_S is rank deficient, so no ad hoc eigenvalue floor is needed.
- **A:** f_A(S) = 1 − tr(M(S)⁻¹)/K, the mean fractional posterior variance reduction (0 = prior only, → 1 = exact).
  Maximised greedily (`aopt`).
- **E:** f_E(S) = log λ_min(M(S)), the worst-informed source direction. Reported only; not optimised (it is flat under
  greedy additions until all directions are informed).

Greedy: add the feasible candidate with the largest objective value, k times. f_D is monotone and submodular in S (information
gain with observations that are conditionally independent given θ; Krause & Guestrin 2005 discuss this class, we have not
read that paper in this project), so without a binding spacing constraint greedy attains at least (1 − 1/e) of the optimum
(Nemhauser, Wolsey & Fisher 1978). The spacing rule is not a matroid constraint in general, so with it no such guarantee is
claimed. f_A is not submodular in general; no guarantee is claimed. `identifiability.greedy_design` (E04) is now a thin
wrapper of the same code.

Relation to Green (2006): the thesis designs sensors by D-optimality of the local Fisher information G ᵀR⁻¹G for parameter
identification (reading notes: `docs/research/green-2006-reading-notes.md`, PDF pp. 21–46). PRAVAH's version differs: the
parameters are K source strengths (linear in G, so Fisher information does not depend on θ), the prior regularises
singular designs, and the design is a discrete greedy subset rather than Green's approximate (weighted) design.

### B. Footprint diversity

The footprint of site i is f_i = vec(G_i) ∈ R^{hours·K}: how that sensor's time series responds to each source. Coherence
of a design is μ(S) = max_{i≠j∈S} |cos(f_i, f_j)| (1 if a footprint is zero). Noise scaling is explicit through an eligibility
threshold: site i is above noise if snr_i = σ_max(A_i) ≥ τ, τ = 1, i.e. the site alone has practical rank ≥ 1.

Greedy rule (`diversity`): for each feasible candidate compute the key (snr_i ≥ τ, −max_{j∈S}|cos(f_i, f_j)|, snr_i) and add
the lexicographically largest. So an above-noise site always beats a sub-noise one; among above-noise sites the least
coherent with the current design wins; the first site is the strongest one. **Own goal:** small μ(S) using above-noise sites
only. Exhaustive search over above-noise candidates gives its gap. μ ignores signal magnitude beyond the threshold and the
number of sources a site sees, so a low-coherence design can still be poorly conditioned for the sources. This is a
heuristic (the plan's B), distinct from D-optimality, not an approximation of it. The column coherence of G_S (between
sources) is a different quantity; it is computed by `information.rank_coherence` and not optimised here.

### C. POD/QR

Ψ_r = first r left singular vectors of the time-centred training snapshot matrix X (sites × train hours). With r = k, QR
column pivoting of Ψ_r[cand]ᵀ (Businger–Golub) picks the k pivots. Each pivot step multiplies |det Ψ_r[S]| by the largest
remaining residual column norm, so QR pivoting is a greedy maximiser of f_QR(S) = log det(Ψ_r[S]ᵀ Ψ_r[S]) (the conditioning of
reconstructing the r POD coefficients from the k readings). **Own goal:** large f_QR. The spacing rule is applied inside the
pivoting (a candidate that violates it is skipped at that step); with no binding spacing our pivots equal
`scipy.linalg.qr(Ψᵀ, pivoting=True)` and PySensors 0.4.3 (`optimizers.QR`, `SSPOR` with an SVD basis), both tested.

**Noise-aware truncation (added after a first run; see below).** Mode j is *above noise* if some candidate would see it
at ≥ 1 noise sd: max_{i∈cand} s_j |U_ij| / (√T σ_i) ≥ 1 (rms over the T training hours of the mode's contribution to that
site's reading). r_eff counts such modes. `podqr` is **undefined** (ValueError, recorded as such) when k > r_eff, because
pivots beyond the above-noise modes would be chosen by numerical dust. `podqr_num` is an explicitly exploratory,
noise-unaware variant that only requires k ≤ r_num (numerical rank, singular values > 1e-10 × the largest); it shows what
the QR algorithm does, not a design we would recommend at this noise level. f_QR is evaluated with r = min(|S|, r_num).

Why this was added: the first full run stopped because the steady-wind training snapshots had only 2 nonzero modes. A
diagnostic (4 scenarios, seed 1) showed that at the preregistered σ the time-varying part of the training field is at or
below noise: the largest per-site mode amplitude in noise units was 1.07 and 0.69 (steady, K = 2/4) and 1.62 and 0.76
(veering), with every further mode < 0.45. The cause is the smoke calendar: 4 training days are two 48 h wind blocks, the
field reaches a steady state within each block (spin-up 12 h is enough on 10 km), and sources are constant, so the
centred field is nearly piecewise constant in time. The rule was fixed before the full run but after seeing this
diagnostic; it is a revision, not a preregistered choice.

Source: Manohar et al. (2018), IEEE Control Systems Magazine 38(3), 63–86 (DOI verified via OpenAlex; we have read only the
abstract, not the equations). The mechanics above follow the PySensors source code that we did read (`QR.fit` calls
`scipy.linalg.qr(basis.T, pivoting=True)`). k > r oversampling (Manohar's QR of ΨΨᵀ) is not implemented; we always use r = k.
Zhou et al. (2022) applied POD-based site design to PM2.5 in China; we have not read it and draw nothing from it beyond the title.

### D. Baselines

- **uniform:** targets at the cell centres of a ⌈√k⌉ × ⌈k/⌈√k⌉⌉ grid over the candidates' bounding box, row by row; each takes
  its nearest feasible candidate.
- **random:** seeded permutation of the candidates, first k feasible.
- **coverage:** farthest-point sampling; start at the candidate farthest from the candidates' centroid.
- **source_near:** round robin over the known sources; each takes its nearest feasible candidate (the mask keeps it
  ≥ 1 footprint σ away from the centre).

## 3. Links between objectives (stated without equivalence claims)

- D (here) and GP mutual information (Krause, Singh & Guestrin 2008, JMLR 9; verified title/year, not read in full) both
  use log determinants and greedy selection, but they measure different things: D is the information about K source
  strengths given a physical model; Krause et al. maximise the information that the chosen locations give about the
  unobserved locations of a Gaussian-process field. With a linear-Gaussian model the field covariance would be
  a_ref² G Gᵀ + σ² I, so related computations appear, but the objectives and optima differ.
- POD/QR conditions field reconstruction in a data-derived basis; it neither knows the noise level nor the sources as
  separate parameters. D conditions the source estimate. A design can be good for one and poor for the other; E07 tests this.
- Mano et al. (2022) use information theory for air-pollution sensor location-allocation; not read, nothing taken from it.

## 4. What E06 checks (and does not)

Checked: valid designs (distinct, admissible, spaced), determinism, greedy monotonicity and nesting, greedy vs exhaustive on
tiny sets (gap reported, optimality not promised), QR vs scipy/PySensors, duplicate-footprint redundancy, training-only POD
input. Not checked: task performance on held-out data (E07), robustness to wind/source error (E15), cost weights.

## Third-party code and licence

PySensors (`python-sensors` 0.4.3, MIT licence, pinned by `uv.lock`) is a dev-only dependency used in one test to cross-check QR
pivots; no PySensors code is copied into `src/`. Its `QR.fit` is scipy's pivoted QR; `placement.qr_select` re-implements the same
pivoting so that it can skip sites violating the spacing rule, and matches both when spacing does not bind (gate-06 §8).
