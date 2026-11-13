# Amendment A2: advection scheme and solver step for 64² runs (gate-00 B3)

**Date:** 2026-11-12 · **Authority:** user chose option (a) "try a sharper FiPy scheme and re-run E01 checks" and, in the
protocol approval (`2026-11-12-protocol-approval.md`), asked that settings proven numerically infeasible be amended,
not forced. The lead agent made the specific choices below; the user may override.

## Evidence (`docs/research/gates/b3-advection-scheme.md`, data in `gates/b3/`)
- Upwind at 64², dt 300 s: effective K 1371–2933 m²/s against physical K = 50, and it fails the dt-halving gate at every
  dt tried. At small dt the spatial floor u·dx/2 ≈ 234 m²/s remains.
- Dimensionally split VanLeer, dt 30 s: effective K 49.7–50.9, convergence order ≈ 2, dt-halving change 0.54–0.71 %
  (gate ≤ 1 %), mass conserved to 1e-15, no negative values.
- Lead's independent check (diagonal 3 m/s wind, 20 km periodic box at dx = 156 m, 30 min):
  - upwind, dt 300: K_eff 1545 / 235 (along / across the wind)
  - upwind, dt 30: 351 / 216
  - vanleer, dt 30: 50.1 / 50.1

## Change
1. `ModelConfig.advection_scheme` added. The default stays "upwind", so all existing results and the 32² smoke work are
   unchanged. **64² runs (E07 sweep onward) use `advection_scheme: vanleer`, `dt_s: 30`.** dt_s is the solver step;
   observations stay hourly. `simulate` refuses vanleer when the per-axis Courant number exceeds 1. dt_s must satisfy
   dt ≤ dx / max|u| over the whole wind calendar.
2. The protocol's "first-order scheme, fixed dt = 300 s" (§3) is replaced by this for 64². The numerical gates themselves
   (order, mass, dt-halving ≤ 1 %) are unchanged.
3. Truth and estimator responses both use vanleer. VanLeer's limiter is not exactly linear: in one test, superposition
   was off by about 1 %. Per-source responses G therefore carry a small model error relative to the joint truth run.
   This was chosen over upwind for the estimator, which would impose a 400–1400 m²/s diffusivity mismatch and turn
   plan 2 into a model-error study.
   - Measured on benchmark footprints (64², dt 30 s, mixed wind, K = 4, 96 h): Σ a_k F_k + c_bg vs the joint truth run differs by at
     most 6.7e-3 µg/m³, which is 0.74 % of the peak anomaly (0.29 % relative RMS). Upwind gives 0.011 %. This is small
     next to the 5 % noise σ but is a model error. The E07 sweep records it per scenario (`linearity_max_abs`). **Full calendar (E07 sweep, 1488 h, 40 scenarios):** the max
     error is a median 0.038 µg/m³ and 0.183 at worst (0.73 σ at a single cell-hour), about 27× the 96 h figure. Gate-07
     re-scored designs on a perfectly linear truth: RMSE changed by ≤ 1e-4 and the ordering was unchanged, so the effect on E07 is negligible.

## Costs and limits
- Compute is about 17× upwind at dt 300 s: 7.8 s per simulated hour on 64², serial.
- The split is first order in time. It is exact for uniform wind on a periodic box but not at walls.
- Study-BC behaviour has no independent check yet, the same gap as for upwind.
- At 32² and 16², vanleer over-diffuses narrow (1 km) puffs because they are only a few cells wide.
