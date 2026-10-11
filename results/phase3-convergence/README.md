# phase3-convergence: EXP-015 stage 1 (field-level verification), RUN-0028

`python scripts/convergence_stage1.py --profile production` via `research run exec EXP-015`. Case: configs/geodesign.json draw d0
(design seed 101, 1 case), identical sources/wind/physics everywhere; only mesh, dt, scheme vary. Real geography, synthetic emissions
and meteorology. Stage 2 (sensor design) was not run. No interpretation here.

Files
* `stage1_runs.csv`: one row per unique solve, key (res_m, dt_s, scheme, group; group empty = all sources). `modes` = which of A (space),
  B (time), C (joint), F (fingerprint) the solve serves (a solve shared by modes is run once). Columns: cells, steps, max_substeps
  (vanleer sub-steps, 1 for exponential), dt_eff_s = dt/max_substeps, wall_s, peak_rss_gb (fresh subprocess), rel_budget_residual
  (max relative mass-budget residual, forward diagnostics), min_conc, n_negative (native mesh), wind_speed_ms, keff_over_k =
  1 + numerical_diffusion_estimate(u, dx, dt, K) (exponential-scheme formula; keff_over_k_dteff uses dt_eff; for vanleer both are the
  exponential-scheme formula, NOT a vanleer measurement), mass_*_ug = q * sum(f)dA * integral of the emission profile over the solver steps,
  instudy_* = fraction of the source shape inside the study mask on that mesh, field_file.
* `fields/*.npz`: `field` (time, y, x) block-mean to 500 m (66 x 65), `times` (s).
* `stage1_summary.csv`: per solve x reference. ref = R_exp (125 m, dt 25, exponential), R_vl (125 m, dt 25, vanleer), or F_ref (finest
  exponential fingerprint of the same group). rel_L2 = relative space-time L2 error on the 500 m grid, t >= t_exclude = 7200 s.
  mean_err_max / mean_err_rms = error of the study-mask-weighted mean concentration series (ug/m3; weights = 500 m study_fraction;
  mean_err_max_rel relative to the reference peak). arrival_s = first time (linear interpolation, whole record t >= 0) the study-mean
  exceeds 10 % of the reference's own peak; arrival_err_s = difference to the reference arrival.
* `stage1_orders.csv`: observed order p. `L2 error ratio` = log2(e_coarse/e_fine) from errors against the stated reference (A: levels
  500->250->125 m at dt 25; B: against the 500 m dt 12.5 run, so the 12.5 s level itself has no error). `Richardson` = log2((u0-u1)/(u1-u2))
  on the time-mean (t >= t_exclude) of the study-mean. Non-monotone / non-converging estimates are declared in `status`, nan p.
* `stage1_mass_check.csv`: max relative difference of the emitted mass per group over all runs (flag if > 1e-3), and range of instudy fraction.
* `figures/error_vs_dx_dt.png`, `figures/keff_vs_mesh.png` (titled EXP-015 stage1, RUN-0028).

Deviations / notes
* Run key includes `group`; solves shared between A/B/C are run once (modes lists them). Summary/orders/mass names differ from the
  pre-registered `stage1_fields.csv`.
* Source normalisation is imposed by construction (sum f dA = 1), so the mass check is near-trivial; the grid-dependent quantity is the
  in-study fraction (road 0.949-0.986, builtup 0.977-0.995 across meshes/runs files; external patch ~0.0015-0.0017).
* The 10 % arrival threshold is crossed at ~335 s, driven by local sources (before the external-patch onset at 7200 s).
* Study mask for the study-mean is the 500 m mask on the aggregated fields for every run.
* Cost guard (3 x benchmark per step, scaled by sub-steps) never triggered.
