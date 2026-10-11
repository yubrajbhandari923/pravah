# EXP-021 (production) -- EXP-021 prescribed wind, not terrain-derived

checks: {"max_closure_rel": 3.265741480438904e-14, "closure_limit": 1e-08, "closure_ok": true, "max_negative_rel_to_max_conc": 5e-324, "negative_limit": 1e-09, "negative_ok": true, "max_side_vs_budget_rel": 0.005458827875346475, "advection_distance_m": 86400.0, "reference_half_width_m": 175000.0, "reference_over_2UT": 1.0127314814814814, "worst_exposure_rel_diff_two_largest": 1.3555702168349213e-16, "convergence_tol": 0.01, "converged": true, "enlarged_half_widths_m": [], "enlarge_cap_m": 300000.0, "numerical_exposure_rel_dx_vs_finest": -0.005245049403118237, "numerical_exposure_rel_dx2_vs_finest": -4.617070487634071e-15, "truncation_exposure_rel_smallest_vs_ref_base": -2.32070746914287e-15, "reversals_with_plume_not_cleared": []}

## Convergence (two largest domains)

```
    scenario   t_r_s  hw_second_m  hw_ref_m  exposure_rel_diff   ok
channel_bump     NaN      80000.0  175000.0                0.0 True
channel_step     NaN      80000.0  175000.0                0.0 True
    reversal 18000.0      80000.0  175000.0                0.0 True
    reversal 22500.0      80000.0  175000.0                0.0 True
    reversal 27000.0      80000.0  175000.0                0.0 True
      steady     NaN      80000.0  175000.0                0.0 True
      vortex     NaN      80000.0  175000.0                0.0 True
```

## Numerical check (base scenario, smallest domain)

```
 dx_m  dt_s  ext_exposure  ext_exposure_rel_to_finest  ext_peak_ug_m3
500.0 100.0    881.784157               -5.245049e-03        0.154296
250.0 100.0    886.433545               -4.617070e-15        0.165019
250.0  50.0    886.433545                0.000000e+00        0.169905
```

## Against the reference domain

```
    scenario   t_r_s  half_width_m  ext_exposure_rel_to_ref  returned_eul_minus_ref  ret_eul_after_exit_minus_ref  ret_mass_frac_minus_ref  leave_west_frac_minus_ref  leave_east_frac_minus_ref  leave_south_frac_minus_ref  leave_north_frac_minus_ref
channel_bump     NaN       20000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.47617                        -0.0                        -0.0
channel_bump     NaN       40000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.30049                         0.0                         0.0
channel_bump     NaN       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00823                         0.0                         0.0
channel_bump     NaN      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
channel_step     NaN       20000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.47617                        -0.0                        -0.0
channel_step     NaN       40000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.30049                         0.0                        -0.0
channel_step     NaN       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00823                         0.0                         0.0
channel_step     NaN      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 18000.0       20000.0                 -0.06317                -0.01766                      -0.01766                 -0.05203                    0.17223                    0.14492                         0.0                         0.0
    reversal 18000.0       40000.0                 -0.00000                -0.00000                      -0.00000                 -0.00000                    0.03498                    0.00001                         0.0                         0.0
    reversal 18000.0       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 18000.0      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 22500.0       20000.0                 -0.16806                -0.08573                      -0.08573                 -0.12209                    0.02131                    0.41884                         0.0                         0.0
    reversal 22500.0       40000.0                 -0.00054                -0.00001                      -0.00001                 -0.00039                    0.00013                    0.00383                         0.0                         0.0
    reversal 22500.0       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 22500.0      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 27000.0       20000.0                 -0.07071                -0.06673                      -0.06673                 -0.03264                    0.00010                    0.47454                         0.0                         0.0
    reversal 27000.0       40000.0                 -0.00096                -0.00286                      -0.00286                 -0.00067                    0.00000                    0.07273                         0.0                         0.0
    reversal 27000.0       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
    reversal 27000.0      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
      steady     NaN       20000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.47617                         0.0                         0.0
      steady     NaN       40000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.30049                         0.0                         0.0
      steady     NaN       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00823                         0.0                         0.0
      steady     NaN      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
      vortex     NaN       20000.0                 -0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.43837                         0.0                         0.0
      vortex     NaN       40000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.27362                         0.0                         0.0
      vortex     NaN       80000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00483                         0.0                         0.0
      vortex     NaN      175000.0                  0.00000                 0.00000                       0.00000                  0.00000                    0.00000                    0.00000                         0.0                         0.0
```
