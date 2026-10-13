# EXP-015 stage 2: decision stability across mesh and dt (convergence_stage2)

Real geography; synthetic emissions and meteorology. Numbers only, no interpretation. Reference = r250_dt50. alpha frozen = 0.01; noise sigma = fraction x 0.25096 ug/m3 (frozen validation signal scale), identical on every mesh. Replicate baseline = replicate design ensembles on the r500_dt100 mesh; cells show median [IQR].

Success-criterion flags: `in_iqr` = mesh-induced value within the replicate-pair IQR; `monotone` = displacement of the coarse mesh >= that of the refined mesh at fixed dt 100 (500 -> 250 m).

## noise 0.05, budget 3

Replicate pairs (n=28): mean disp [m] 3.94e+03 [3.01e+03, 4.56e+03]; Jaccard 0 [0, 0]; |regret| 1.76e-05 [9.81e-06, 4.42e-05]. Replicate vs main (n=8): mean disp [m] 3.35e+03 [2.94e+03, 4.06e+03]; Jaccard 0 [0, 0.05]; regret 3.74e-05 [2.23e-05, 4.15e-05].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 0.20 | 2532 | 4243 | 0.33 | 0.33 | -1.71e-06 | 0.004662 / 0.00634 | 88/153 | False, False |
| r500_dt50 | 0.20 | 2532 | 4243 | 0.33 | 0.33 | -1.71e-06 | 0.004662 / 0.00634 | 88/153 | False, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.00634 / 0.00634 | 88/153 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.00634 / 0.00634 | 88/153 | False, False |

monotone (500 -> 250 m at dt 100): True

## noise 0.05, budget 5

Replicate pairs (n=28): mean disp [m] 3.19e+03 [2.57e+03, 3.67e+03]; Jaccard 0 [0, 0.111]; |regret| 6.72e-06 [2.73e-06, 1.15e-05]. Replicate vs main (n=8): mean disp [m] 2.97e+03 [2.84e+03, 3.33e+03]; Jaccard 0.0556 [0, 0.146]; regret 6.34e-06 [4.07e-06, 1.26e-05].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 0.43 | 2771 | 10500 | 0.60 | 0.60 | -5.92e-07 | 0.003227 / 0.00499 | 202/245 | True, False |
| r500_dt50 | 0.43 | 2771 | 10500 | 0.60 | 0.60 | -5.92e-07 | 0.003227 / 0.00499 | 202/245 | True, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.00499 / 0.00499 | 202/245 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.00499 / 0.00499 | 202/245 | False, False |

monotone (500 -> 250 m at dt 100): True

## noise 0.05, budget 8

Replicate pairs (n=28): mean disp [m] 2.65e+03 [2.26e+03, 2.99e+03]; Jaccard 0.0667 [0.0667, 0.143]; |regret| 1.11e-06 [6.19e-07, 1.74e-06]. Replicate vs main (n=8): mean disp [m] 3.21e+03 [2.92e+03, 3.58e+03]; Jaccard 0.0667 [0, 0.0857]; regret 2.99e-06 [2.31e-06, 3.47e-06].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 0.60 | 607 | 3354 | 0.75 | 0.75 | -2.14e-07 | 0.002355 / 0.002517 | 367/368 | False, False |
| r500_dt50 | 0.60 | 607 | 3354 | 0.75 | 0.75 | -2.14e-07 | 0.002355 / 0.002517 | 367/368 | False, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.002517 / 0.002517 | 367/368 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.002517 / 0.002517 | 367/368 | False, False |

monotone (500 -> 250 m at dt 100): True

## noise 0.15, budget 3

Replicate pairs (n=28): mean disp [m] 3.56e+03 [2.45e+03, 4.85e+03]; Jaccard 0 [0, 0.2]; |regret| 0.000364 [4.95e-05, 0.000613]. Replicate vs main (n=8): mean disp [m] 2.35e+03 [1.96e+03, 3.01e+03]; Jaccard 0 [0, 0.05]; regret 0.000112 [7.95e-05, 0.000453].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.01316 / 0.01316 | 21/153 | False, False |
| r500_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.01316 / 0.01316 | 21/153 | False, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.01316 / 0.01316 | 21/153 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.01316 / 0.01316 | 21/153 | False, False |

monotone (500 -> 250 m at dt 100): True

## noise 0.15, budget 5

Replicate pairs (n=28): mean disp [m] 2.78e+03 [2.39e+03, 3.06e+03]; Jaccard 0.111 [0, 0.111]; |regret| 3.43e-05 [1.9e-05, 4.83e-05]. Replicate vs main (n=8): mean disp [m] 2.49e+03 [2.18e+03, 3.02e+03]; Jaccard 0.0556 [0, 0.146]; regret 6.1e-05 [4.12e-05, 7.63e-05].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.007503 / 0.007503 | 151/245 | False, False |
| r500_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.007503 / 0.007503 | 151/245 | False, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.007503 / 0.007503 | 151/245 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.007503 / 0.007503 | 151/245 | False, False |

monotone (500 -> 250 m at dt 100): True

## noise 0.15, budget 8

Replicate pairs (n=28): mean disp [m] 2.87e+03 [2.37e+03, 3.08e+03]; Jaccard 0.0667 [0.0667, 0.143]; |regret| 1.98e-05 [5.28e-06, 2.86e-05]. Replicate vs main (n=8): mean disp [m] 2.41e+03 [2.29e+03, 2.57e+03]; Jaccard 0.0667 [0, 0.165]; regret 2.64e-05 [2.23e-05, 4.76e-05].

| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |
|---|---|---|---|---|---|---|---|---|---|
| r500_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.006144 / 0.006144 | 196/368 | False, False |
| r500_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.006144 / 0.006144 | 196/368 | False, False |
| r250_dt100 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.006144 / 0.006144 | 196/368 | False, False |
| r250_dt50 | 1.00 | 0 | 0 | 1.00 | 1.00 | 0 | 0.006144 / 0.006144 | 196/368 | False, False |

monotone (500 -> 250 m at dt 100): True

Files: stability.csv, replicate_baseline.csv, layouts.csv/geojson, test_scores.csv, epsilon_swaps.csv, candidate_mapping.csv, figures/. eps set = number of single-site swaps of the reference layout whose reference-mesh design objective is within one SD (over replicates) of the reference layout's objective on the replicate ensembles.
