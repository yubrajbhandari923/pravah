# Superseded or mislabeled figures in docs/figures/dynamic (2026-10-15)

Nothing here is deleted. These figures were produced by `scripts/make_dynamic_figures.py` and `scripts/make_dyn34_figures.py` for the earlier report
(`docs/DYNAMIC_POPULATION_SENSOR_RESEARCH_REPORT.md`). Use the replacements in `docs/figures/final/` (index: `docs/figures/final/figure_index.json`).

| Stale file | Why | Replacement |
|---|---|---|
| `fig10_dyn03_history.png` | RUN-0092 (DYN-03 v1; F-052/F-053), non-causal CT/CAUSAL comparators, withdrawn | `final/fig04_dyn03_forest.png` (RUN-0094, DYN-03 v2, F-065), `final/fig03_history_reconstruction.png` |
| `fig07_dyn06_detection_misattribution.png`, `fig07b_dyn06_wrong_group_heatmap.png` | RUN-0086 (DYN-06 v3; F-044), superseded by F-060 -> F-063 (RUN-0093) | `final/fig07_dyn06_detection_misattribution.png` |
| `fig04_dyn02v2_inversion.png` | Name says "v2" but the data are RUN-0087 (DYN-02 v3, F-054 -> F-055) | `final/fig10_dyn02_by_wind.png` |
| `fig05_dyn07_robustness.png` | RUN-0088 (DYN-07 v2; F-049 -> F-056); interpretation withdrawn, numbers stand | `final/fig08_dyn07_model_error.png` |
| `fig03_dyn00_verification.png` | Older DYN-00 check set (F-038 -> F-051, 25/25 checks) if not regenerated from RUN-0091 | `final/fig00_dyn00_verification.png` |
| `fig09_dyn04_layouts.png` | RUN-0090 numbers are current (F-061), but the figure shows F-050-era wording | `final/fig05_dyn04_forest.png`, `final/fig02_sensor_layouts.png` |
| `fig06_dyn05_sufficiency.png` | RUN-0078/0080 (F-041/F-045/F-046/F-058 -> F-062) | `final/fig06_dyn05_sufficiency.png` |

Not superseded (no replacement): `fig01_domain_population_layouts.png`, `fig02_winds_and_example_histories.png` (final/fig01 is related but different), `fig08_performance.png` (EXP-025 benchmark).
The `tables/` CSVs belong to the same figures; `tables/fig06_design_summary.csv` and `dyn05_actual_regret_ci.json` are still cited as sources by `final/figure_index.json`, so keep them.
