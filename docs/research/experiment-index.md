# Experiment index — PRAVAH Phase 1

One master plan with 21 ordered experiments and 96 tasks. All experiments have hypotheses or testable benchmark goals and preregistered metrics. Later tasks depend on earlier tasks; a Research OS dependency **does not merge Git worktrees**.

| Experiment | Aim | Tasks | HPC tasks | Reference cues |
|---|---|---:|---:|---|
| `e00-protocol` | Protocol, literature replication map and frozen evaluation contract | 4 | 0 | green2006, bhardwaj2026, manohar2018 |
| `e01-forward` | Verify 2D advection-diffusion-removal and physical units | 4 | 0 | pennington2024, schlutow2026 |
| `e02-benchmark` | Reproducible synthetic truth, sensors and temporal scenarios | 4 | 0 | green2006, bhardwaj2026, raanes2024 |
| `e03-static-inverse` | Static constrained source-strength inverse baseline | 4 | 0 | green2006, bhardwaj2026, mons2017 |
| `e04-footprints` | Footprints, Fisher information and identifiability | 4 | 0 | green2006, schlutow2026, bhardwaj2026 |
| `e05-duration` | Temporal observability: sensor count versus observation window | 4 | 0 | green2006, bhardwaj2026, bartos2021 |
| `e06-placement-algorithms` | Reproduce core placement algorithms and information criteria | 4 | 0 | green2006, manohar2018, zhou2022 |
| `e07-cross-objective` | Cross-objective placement ablation: source vs field vs forecast proxies | 6 | 1 | green2006, manohar2018, mano2022 |
| `e08-sufficiency` | Sensor sufficiency, field complexity and measurement cost | 5 | 1 | zhou2022, bhardwaj2026, green2006 |
| `e09-kalman` | Causal sequential filtering of the concentration state | 4 | 0 | enkf2009, sharma2019, cheng2022 |
| `e10-joint-assimilation` | Joint source-strength and field sequential inference | 4 | 0 | enkf2009, sharma2019, bhardwaj2026 |
| `e11-forecast-design` | Placement optimized for causal forecasting and sequential reconstruction | 6 | 1 | choi2026, cheng2022, bartos2021 |
| `e12-wind-mismatch` | Meteorological uncertainty and source-transport compensation | 5 | 1 | webster2022, mues2018, mons2017 |
| `e13-nonwind-mismatch` | Background, source-basis mismatch and realistic observation errors | 4 | 0 | bhardwaj2026, pennington2024, choi2026 |
| `e14-uncertainty-calibration` | Reliability diagnostics and uncertainty calibration | 5 | 0 | bhardwaj2026, raanes2024, webster2022 |
| `e15-robust-placement` | Nominal vs robust sensor-placement objectives | 6 | 1 | choi2026, mano2022, webster2022 |
| `e16-generalization` | Generalization, sensitivity and ablation consolidation | 6 | 1 | mues2018, webster2022, choi2026 |
| `e17-regional` | Optional real-data-informed Belauri or public regional transport case | 5 | 1 | era5, worldpop, firms |
| `e18-observation` | Optional independent real monitoring validation and gap assessment | 3 | 0 | choi2026, mues2018, pennington2024 |
| `e19-synthesis` | Integrated results, statistical conclusions and Quarto monograph | 4 | 0 | green2006, choi2026, bhardwaj2026 |
| `e20-audit-release` | Independent replication, final review and research-ready release | 5 | 0 | raanes2024, desilva2021, green2006 |
