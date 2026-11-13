# PRAVAH: literature and implementation map

This file is an agent navigation map, not evidence that PRAVAH has reproduced a paper. Open original papers and verify equations, sections, dates and claims. Citations use the keys in `reports/quarto/references.bib`.

**Replication rule:** A reproduction reproduces the equation, benchmark assumptions and metric, not merely a qualitatively similar plot. A PRAVAH adaptation must be labeled as such. Synthetic benchmark performance does not validate observed PM2.5.

## Papers and what to use

- **`pennington2024`** — Pennington et al. (2024), Los Angeles CMAQ chemistry-transport model. https://doi.org/10.5194/acp-24-2345-2024  
  Related experiments: e01-forward; e13-nonwind-mismatch; e17-regional; e18-observation.
- **`schlutow2026`** — Schlutow et al. (2026), BLDFM transport and concentration/flux footprints. https://doi.org/10.5194/gmd-19-6259-2026  
  Related experiments: e01-forward; e04-footprints; e12-wind-mismatch.
- **`green2006`** — Kary Green (2006), Optimal Sensor Placement for Parameter Identification (Rice thesis). https://repository.rice.edu/items/4d4b4f81-1c4b-4362-8513-685eac332c7c  
  Related experiments: e00-protocol; e02-benchmark; e03-static-inverse; e04-footprints; e05-duration; e06-placement-algorithms; e07-cross-objective; e08-sufficiency; e16-generalization; e19-synthesis; e20-audit-release.
- **`mons2017`** — Mons et al. (2017), data-assimilation reconstruction of pollutant release. https://doi.org/10.1016/j.jweia.2017.07.007  
  Related experiments: e03-static-inverse; e12-wind-mismatch; e13-nonwind-mismatch.
- **`bhardwaj2026`** — Bhardwaj and Subramanian (2026), identifiability-aware source apportionment (preprint). https://arxiv.org/abs/2608.00050  
  Related experiments: e00-protocol; e02-benchmark; e03-static-inverse; e04-footprints; e05-duration; e08-sufficiency; e10-joint-assimilation; e12-wind-mismatch; e13-nonwind-mismatch; e14-uncertainty-calibration; e19-synthesis.
- **`manohar2018`** — Manohar et al. (2018), sparse sensor placement for reconstruction. https://doi.org/10.1109/MCS.2018.2810460  
  Related experiments: e00-protocol; e06-placement-algorithms; e07-cross-objective; e11-forecast-design.
- **`zhou2022`** — Zhou et al. (2022), POD-based PM2.5 monitoring-site design across China. https://doi.org/10.1021/acsenvironau.1c00051  
  Related experiments: e06-placement-algorithms; e08-sufficiency; e19-synthesis.
- **`mano2022`** — Mano et al. (2022), information-theory air-pollution sensor location-allocation. https://doi.org/10.3390/s22103808  
  Related experiments: e06-placement-algorithms; e07-cross-objective; e15-robust-placement.
- **`krause2008`** — Krause, Singh & Guestrin (2008), GP near-optimal mutual-information placement. https://www.jmlr.org/papers/v9/krause08a.html  
  Related experiments: e06-placement-algorithms; e15-robust-placement.
- **`choi2026`** — Choi & Hummel (2026), low-cost PM2.5 placement and value of information. https://doi.org/10.1088/1748-9326/ae2b89  
  Related experiments: e00-protocol; e07-cross-objective; e11-forecast-design; e13-nonwind-mismatch; e15-robust-placement; e16-generalization; e18-observation; e19-synthesis.
- **`bartos2021`** — Bartos et al. (2021), observability-based river contaminant sensor placement. https://doi.org/10.1029/2020WR029551  
  Related experiments: e04-footprints; e05-duration; e11-forecast-design; e15-robust-placement.
- **`roy2018`** — Roy, Simonetto & Leus (2018), kriged Kalman filtering and sparse placement. https://doi.org/10.3390/s18061778  
  Related experiments: e09-kalman.
- **`enkf2009`** — Online update of atmospheric dispersion state/parameters using EnKF (2009). https://doi.org/10.1016/j.atmosenv.2009.01.014  
  Related experiments: e09-kalman; e10-joint-assimilation.
- **`sharma2019`** — Sharma et al. (2019), Perron-Frobenius contaminant estimation with EnKF. https://doi.org/10.1016/j.buildenv.2019.05.026  
  Related experiments: e09-kalman; e10-joint-assimilation.
- **`cheng2022`** — Cheng & Paley (2022), 2D diffusion-advection estimation and sensor guidance. https://doi.org/10.1016/j.automatica.2021.110112  
  Related experiments: e09-kalman; e11-forecast-design.
- **`webster2022`** — Webster & Thomson (2022), meteorological ensemble uncertainty in inverse ash transport. https://doi.org/10.1029/2022JD036469  
  Related experiments: e12-wind-mismatch; e14-uncertainty-calibration; e15-robust-placement; e16-generalization.
- **`mues2018`** — Mues et al. (2018), WRF and WRF-Chem in Kathmandu Valley. https://doi.org/10.5194/gmd-11-2067-2018  
  Related experiments: e12-wind-mismatch; e16-generalization; e17-regional; e18-observation.
- **`raanes2024`** — Raanes, Chen & Grudzien (2024), DAPPER software paper. https://doi.org/10.21105/joss.05150  
  Related experiments: e00-protocol; e02-benchmark; e09-kalman; e10-joint-assimilation; e11-forecast-design; e14-uncertainty-calibration; e19-synthesis; e20-audit-release.
- **`desilva2021`** — de Silva et al. (2021), PySensors. https://doi.org/10.21105/joss.02828  
  Related experiments: e06-placement-algorithms; e20-audit-release.
- **`karnik2026`** — Karnik et al. (2026), PySensors 2.0. https://doi.org/10.21105/joss.09265  
  Related experiments: e06-placement-algorithms.
- **`patel2026`** — Patel et al. (2026), gradient-based mutual information placement (advanced optional). https://github.com/sustainability-lab/gsm_aqs  
  Related experiments: optional/support.
- **`era5`** — ECMWF/Copernicus, ERA5 single-level hourly meteorology. https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels  
  Related experiments: e17-regional.
- **`firms`** — NASA FIRMS VIIRS fire detections. https://firms.modaps.eosdis.nasa.gov/download/  
  Related experiments: e17-regional.
- **`worldpop`** — WorldPop Nepal 2026 population estimate. https://hub.worldpop.org/geodata/summary?id=74560  
  Related experiments: e17-regional.

## Upstream code for independent cross-checking

- **fipy** — FiPy finite-volume transport solver. https://github.com/usnistgov/fipy
- **pysensors** — PySensors (POD/QR pivot and constrained placement). https://github.com/dynamicslab/pysensors
- **filterpy** — FilterPy Kalman reference. https://github.com/rlabbe/filterpy
- **dapper** — DAPPER data-assimilation reference. https://github.com/nansencenter/DAPPER
- **bldfm** — BLDFM footprints reference; noncommercial license, do NOT copy into PRAVAH. https://github.com/SchlutowSM2Group/BLDFM
- **gsm** — Gradient-based mutual-information air-quality placement, optional comparator. https://github.com/sustainability-lab/gsm_aqs

## License and computational expectations

- Follow license and pinned commit. Do not copy noncommercial BLDFM code into PRAVAH. Source references are not automatically executable dependencies.
- FiPy remains the one forward PDE solver and is CPU-oriented. GPUs should only be considered for separately verified, profiled dense/ensemble kernels.
- Compare simple placement methods against PySensors; validate causal filter toy examples against FilterPy and optional ensemble diagnostics against DAPPER.
- Use the user-supplied Green thesis PDF locally (recommended path `docs/papers/Green_2006_Optimal_Sensor_Placement.pdf`) and make a concise cited derivation note in `docs/research/literature-replication.md`. Copyright status of the PDF must be checked before GitHub redistribution.

## Suggested initial reading order

1. Green (2006), Chapters 2.2–2.3 and 3 + Appendix A: weighted inverse, Fisher information, D-optimal sensor placement.
2. Bhardwaj & Subramanian (2026, preprint): source-time fingerprint identifiability and background projection.
3. Manohar et al. (2018) + Zhou et al. (2022): POD/QR reconstruction and sensor-count/mode-count ablations.
4. Mano et al. (2022) and Choi & Hummel (2026): meteorology, information theory, value-of-information and cost/population objectives.
5. 2009 atmospheric EnKF, Sharma (2019), Cheng & Paley (2022), DAPPER (2024): causal inference, state/source tracking, forecast-before-update.
6. Webster & Thomson (2022), Mues et al. (2018): uncertainty and real meteorological limits.
