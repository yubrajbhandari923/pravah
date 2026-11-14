# E17 regional case (Belauri): design note (pre-pilot)

Status: written 2026-11-14 on a subagent branch after the ERA5/WorldPop download and the wind rose, **before any E17
model run**, and committed alone with the data manifest. Later changes go into §9 (Deviations). Plan: `plans/plan-03-
robustness-synthesis.yaml` key `e17-regional`; feasibility audit: `regional-feasibility.md` (its proposed case is
followed except where §9 says otherwise). User decision 2026-11-14: ERA5 via the user's own CDS account; WorldPop
allowed; FIRMS skipped.

**Label for every table, figure and sentence: "synthetic tracer under ERA5 winds, not validation".** All concentrations
are model-generated synthetic fields; nothing here measures real pollution, emissions or exposure, and ERA5 is reanalysis,
not observed truth.

## 1. Question

Do the main synthetic-study conclusions survive realistic reanalysis wind sequences (regime persistence, calm spells,
the diurnal cycle) instead of the benchmark's hand-made regimes?

1. **Placement ranking (E07/E08):** dopt vs source_near vs random vs coverage, held-out forecast RMSE.
2. **Estimator memory (E09, gate-09):** under the 24 h windowed MAP (`map24`) source_near beat dopt; with all past data
   (`map_grow`) dopt won. Does that flip reappear?
3. **Wind-error sensitivity (E12):** how much does forecast skill degrade when the model wind is a realistic wrong wind
   (the ERA5 grid point nearest Belauri, a "one station" wind, instead of the box mean) or a 30° veer (E12's severe dir)?

Exploratory throughout (no preregistered test, plan marks E17 optional). Expected outcomes are not assumed: the weak,
unsteady cool-season winds (§3) may make every design look alike, which would itself be a result.

## 2. Data (manifest: `docs/research/data/e17-manifest.json`)

- ERA5 single levels (`reanalysis-era5-single-levels`), 10 m u/v (+ boundary-layer height, for a note only), hourly,
  area N 29.5 / W 79.5 / S 28.0 / E 81.25 (the box padded by ≥ one 0.25° cell), Jul 2021 and Nov 2021–Feb 2022.
  Five monthly NetCDF files, 0.37–0.41 MB each, sha256 in the manifest; raw files read-only under the gitignored
  `.cache/e17/raw/`. Licence: CC-BY 4.0 (the CDS replaced the Copernicus licence on 2025-07-02). Required notice:
  "Contains modified Copernicus Climate Change Service information [2021–2022]. Neither the European Commission nor
  ECMWF is responsible for any use that may be made of the Copernicus information or data it contains." Cite Hersbach
  et al. (2023), ERA5 hourly data on single levels from 1940 to present, C3S CDS, doi:10.24381/cds.adbb2d47.
- WorldPop 2020 UN-adjusted population counts, Nepal, 1 km (CC BY 4.0, no registration). Used only as population
  weights and to shape masks. The raster covers Nepal only: the Indian part of the box (west of ~80.06° E and the south)
  is NoData, so there are no candidate sensors and no synthetic sources there.

## 3. Empirical wind rose (box-mean 10 m wind, UTC hourly; `reports/quarto/tables/e17-local-winds.csv`, figure `e17-local-windrose.png`)

| month | mean speed (m/s) | p95 | max | hours < 1 m/s | vector-mean FROM | steadiness | median BLH (m) |
|---|---|---|---|---|---|---|---|
| 2021-07 | 1.42 | 2.38 | 3.72 | 22 % | 99° (E) | 0.75 | 240 |
| 2021-11 | 1.05 | 1.96 | 3.00 | 50 % | 286° (WNW) | 0.40 | 47 |
| 2021-12 | 1.07 | 1.87 | 2.81 | 46 % | 307° (NW) | 0.48 | 49 |
| 2022-01 | 1.15 | 2.07 | 3.47 | 41 % | 290° (WNW) | 0.07 | 77 |
| 2022-02 | 1.52 | 3.06 | 4.65 | 27 % | 320° (NW) | 0.24 | 88 |

(steadiness = |vector mean| / mean speed.) Observations:
- **Cool season: weak and unsteady.** Box-mean speed ~1 m/s, 27–50 % of hours below 1 m/s; no sector holds more than
  ~12 % of hours. The net drift is from W–NW, as the feasibility audit expected from IGP trajectory studies, but it is
  small (0.1–0.5 m/s vector mean). There is a clear diurnal cycle (Nepal time UTC+05:45): afternoons (12–15 NPT) from
  SW–WSW at ~1.6–1.7 m/s, nights and mornings from ~N at ~1.0–1.1 m/s (plausibly foothill drainage flow; not verified).
- **Monsoon (Jul 2021): steady easterlies** (67–112° holds ~67 % of hours, steadiness 0.75), a genuinely different regime.
- **Spatial variability inside the box** (bilinear field at 32² cell centres): median within-box sd 0.64 m/s (u), 0.40
  m/s (v), comparable to the mean speed. The solver only takes spatially uniform winds (`forward.py`), so the box mean is
  the primary forcing (§4); the ERA5 point nearest Belauri (28.75 N, 80.25 E) differs from the box mean by 8–19° in mean
  absolute direction and +10–30 % in speed over the test segments (used as a realistic wrong wind, §6).
- ERA5 boundary-layer height is very shallow in the cool season (median 47–88 m, nocturnal), far from the model's
  500 m mixing depth. Noted only: the 2-D model is depth-averaged and the concentration scale is arbitrary.

**Box decision.** With a ~0.4 m/s net drift (~35 km/day) and no dominant sector, an off-centre box is not justified; the
proposed 100 km square centred on Belauri (79.80–80.82 E, 28.34–29.24 N) is kept. Upwind inflow (India, W/NW; foothills,
N) enters only as the clean-background Dirichlet boundary.

## 4. Domain, grid, physics

- Local tangent-plane projection about (28.79 N, 80.31 E), x east / y north in metres; ERA5 u, v map to the model's
  (ux, uy) with no rotation (distortion < 0.1 % over 100 km). The feasibility note proposed UTM 44N; the tangent plane
  avoids a pyproj dependency and is as accurate here (§9).
- **Geometry scaled ×10 from the benchmark** so the network/source geometry is non-dimensionally E07's: domain 100 km,
  source footprint σ 4 km, 12×12 source lattice, source separation ≥ 15 km, 16×16 candidate lattice, sensor separation
  ≥ 12.5 km. Removal 1/(12 h), mixing depth 500 m, background 5, a_ref 1 g/s, relative noise 5 % (unchanged).
- **K and dt revisited.** K = 500 m²/s (×10) keeps E07's cell Péclet number u dx/K at equal wind speed:
  32² (dx 3.1 km): Pe ≈ 6 at 1 m/s, 29 at the 4.65 m/s maximum; 64² (dx 1.56 km): Pe ≈ 3 / 15 (E07 64²: ≈ 9 at 3 m/s).
  The value is within the range of km-scale horizontal eddy diffusivities but not tuned to Belauri.
  Pilot 32²: implicit upwind, dt 300 s (as the E07 pilot; upwind numerical diffusion u dx/2 ≈ 1.6 × 10³ m²/s at 1 m/s
  dominates K, as in E07's pilot). 64²: split VanLeer (amendment A2), **dt 240 s**: per-axis Courant
  max|u| dt/dx = 4.82 × 240 / 1562 = 0.74 for the largest component of any forcing used (incl. the point wind); limit 1, A2 ran at 0.69.
- The physical time scales differ from E07 and that is part of what is tested: box transit at 1 m/s is ~28 h (E07:
  ~1 h at 3 m/s on 10 km), longer than the 12 h removal time, so plumes recirculate and accumulate in calm spells.
  Spin-up 48 h (replaying the first winds, benchmark rule).

## 5. Sources, candidates, splits

- Synthetic sources: benchmark generator (`draw_sources`, same seeds and log-uniform strengths), restricted to the
  12×12 lattice blocks whose WorldPop population is ≥ the median of populated blocks (settlement-linked proxy, not an
  inventory). Candidate sensors: benchmark admissible sites (no outer ring, > 1 σ from sources) whose cell is ≥ 50 %
  inside the WorldPop raster (deployable in Nepal).
- Splits by contiguous **date blocks** (UTC), train → validation → test, solved as one continuous calendar:

| case | pilot (days 10/2/10) | sweep (days 30/12/18) | test regime |
|---|---|---|---|
| coolA | 1–10 Nov / 11–12 Nov / 13–22 Nov 2021 | Nov 1–30 / Dec 1–12 / Dec 13–30 | same season, later dates |
| coolB | 1–10 Jan / 11–12 Jan / 13–22 Jan 2022 | Jan 1–30 / Jan 31–Feb 11 / Feb 12–28 (17 d) | same season, later dates |
| monsoon | train/val as coolA, test 1–10 Jul 2021 | train/val as coolA, test Jul 1–18 2021 | unseen regime (E07 analogue) |

  The monsoon case concatenates November and July winds (one discontinuity at the val→test boundary, after which the
  first 6 test hours are warm-up, as in E07).

## 6. Methods and endpoints (identical to E07/E08 where possible)

- Designs from the oracle training responses only: dopt (greedy + swap), source_near, coverage, random (5 networks,
  averaged per scenario), k = 2, 4, 8 (pilot) / 4, 8, 16 (sweep).
- Estimators: `map24` (24 h window, E07) and `map_grow` (all past data, gate-09), prior N(0, I) in a_ref units, white
  noise, R = 3 (pilot) / 5 (sweep) paired noise draws shared by all designs.
- **Primary endpoint:** held-out (test) 1 h forecast RMSE over the full grid (`cross_objective.score`); secondary:
  analysis RMSE, source RMSE/a_ref, population-weighted forecast RMSE with **WorldPop weights** (replaces the synthetic
  3-Gaussian weight). Baselines: background-only, persistence.
- Summaries: per (case, estimator, k) the paired relative reduction ρ = (RMSE_b − RMSE_a)/RMSE_b for each strategy vs
  random and dopt vs source_near, median and range over clusters (K, seed), sign counts. Pilot: K ∈ {2, 4} × seeds
  {0, 1, 2} = 6 clusters per cell, too few for inference; read as direction only.
- Wrong model wind (coolA only): the truth and readings use the box-mean wind; the estimator and forecast use unit
  responses under (a) the ERA5 point nearest Belauri, (b) the box-mean wind veered 30° (E12 `dir30`). Designs stay the
  oracle designs. D = RMSE(spec)/RMSE(oracle), as in E12.

## 7. What it cannot show

No real concentrations, emissions, exposure or validation. Spatially varying wind is not used (solver limitation; future
work needs a non-uniform-wind convection term and a protocol amendment). No terrain, boundary-layer depth, chemistry or
deposition beyond linear removal. Indian sources and inflow are absent (clean boundary). ERA5 error against local winds
is not measured. WorldPop masks are proxies.

## 8. Compute

Pilot: 32², 3 cases × 2 K × 3 seeds = 18 scenarios, 72 + 36 (wrong-wind) solves of 576 model hours (~95 s each,
measured 0.165 s per model hour), ≤ 6 local workers. Sweep plan: `docs/research/hpc/e17-regional-run.md` (one ≤ 64-CPU
job, not submitted).

## 9. Deviations

- From the feasibility note: tangent-plane instead of UTM 44N (§4); WorldPop 2020 UN-adjusted 1 km (R2020) instead of
  the R2025A 100 m entry (1 km suffices for 1.6–3.1 km cells, smaller file, stable URL); box unchanged.
- Noted after the pilot (no rule changed): for a given (K, seed) the three cases share source locations and strengths
  (same truth seed and mask), so case differences are wind-driven only. The driver gained `--dry-run` and caches the
  per-case winds and masks (no numerical change).
