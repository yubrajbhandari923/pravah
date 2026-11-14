# E17 regional case: data feasibility audit (Belauri, Kanchanpur)

Status: audit only, written 2026-11-14 on a subagent branch. No data downloaded, no account created, no licence accepted,
no credentials used. Plan: `plans/plan-03-robustness-synthesis.yaml` key `e17-regional`. Facts marked **[unverified]**
could not be confirmed from a page I opened. URLs cited are pages I opened or search results I read.

## 1. Verdict

A regional wind-forcing case is **buildable from public data**, but the cleanest wind source (ERA5 via the Copernicus CDS)
**needs the user's own ECMWF account and a licence acceptance**, which the plan forbids me to do. A credential-free path
exists (ERA5 files on a public S3 bucket, anonymous read) but its terms are not fully verified; a second path (Open-Meteo)
requires accepting a licence in the API terms. E17 is therefore gated on a user decision (section 6). Until then the honest
status is "feasible, not yet authorised".

## 2. Wind data options

| Option | Credential-free? | Resolution / period | Evidence | Concern |
|---|---|---|---|---|
| ERA5 single levels via CDS | **No.** ECMWF account + API key, and the dataset licence ("Licence to use Copernicus products") must be accepted on the dataset download form, also for API use | 0.25 deg, hourly, 1940-present | ECMWF Confluence: https://confluence.ecmwf.int/spaces/CKB/pages/392594360/ and https://confluence.ecmwf.int/spaces/CKB/pages/174856258/ . The CDS dataset page (https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels) opened but showed only title, "from 1940 to present", Login/Register and an outage banner, no licence text | Needs user credentials and acceptance |
| ERA5 mirror, NSF NCAR on AWS (`s3://nsf-ncar-era5/`) | **Yes** for reading ("No AWS account required", `--no-sign-request`) | Hourly NetCDF4, 0.25 deg (`ll025sc`); monthly update, 3-4 month lag | https://registry.opendata.aws/nsf-ncar-era5/ . I made an anonymous prefix listing (no file fetched) and saw `e5.oper.an.sfc/202201/e5.oper.an.sfc.128_165_10u...nc` and `..._166_10v...nc`, so 10 m u/v monthly global files exist for at least 2022-01. Latest available month not checked | Terms at https://www.ucar.edu/terms-of-use/data **not opened** [unverified]. Content is still Copernicus-licensed ERA5 (acknowledgement "contains modified Copernicus Climate Change Service information", https://gdex.ucar.edu/datasets/d633000). Whether anonymous use counts as accepting a "new licence" is the user's call. Global monthly files are large (listing showed ~0.2-0.75 GB objects; per-variable size not verified) |
| Older AWS copy `era5-pds` | Yes | since 2008 | https://registry.opendata.aws/ecmwf-era5/ marks it deprecated | Do not use |
| Open-Meteo Historical API (serves ERA5, 0.25 deg hourly) | No key for free tier, **but** free use means accepting CC-BY 4.0 and non-commercial terms; <10,000 calls/day, 5,000/h, 600/min; abusers may be blocked | 1940-present, 10 m wind speed (and direction) | https://open-meteo.com/en/docs/historical-weather-api , https://open-meteo.com/en/terms | Third-party reprocessing of ERA5; point queries (a 4x4 point grid is trivial); attribution to Open-Meteo and ERA5; returns speed/direction (convert to u,v, document convention). Click-through-style term, so a user decision |
| NOAA GFS archive on AWS (`s3://noaa-gfs-bdp-pds`) | Yes ("No AWS account required") | 0.25 deg, 3-hourly, 2021-present | https://registry.opendata.aws/noaa-gfs-bdp-pds | Operational NWP, not reanalysis; model upgrades (v16.3.0 Nov 2022 noted). Fine as a robustness comparison, not as the primary |
| NCEP/NCAR Reanalysis 1 (NOAA PSL) | Yes (CC0 per data.gov record https://catalog.data.gov/dataset/ncep-ncar-reanalysis-1948-present) | 2.5 deg, 6-hourly | https://psl.noaa.gov/data/gridded/reanalysis ; production ended 17 Mar 2026 | About one cell over a 100 km box. Not recommended |

Station wind: the plan notes the Belauri Santipur station is precipitation-only. The DoEnv Bhimdatta station carries a Lufft
WS 700 met sensor (DoEnv 2021 report,
http://doenv.gov.np/progressfiles/Status-of-Air-Quality-in-Nepal-Annual-Report_2021-1676438644.pdf), about 23 km from Belauri.
Whether its wind records are published for download is **[unverified]**. At most a sanity check of reanalysis wind.

## 3. Ancillary data

- **WorldPop** population counts, Nepal 100 m, CC BY 4.0, no registration mentioned. Opened
  https://hub.worldpop.org/geodata/summary?id=74555 (R2025A, alpha, 2021 estimates, DOI 10.5258/SOTON/WP00839). The plan cites
  id=74560, which I could not open (certificate error); 74555 is the Nepal R2025A entry the search returned. The 2020
  UN-adjusted product (https://hub.worldpop.org/geodata/summary?id=50027) is also CC BY 4.0. The page asks users not to use
  the data for harm or surveillance. Use only as population weights; needs attribution only.
- **FIRMS** active fires: download requires Earthdata login or an emailed one-time code
  (https://firms.modaps.eosdis.nasa.gov/download/); the API requires a free MAP_KEY sent by email, 5000 transactions per
  10 minutes (https://firms.modaps.eosdis.nasa.gov/api/map_key). That is account creation, so **not credential-free**.
  I found no data-use policy text on the pages opened [unverified]. Use would be as a source-location proxy (detections,
  not emission rates), as the plan says. Alternative with no key: omit fires, build candidate masks from WorldPop only.
- Land cover: not audited, not needed for the minimum case.

## 4. Proposed case (conditional on approval)

**Location.** Belauri municipality, 28.79 N, 80.31 E (Wikipedia, https://en.wikipedia.org/wiki/Belauri; a municipal
centroid). DoEnv stations Bhimdatta (28.9651 N, 80.1829 E) and Dhangadhi (28.7041 N, 80.5945 E) (2021 report Annex 1) lie
23 km and 29 km away (great-circle, my calculation). Kanchanpur borders India to the south and west
(https://en.wikipedia.org/wiki/Kanchanpur_District).

**Box.** Primary ~100 x 100 km: 79.80 to 80.82 E, 28.34 to 29.24 N; working CRS UTM 44N (EPSG:32644, to confirm at
implementation). Upwind reasoning (reasoning, not a finding): a passive tracer at 2-4 m/s moves 70-150 km in 10-12 h, so
100 km spans about half a day of advection from the centre to the upwind edge; Indian inflow is then an explicit synthetic
boundary condition. Published trajectory studies put wintertime IGP transport under predominantly north-westerly flow
(https://www.nature.com/articles/s41598-026-37934-9 ; https://www.nature.com/articles/s41467-022-34093-z), so the upwind edge
is mainly west/north-west in the cool season. The seasonal wind direction at Belauri itself, including monsoon-season flow,
I did **not** verify from an opened source; the ERA5 series will show it and E17 must report the empirical wind rose before
fixing the box. Secondary: the benchmark's 10 km domain placed at Belauri is smaller than one 0.25 deg ERA5 cell (~27 km),
so it sees one spatially uniform wind sequence; that tests temporal variability only.

**Grid.** 64 x 64 on 100 km gives ~1.6 km cells (benchmark: 64 x 64 on 10 km, 156 m cells; `configs/benchmark-v1.yaml`), so the
Peclet regime changes and K, dt must be revisited, not copied. ERA5 u,v on ~4 x 4 cells, bilinearly interpolated. A
uniform box-mean wind variant is the matched comparison to the synthetic study.

**Period.** Cool season 1 Nov 2021 to 28 Feb 2022 (2,880 hourly steps) plus one monsoon month as a contrasting regime,
split by date blocks per the preregistered protocol. 2021 is proposed because the DoEnv Bhimdatta station reports it
(station record quality unverified, see E18 audit); the choice is a proposal.

**What it can show.** Whether the synthetic-study method ranking survives a realistic temporal wind spectrum (regime
persistence, stagnation, diurnal change) and spatially varying wind, under synthetic emissions with known truth fields;
sensitivity at ERA5-like wind variability (compare E12).

**What it cannot show.** Real concentrations, emissions, exposure or any validation: all fields are synthetic tracer fields
under reanalysis winds. It cannot test boundary-layer depth (2-D, depth-averaged), sub-grid foothill flow, chemistry or
removal. ERA5 error against local winds is never measured. FIRMS/WorldPop only shape where synthetic sources may sit.

## 5. Effort and compute

- Data: 4-5 monthly global 10u/10v files from the NCAR mirror, order 4-7 GB transient (estimate from listed object sizes),
  subset to the box (MB) and discarded; manifest and hashes committed. CDS or Open-Meteo downloads are KB-MB.
- Code: `src/pravah/data/region.py` (read, u/v conversion, reprojection, bilinear interpolation, UTC to NPT = UTC+05:45,
  quality flags, masks) with unit tests, ~1.5-2 days. Reuses the existing solver and E12 wind plumbing.
- Compute: one 64-CPU job, hours (extrapolated from existing pilots; not measured for this case). Heavy 64^2 sweeps stay
  deferred per project policy; a few-run pilot fits the preliminary allowance.

## 6. Decision table for the user

| ID | Decision | Options | Recommendation |
|---|---|---|---|
| D1 | ERA5 wind source | (a) you supply your own CDS/ECMWF account, accept the Copernicus licence yourself, give me the key; (b) approve anonymous read of the NCAR AWS ERA5 mirror (I read the UCAR terms first and report); (c) approve Open-Meteo free API (you accept CC-BY 4.0 + non-commercial terms); (d) decline: E17 recorded "not conducted" | (b) after the UCAR terms are read; (a) as fallback; (c) only for a quick 16-point pilot |
| D2 | Box and period | section 4 or different | accept; empirical wind rose first |
| D3 | FIRMS | (a) skip fires, WorldPop masks only; (b) you create a MAP_KEY | (a) |
| D4 | Scope label | E17 stays exploratory/optional, labelled "synthetic tracer under ERA5 winds, not validation" | confirm |

I stopped before any access because each of D1(a-c) needs credentials or a term acceptance.
