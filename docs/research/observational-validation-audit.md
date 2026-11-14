# E18 observational validation: public PM2.5 audit near Belauri (far-west Nepal)

Status: audit only, written 2026-11-14 on a subagent branch. No data downloaded, no account created, no licence accepted,
no credentials used. Plan: `plans/plan-03-robustness-synthesis.yaml` key `e18-observation`. Machine-readable summary:
`reports/quarto/tables/e18-status.csv`. Anything not read on an opened page is marked **[unverified]**; I give no station
count that I could not read. Belauri reference point 28.79 N, 80.31 E (Wikipedia). Distances are great-circle, my calculation.

## 1. Sources audited

**Nepal Department of Environment (DoEnv) network.** 27 real-time stations established since 2016, all with a GRIMM EDM 180/180+
optical monitor reporting PM1, PM2.5, PM10 and TSP at one-minute resolution; QC rule: PM2.5 above 1500 ug/m3 removed; daily
values used only when hourly availability >= 80% (DoEnv 2021 and 2022 reports:
http://doenv.gov.np/progressfiles/Status-of-Air-Quality-in-Nepal-Annual-Report_2021-1676438644.pdf ;
https://giwmscdnone.gov.np/media/pdf_upload/Status-of-Air-Quality-Report-of-Nepal,-2022-1714923705%20(1)_gmryjft.pdf).
Near Belauri:
- **Bhimdatta (Mahendranagar), Kanchanpur**: 28.9651 N, 80.1829 E, ~23 km; established 2020; urban; GRIMM EDM 180+ plus a Lufft
  WS 700 met sensor. 2021 PM2.5 ranged 1 to 967.2 ug/m3 (2021 report). A 2021 monthly table printed in Environment Statistics
  of Nepal 2024 (https://giwmscdnone.gov.np/media/pdf_upload/Environment_Statistics_of_Nepal_2024_1j2qkgo.pdf) shows only about
  five months with data for Bhimdatta and a total of ~103 valid days (the excerpt I read was truncated; treat as approximate).
  The 2022 report analysed 11 stations, none in Sudurpashchim, and the World Bank report (below) says latest-year data were
  not reported for some stations. The live portal listing I read (https://pollution.gov.np/portal, capture date unknown)
  showed Bhimdatta with no current value.
- **Dhangadhi, Kailali**: 28.7041 N, 80.5945 E, ~29 km; 2021 table excerpt shows 210 valid days over eight listed months
  (same source; approximate). The portal listing showed 17.1 ug/m3 at capture.
- Next stations: Nepalgunj ~152 km; everything else (Dang, Surkhet, Kathmandu valley, etc.) is >150 km away. A 2021 Annex lists
  coordinates for the stations (Nepalgunj 28.0528 N, 81.6222 E; Dang 27.9914 N, 82.5346 E).
- **Count within 100 km of Belauri: 2 (Bhimdatta, Dhangadhi)**, from the 2021 Annex 1 list. Whether other stations (for
  example Surkhet, Karnali, or stations added after 2021) fall within 100 km: Surkhet is listed on the portal but its
  coordinates were not in the excerpt I read, so **[unverified]**; Surkhet is in the hills and I expect it is farther than
  100 km but did not check.
- Access: the World Bank report states one-minute data are sent to the NITC server and "made available for download in CSV
  format through the official pollution monitoring website"
  (https://documents1.worldbank.org/curated/en/099060525033528377/pdf/P176456-fd25d1c4-c05a-4740-9984-ada6a89056a2.pdf).
  Whether download needs a login, and the data licence, are **[unverified]**: the report PDFs say DoEnv "encourages
  reproduction and dissemination" of its information products (that is the report, not necessarily the raw data).
- Calibration: no co-location or calibration statement for the GRIMM units was seen in the excerpts read **[unverified]**.
  The World Bank report notes annual averages were computed from partial data coverage, which "could affect reliability".

**US Embassy Kathmandu and Phora Durbar (US Department of State) via AirNow.** Two stations with hourly PM2.5 measured by MetOne
E-BAM, published on AirNow (World Bank report above). ~505 km from Belauri (Kathmandu approx. 27.7 N, 85.3 E). Access is
through the AirNow API, which requires a free account (https://docs.airnowapi.org/docs/AirNowAPIFactSheet.pdf) and agreement to
the AirNow Data Use Guidelines (https://docs.airnowapi.org/account/request): data are preliminary and unvalidated, must not be
altered, must be attributed, and analyses must say the data are preliminary. Whether credential-free file products at
files.airnowtech.org cover Nepal is **[unverified]**.

**OpenAQ.** Aggregator of openly licensed data; v3 API needs a registered API key (https://docs.openaq.org/using-the-api/api-key),
free tier 60 requests/minute and 2,000/hour (https://docs.openaq.org/using-the-api/rate-limits); terms require attribution and
forbid scraping (https://docs.openaq.org/about/terms); each location carries its source licence; it ingests only openly licensed
concentration data (https://docs.openaq.org/about/faq). A bulk AWS S3 route exists (FAQ) and may need no key [unverified].
**Which Nepal locations OpenAQ holds, and how many lie within 100 km of Belauri, I could not verify** without an API call.
It is not credential-free for the API, so I did not query it.

**PurpleAir.** Low-cost optical sensors; API needs a developer account and keys, points-based
(https://community.purpleair.com/t/about-the-purpleair-api/7145); terms restrict use to internal development, forbid
redistribution of data or derivatives and contain a grant-back clause for improvements
(https://www.purpleair.com/policies/terms-of-service). Sensors near Belauri: **[unverified]** (not queried). Low-cost
optical PM sensors require humidity and local calibration against reference instruments; I did not find a calibration source
for this area. These terms conflict with the project's open, reproducible, publishable-data practice.

**Other possibly relevant, not audited:** Indian CPCB stations across the border (south and west of Kanchanpur), satellite
surface-PM products, ICIMOD or Swachh Hawa low-cost sensor deployments in Dhangadhi (https://www.cen.org.np/clean-air-and-urban-mobility/swachh-hawa-pariyojana.html
lists planned 12-month deployments in seven provincial capitals; whether Dhangadhi data are public is unknown).

## 2. Scientific defensibility of a held-out station test

PRAVAH is a synthetic 2-D depth-averaged passive tracer (`configs/benchmark-v1.yaml` scope note: "not PM2.5"). A comparison with
station PM2.5 faces these problems:

1. **Species and scale.** PM2.5 is emitted, chemically formed, deposited and boundary-layer mixed; the tracer has none of
   these and no absolute scale. Any comparison needs a fitted scale/offset, which absorbs the very quantities in question.
2. **No source inventory.** Real emission rates are unknown. If sources are fitted to the observations, the model is fitted
   not tested; if they are assumed, the result tests the assumption.
3. **Few stations.** Two stations within 100 km, so a "block by station" split leaves one training and one test station:
   held-out error has no meaningful spread, and the two are probably strongly correlated (both urban, 23 km and 29 km from
   Belauri, shared regional haze; correlation not measured).
4. **Data quality.** Record gaps (Bhimdatta about five months in 2021, nothing current), unknown calibration, 80%
   completeness rule, extreme values up to 967 ug/m3, and a regional background (crop-residue and Indo-Gangetic transport)
   that is outside any plausible box.
5. **Circularity.** Choosing wind period, scale and source mask after seeing the same stations would leak.

What is defensible, if data were obtained under permitted terms: a **point-forecast comparison** of a statistical or tracer-based
predictor against persistence and climatology baselines at withheld times at one station (temporal bias, RMSE, skill vs
persistence), reported as a proxy-mismatch exercise. It cannot validate the field, the source maps, the placement rule or the
identifiability claims, and the plan itself anticipates this. With two partially covered stations, even this is
underpowered.

## 3. Conclusion and recommendation

- A held-out station test is **not scientifically defensible as a validation of PRAVAH** with the data seen. At best, a
  one-station point-forecast comparison against baselines is possible and should be labelled "proxy mismatch, not
  validation".
- **Recommended outcome: E18 recorded as "not conducted: gap assessment"** (plan permits this), with the audit table above as
  the deliverable. Do not substitute synthetic data. Keep the figure of what is missing: a collocated, calibrated, multi-site
  PM2.5 network with a known emission inventory and wind observations.
- Optional follow-up only if the user wants it: (i) user-approved OpenAQ key to count Nepal locations and to find a public
  multi-site urban benchmark elsewhere (also a plan option), (ii) read the DoEnv portal download terms and test whether a
  Bhimdatta/Dhangadhi CSV is retrievable without login, no data used beyond a completeness count.

## 4. Decisions for the user

| ID | Decision | Recommendation |
|---|---|---|
| O1 | Accept "not conducted: gap assessment" for E18 | yes |
| O2 | Allow me to open DoEnv portal download pages (read terms, count available periods, no analysis data) | yes, low cost; I will stop at any login |
| O3 | Create/provide an OpenAQ API key for a coverage count (account creation is yours) | optional; only if an alternative multi-site benchmark is wanted |
| O4 | PurpleAir | no: terms conflict with open reporting |
