# Population data: sources, units and how to plug a raster in

Scope: what real population rasters could feed the population-aware design (EXP-016/017), what their numbers mean, and what `src/pravah/population.py` does with
them. **No real population raster is used anywhere in this repository yet**; every result so far (EXP-017, F-023) uses the five synthetic fixtures. Source facts below
were read from the providers' pages on 2026-10-11 (URLs given); items I could not verify are listed at the end. No data was downloaded.

## 1. Units: people per cell, not density

`population.py` works in **counts = people per cell** on a `geo.regionio.GridSpec` (UTM metres, arrays `[y, x]`). Counts are extensive (the sum is a head count and survives
aggregation); densities (people/km2) are intensive (they do not add up). Only `counts_to_density` and `density_to_counts` convert (cell area `dx*dy / 1e6` km2). Nothing else accepts a density.
`dataset_record(...)` refuses units other than `people/cell` and `people/km2`; record what the SOURCE file contains, then convert explicitly. Mixing the two is the most likely silent error:
a density raster summed as if it were counts is off by the cell area in km2 (0.0081 km2 for a 90 m cell, 0.25 km2 for 500 m).

## 2. Candidate sources

### WorldPop Global 2 (2015-2030 gridded population)
* What: modelled population, "Random Forest-based dasymetric redistribution" of census/administrative counts using covariates; WorldPop (University of Southampton), Global Demographic Data project
  funded by the Bill and Melinda Gates Foundation. "Constrained" estimates (counts only where settlement/buildings are mapped), release R2025A v1 (alpha version, "may change over the coming year"),
  100 m (3 arc-seconds) and 1 km (30 arc-seconds) products per country and as global mosaics, 2015-2030. The older series (2000-2020, "unconstrained", 100 m and 1 km, country totals optionally
  adjusted to UN estimates) is a different product with different semantics: record which one you used.
* Units: "number of people per pixel" (counts). CRS: geographic WGS84 (EPSG:4326). Resolution: 3 arc-seconds is about 92 m north-south and about 81 m east-west at Belauri (28.7 N; derived).
* Licence: Creative Commons Attribution 4.0 International (stated on every hub page). Attribution: cite the dataset DOI and the suggested citation, e.g. for the 100 m country series R2025A:
  Bondarenko M., Priyatikanto R., Tejedor-Garavito N., Zhang W., McKeen T., Cunningham A., Woods T., Hilton J., Cihan D., Nosatiuk B., Brinkhoff T., Tatem A., Sorichetta A. (2025). Constrained estimates of
  2015-2030 total number of people per grid square at a resolution of 3 arc (approximately 100m at the equator) R2025A version v1. WorldPop, University of Southampton. DOI 10.5258/SOTON/WP00839.
  (Take the DOI and citation text from the page of the exact file you download; the 1 km series has another DOI, 10.5258/SOTON/WP00845 for the global mosaic.)
* Where: https://hub.worldpop.org/geodata/listing?id=135 (100 m, individual countries 2015-2030 R2025A); https://hub.worldpop.org/geodata/listing?id=136 (1 km);
  release statement https://data.worldpop.org/repo/prj/Global_2015_2030/R2025A/doc/Global2_Release_Statement_R2025A_v1.pdf. Nepal 100 m summary page: https://hub.worldpop.org/geodata/summary?id=74555.
* Caveats: modelled, not a census; the spatial detail inside an administrative unit comes from covariates; constrained and unconstrained versions differ in which cells hold people; R2025A is labelled alpha.

### GHS-POP R2023A (Global Human Settlement Layer, JRC)
* What: residential population grid, "number of people per cell", census/administrative population from CIESIN GPWv4.11 disaggregated to cells using the built-up surface of the GHSL for each epoch.
  Epochs 1975-2020 in 5-year steps plus projections to 2025 and 2030. European Commission, Joint Research Centre.
* Units: absolute number of inhabitants per cell, stored as float (decimals). NoData value -200 (stated on the product page).
* Resolutions and CRS: 100 m and 1 km in World Mollweide (EPSG:54009); 3 arc-seconds and 30 arc-seconds in WGS84 (EPSG:4326). File names such as
  `GHS_POP_E2020_GLOBE_R2023A_4326_3ss_V1_0` (3 arc-seconds WGS84) and `..._54009_100_V1_0` (100 m Mollweide).
* Licence/attribution: dataset citation Schiavina M., Freire S., Carioli A., MacManus K. (2023). GHS-POP R2023A - GHS population grid multitemporal (1975-2030). European Commission, Joint Research Centre (JRC).
  PID http://data.europa.eu/89h/2ff68a52-5b5b-4a22-8f40-c41da8332cfe, doi 10.2905/2FF68A52-5B5B-4A22-8F40-C41DA8332CFE. Methodology: Pesaresi M. et al. (2024), Advances on the Global Human Settlement Layer
  by joint assessment of Earth Observation and population survey data, International Journal of Digital Earth 17(1), doi 10.1080/17538947.2024.2390454. Licence: see "not verified" below.
* Where: https://human-settlement.emergency.copernicus.eu/ghs_pop2023.php.
* Caveats: a disaggregation proportional to built-up volume, so population follows buildings (it can place people in non-residential buildings and none in sparse rural housing the built-up layer misses);
  it inherits the coarse input census units; the 1 km products are aggregates.

### Practical choice for this repository
`counts_from_geotiff` reads only **north-up EPSG:4326** rasters (`rasterio`, extra `geo`). Therefore use the WGS84 products: WorldPop 100 m, or GHS-POP `4326_3ss`. The Mollweide 100 m GHS-POP file raises
`ValueError("only EPSG:4326 rasters are supported; reproject first")`. Do not use 30 arc-second (about 900 m) rasters on a 500 m grid, see the caveat below.

## 3. What the code does with a raster, and its shortcuts

`population.py` (commit fe9e1da):

* `regrid_counts(counts, src, dst)`: **conservative** area-overlap regrid between axis-aligned grids in the SAME projected CRS; each source cell is split by overlap area (uniform within the cell);
  NaN cells are excluded from every total and contribute nothing (their number is reported); negative input raises. Returns `(out, report)` with `total_in`, `total_out`, `lost_outside`,
  `nodata_cells`. Totals are conserved except for people outside the destination grid.
* `counts_from_lonlat(arr, west, north, dlon, dlat, dst, nodata)`: for lon/lat rasters. **Shortcut: each source cell's people go whole to the destination cell containing the source cell CENTRE**
  (the docstring says so: `shortcut: ... not area-split`). Totals inside the destination grid are exact, but the allocation is only accurate when source cells are much smaller than destination cells.
  Consequences: a 3 arc-second source (about 80-90 m) into 500 m or 250 m cells is fine (several source cells per destination cell; boundary errors of at most a source cell); into 125 m cells it is marginal
  (about 1.5 source cells per side); a 30 arc-second source (about 900 m) into 500 m cells leaves destination cells EMPTY while others get a whole source cell: do not do this (upgrade to polygon overlap first).
* **Nodata handling.** If `nodata` is passed it is replaced by NaN, then NaN is turned into 0 people in the output and counted in `report["nodata_cells"]`. `counts_from_geotiff` passes the file's
  declared nodata only (`ds.nodata`): a file that stores its nodata value without declaring it (or uses a value other than the declared one) is NOT masked, and a large negative value would be summed into the counts.
  `counts_from_lonlat` does not reject negative values (only `regrid_counts` does): check `arr.min() >= 0` after masking, and compare `report["total_in"]` with the file's own total (the nodata count must be 0 or understood).
  Nodata cells inside the study polygon count as zero people, which biases population-weighted metrics low there; report the number.
* The destination `GridSpec` is the Belauri grid at the working resolution (`geo.region.level(res).grid`, EPSG 32644). It includes an 8 km buffer around the study polygon (`grid.buffer_m`), so counts outside
  the polygon exist. **The study polygon is a mask**: weights multiply counts by `study_fraction` (`pop_weighted_rmse(..., study_weights=)`); `mixed_weight` does not apply the mask itself. A Belauri-scale design must mask
  consistently (see `hpc-runbook.md` section 6).
* Time: the code uses one static raster. A population epoch (for example 2020) is a choice to record, not a time series.
* Fixtures (`FIXTURES`): uniform, one_town, separated_settlements, dense_core_sparse_surround, mixed_rural_urban, integer counts with exactly the requested total (largest-remainder allocation), seeded.
  They are synthetic and say nothing about Belauri.
* Population is a **weight** in the design objective only (`DesignOptions.weight`, `population.mixed_weight`); it does not enter the transport physics, and "population-weighted RMSE" is a concentration-error
  metric weighted by head count, not a personal exposure estimate.

## 4. Recording a dataset: `dataset_record`

Every real raster that is used must be recorded next to the results (e.g. in `provenance.json`), by calling `population.dataset_record` with ALL fields (the function does no network access and no defaulting
except the date):
```python
from pravah.population import dataset_record
rec = dataset_record(
    source="WorldPop Global 2 (constrained, individual countries)",   # product, not just the organisation
    url="<exact download URL of the file>",
    version="R2025A v1 (alpha); epoch 2020",                           # release + epoch
    license="CC BY 4.0",
    attribution="<suggested citation text from the page, with DOI>",
    original_resolution="3 arc-seconds (about 92 m x 81 m at 28.7 N)",
    crs="EPSG:4326",
    units="people/cell",                                               # or "people/km2": must match the SOURCE file
    sha256="<sha256 of the downloaded file>",
    resampling="counts_from_geotiff: cell-centre allocation to 500 m UTM 44N cells; conserves total inside the grid",
    limitations="modelled disaggregation; alpha release; nodata cells counted as 0 people; synthetic emissions and meteorology",
    download_date="2026-..-..")                                        # optional; defaults to today
```
`units` is validated; everything else is free text, so be specific (the exact file name and epoch go in `version` or `url`).

## 5. Plugging a real raster into the pipeline (what exists and what does not)

1. Download the file by hand (not part of this repo's fetch step; record URL, date and sha256). Keep it out of git (large; add to `.gitignore` if kept under `data/`).
2. Build the destination grid: `grid = geo.region.level(res).grid` from `pravah.geo.prepare.load_region("data/belauri/manifest.json")`.
3. `counts, report = population.counts_from_geotiff(path, grid)`; check `report["lost_outside"]` is small relative to `total_in` (the raster should cover the grid and its buffer), that
   `report["nodata_cells"]` is understood, and that `counts.min() >= 0`.
4. Mask to the study polygon: `counts_in = counts * study_fraction` (the study mask on the same level, `region.level(res).arrays["study_fraction"]`); report the in-study total against an independent figure
   (do not tune the raster to match one).
5. Weights: `w = population.mixed_weight(counts_in, area_in, beta)` with `area_in = cell_area * study_fraction` (pass `area` as the masked area map so the area term is also masked), then give `w` to
   `design.DesignOptions.weight`; metrics through `pop_weighted_rmse(err, counts, study_weights)`.
6. Not built: a Belauri driver that does steps 2-5 for a whole experiment (EXP-016, `hpc-runbook.md` section 6), reprojection of non-4326 rasters, polygon-overlap regridding for coarse lon/lat rasters,
   multi-epoch handling, and any rule for choosing between WorldPop and GHS-POP. If both are used, treat the difference between them as a sensitivity (H4 of EXP-016 is the misspecification test) and never choose
   the raster by looking at test results.

## 6. What I could not verify

* The GHSL (GHS-POP) licence text. The product page I read gives the citation and the data description but not the licence; I believe reuse is covered by the European Commission reuse policy
  (Commission Decision 2011/833/EU, usually described as CC BY 4.0), but that is not confirmed from the GHSL page: read the licence on the download page or in the GHSL Data Package 2023 report
  (https://human-settlement.emergency.copernicus.eu/documents/GHSL_Data_Package_2023.pdf) before redistributing.
* ~~The nodata value used inside WorldPop GeoTIFFs~~ **Checked on the real file (2026-10-11):** `npl_pop_2025_CN_100m_R2025A_v1.tif` (WorldPop Global 2 R2025A v1, 32 MB, file dated 2025-08-04) is float32,
  EPSG:4326, 0.00083333 deg (~100 m) cells, shape 4921 x 9773, declares `nodata = -99999.0`, and its TIFF header carries `Copyright: CC-BY-4.0`. Other WorldPop series and the GHS-POP files were NOT checked.
  The Belauri extract (people per simulation cell, 500 m and 250 m; `data/population/`, script `scripts/prepare_population_extract.py`, provenance JSON next to the arrays) gives
  184,030 people in the whole simulation domain and 54,370 (500 m) / 54,563 (250 m) inside the study polygon (cell-area-fraction weighted; the two meshes differ by 0.4% because of the polygon rasterisation). These are modelled estimates, not census counts; I did not compare them with a census figure.
* Whether the GHS-POP 3 arc-second files declare nodata -200 in the GeoTIFF header (the page states -200 as the legend value).
* The exact definition of "Global 2" beyond the hub pages (I used the "Global Demographic Data" 2015-2030 series, release statement title `Global2_Release_Statement_R2025A_v1.pdf`; I did not read the PDF).
* That the Nepal 100 m file for the year you need exists in the same release (the Nepal R2025A page lists an estimate for 2021; the year you want may be a separate file).
* Pixel size arithmetic at 28.7 N is mine (derived), not the providers'.
