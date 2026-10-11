# Transfer to another municipality

Scope: what you change to run `run_geodesign` (and `python -m pravah benchmark`, experiment `geodesign`) on a region that is not Belauri.
Evidence: `tests/test_transfer.py` runs the REAL pipeline (smoke settings) on a synthetic L-shaped region (about 10 E / 33 S, UTM 32S, fake
home `ZZA`, fake neighbour `ZZB`, dense road grid in one arm and one road in the other, no facilities) and on Belauri, and asserts the same
output files and CSV column names. Everything in that fixture is synthetic; it shows the code path transfers, not that results are meaningful.

## (a) Region-independent
`config`, `forward`, `inverse`, `attribution`, `design`, `experiments`, `transboundary`, `indoor`, `sources`, `observations`, and in
`geodesign` everything after the geography is loaded: candidate flags, ensembles, alpha selection, greedy design, metrics, attribution, tables and
figures. `geodesign_robust` reads no manifest roles itself (not run on the new region). Edited for transfer: only `geodesign._load_raw`.

## (b) What a researcher supplies
1. **Raw files + `geo fetch` records** (or hand-built `fetch_records.json`, as the test does) for roles `study` (ADM3 GeoJSON), `adm2`, `dem`,
   `osm_roads`, `osm_buildings`, `osm_amenity`, optional `osm_landuse` / `nominatim` / `osm_relation`, and `adm0_<ISO>` for the home country
   (and a neighbour, if any).
2. **A prepare config** (cf. `configs/belauri_geo.json`): `study.geoboundaries {iso, level, shapeName, shapeID}`, `study.expected`
   (`parent_district`, `bbox_lonlat`, `area_km2`), `study.tolerances`, `expected_epsg`, `adm2`, `adm0` list, `dem`, `grid`
   (`buffer_m`, `resolutions_m`, `pad_to_m`), `osm`, `external_patch` (`bearing_deg` or `xy_m`/`lonlat`, `beyond_polygon_m`, `sigma_m`).
   `geo prepare` then writes the manifest (`georef.epsg`, `study.area_km2`, `datasets` with `role`/`kind`/`file`/`sha256`, `external_patch`).
3. **A geodesign config** = `configs/geodesign*.json` with only the region block changed:
   `region.manifest`, `region.study {shapeName, shapeID}`, `region.crop_buffer_m`, and optional
   `region.roles {home_adm0, neighbour_adm0, roads, amenity}` (defaults `adm0_NPL`, `adm0_IND`, `osm_roads`, `osm_amenity`).
   `neighbour_adm0: null` or a role absent from the manifest means no neighbour (`ind = None`, `inside_coarse_ind_outline` all False).
   `sha` / `raw_sha256` keys in the outputs are the resolved role names.
   Lattice spacings, budgets and `heldout.*` are numbers that must be sensible for the new polygon size (see c).

Output column names keep their Belauri spelling. `in_nepal_adm0` means "inside the HOME-country ADM0 polygon"; `inside_coarse_ind_outline`
means "inside the neighbour outline"; `near_border_km` is the distance to the HOME ADM0 boundary line, whether or not a neighbour exists (with no
neighbour it is a distance to the national outline, not to a shared border). `reference_hypothetical` sites are "outside the home ADM0".

## (c) Belauri assumptions NOT removed
* `scenarios.LOCAL_LAYERS = ("road", "builtup", "facility")` is fixed; the prepare step always builds those three weights plus `external_patch`.
* An `external_patch` in the manifest is mandatory (prepare raises if its centre is inside the polygon or off the grid); `crop_region` raises if the
  crop cuts it, so `crop_buffer_m` must exceed `beyond_polygon_m + 3 sigma` (the fixture needed `crop_buffer_m` 3000 m, `beyond` 1500 m, `sigma` 400 m).
  The patch also fails `check_origin_tags` if it leaks > 1 % into the polygon, so it must be placed well outside.
* The patch bearing `toward_border` needs a neighbour polygon in `adm0`; a region with no neighbour must give `xy_m`/`lonlat` or a numeric bearing (prepare with no neighbour was NOT tested).
* Study selection is by geoBoundaries `shapeName` + `shapeID` in the `study` raw file; other boundary sources need a file with those properties.
* `prepare` needs an ADM2 layer with a feature named `study.expected.parent_district`, and uses the manifest `area_km2` to 1e-6 relative.
* `scenarios.LOCAL_ORIGIN_TOL = 0.06` was calibrated to Belauri's boundary-cell leakage (0.949/0.977/0.968 of road/built-up/facility mass inside). Small or very
  thin polygons have more boundary cells and may exceed it. It held on the L-shaped fixture; with no facilities the facility layer is simply empty.
* `_load_raw` and `prepare` clip the country outlines to centroid +-3 degrees (fine for municipalities, wrong for huge study areas).
* `DEFAULT_FACILITIES` are OSM amenity tags typical for Nepal; override with `candidates.facility_amenities`.
* A DEM is required (hillshade, slope), covering the whole grid with >= `min_valid_fraction` valid pixels. Prepared grids are UTM only.
* `heldout.block` (lattice spacing 1000 m, offset [333, 217], 3 x 1 blocks, exclusion radius 2000 m) and `candidates.*` spacings were tuned on Belauri's shape and size.
  The unmodified smoke values happened to work on the L-shape; a much smaller or differently oriented polygon may give empty blocks or too few interior sites (errors are explicit).
* Names: `BelauriCase`, `belauri_scenario`, `belauri.py`, the `nepal` field of `Geography`, `in_nepal_adm0`, `ASSUMPTIONS` text mentioning "Nepal ADM0", figure titles, and the module docstring are Belauri/Nepal-specific labels (not renamed on purpose).
* Synthetic emission amplitudes, wind sectors, speeds and profiles in `configs/geodesign.json` are Belauri-scale placeholders, not site-specific.

## Not tested
Real data for a second municipality; `geo fetch`/`geo prepare` network paths; a region with no neighbour end to end (only `load_geography` and
`build_candidate_table` are tested with `neighbour_adm0: null`); `geodesign_robust`; full (non-smoke) budgets; southern-hemisphere vs. northern
differences beyond the UTM code path; population wiring (out of scope).
