# Belauri geography: what was acquired, what it means, what it does not

Real geography; synthetic emissions and meteorology. Every number below comes from `data/belauri/manifest.json` or
`results/geography/geography_report.json` (written by `scripts/geo_report.py`); nothing is typed in by hand.

## 1. Reproduce

```bash
python -m pravah geo fetch   --config configs/belauri_geo.json      # network; cached, pinned, sha256-recorded
python -m pravah geo prepare --config configs/belauri_geo.json      # offline: raw -> data/belauri/processed + manifest.json
python scripts/geo_report.py --manifest data/belauri/manifest.json --out results/geography   # offline, deterministic
pytest tests/test_geo.py tests/test_geo_real.py                     # synthetic unit tests + real-data integrity checks
```

Outputs of the report: `fig1_domain_map.png` (UTM km map, lon/lat on the top/right edges), `fig2_dem_hillshade.png`,
`fig3_profiles.png` (N-S and E-W through the polygon centroid), `fig4_slope_valid.png`, `fig5_source_hypotheses.png`
(each hypothesis shows its own sum f*A = 1) and `geography_report.json`.

## 2. Provenance and licences

| Dataset | Release / vintage | Retrieved (UTC) | Licence | sha256 (prefix) |
|---|---|---|---|---|
| geoBoundaries gbOpen NPL ADM3 (study polygon; source Open Data Nepal / MoFAGA, data year 2019) | build 2023-12-12, commit 9469f09 | 2026-10-10 15:40 | CC BY-SA 4.0 | 139e8622ca46 |
| geoBoundaries gbOpen NPL ADM2 (parent-district check only) | build 2023-12-12 (data year 2006) | 2026-10-10 15:40 | public domain | 83e197db0b72 |
| geoBoundaries gbOpen NPL ADM0 (national border) | build 2023-12-12 (data year 2019) | 2026-10-10 15:40 | CC BY 4.0 | fc0f32802e64 |
| geoBoundaries gbOpen IND ADM0 (neighbour outline) | build 2023-12-12 (data year 2014) | 2026-10-10 15:40 | CC0 | 77fb5cb03be9 |
| Nominatim search (independent locator; geometry unused) | live query | 2026-10-10 15:40 | ODbL 1.0 | 20a138daed34 |
| OSM roads / buildings / landuse / amenity via Overpass | live query (mirror, see 5) | 2026-10-10 16:56 | ODbL 1.0 | 43ead5e2e2b5 / dd07bb0ab2d1 / c2785f9ed89b / 896a4a9eba49 |
| Copernicus GLO-30 DSM tile N28_E080 (windowed read) | tile `Copernicus_DSM_COG_10_N28_00_E080_00_DEM` | 2026-10-10 17:07 | Copernicus licence (free, attribution) | 35e3176e19b0 |

Full URLs, release records and full hashes are in the manifest (`datasets`); `tests/test_geo_real.py` re-hashes every raw and
processed file. Attribution required by the licences (OSM contributors; "Contains modified Copernicus DEM GLO-30 data";
geoBoundaries / Open Data Nepal) is carried in the manifest and must go into any publication. CC BY-SA and ODbL are
share-alike: derived data we redistribute inherit those terms.

## 3. Boundary verification

The study polygon is geoBoundaries ADM3 `Belauri` (shapeID 92635248B27542858320702). `boundary_status` = `verified` because
all required checks pass:

| Check | Result |
|---|---|
| name, shapeID | match |
| geometry valid (projected) | yes |
| area | 122.96 km^2 (expected 123 +/- 10 %) |
| bbox | max deviation 0.000 deg (tol 0.03) |
| OSM administrative relation (R5927493) agreement | IoU 0.9384; Hausdorff 238 m; area 122.96 vs 122.89 km^2 (0.05 %) |
| Nominatim locator | name-matching OSM relation at 0.0 km |
| parent district | see below |
| ADM2 geometric coverage (informational) | **fails**: 71.7 % of the polygon lies inside ADM2 `Kanchanpur` |

The ADM2 failure is shown, not hidden. Likely cause (inference from the release years, not verified): the ADM2 layer is the 2006 vintage, the ADM3 layer the 2019 one; they were drawn
against different base data, so a municipality on the border cannot be 100 % covered by the district polygon. It is therefore
marked informational. The parent-district requirement is instead satisfied by independent evidence: the OSM relation agrees
(IoU 0.9384), the hierarchy names Kanchanpur, and the largest overlap with any other district is 0.00 % (must be <= 1 %).
Remaining difference between the two polygons (IoU 0.94, Hausdorff 238 m) is boundary-drawing detail.

## 4. CRS, axes, grids

* Geographic data are WGS84 (EPSG:4326) with **(lon, lat)** order; the projected CRS is **EPSG:32644** (WGS84 / UTM 44N)
  with (x = easting, y = northing). pyproj always uses `always_xy=True`. All metric work is in UTM, never in degrees.
* Local simulation coordinates = UTM minus the domain SW corner (x0 = 419399.65, y0 = 3158725.03); arrays are `[y, x]` with y
  increasing northward. Round-trip WGS84 -> UTM -> WGS84 error at the polygon centroid: 0.0 m (recomputed).
* Domain: municipality bounding box + 8 km buffer, padded to a multiple of 500 m: 33.0 x 32.5 km. Two nested grids share the
  domain: 500 m (66 x 65 cells, 584 polygon cells) and 250 m (132 x 130, 2147 polygon cells).
* Study mask = exact polygon-cell intersection fractions. Mask area equals polygon area to relative error 1e-15 at both
  resolutions (122.9556 km^2).
* Native DEM pixels are 1 arc-second, about 27.1 m (E-W) x 30.7 m (N-S). A 30 m DEM does **not** imply 30 m atmospheric
  accuracy: cells are 500/250 m means, and flow, mixing and emissions are not known to anything like that precision.
* DEM nodata: 0 of 1,614,830 raw pixels; valid fraction is 1.000 in every cell at both resolutions (no cell below 0.95).

## 5. Acquisition notes (honest)

The main Overpass endpoint (`overpass-api.de`) was unavailable during acquisition. The fetch step is configured with a mirror
list and fell back to `maps.mail.ru/osm/tools/overpass/api`; **all four OSM layers come from that mirror**, queried 16:56 UTC.
A mirror may lag the main database, so OSM content may differ slightly from the main-server state of the same day. The
geoBoundaries files are pinned to commit 9469f09 and re-read from cache on later runs.

## 6. Relief: what the data show

All values are m a.s.l. from the Copernicus **DSM** (surface model: canopy and buildings are included, so it is not bare-earth
terrain). At 500 m cells:

| | min | mean | max | range |
|---|---|---|---|---|
| inside the polygon (area weighted mean) | 167.7 | 177.4 | 204.4 | 36.7 m |
| over the whole domain | 161.9 | 183.3 | 241.3 | 79.4 m |
| native 30 m pixels inside the polygon | 167.0 | 177.4 | 218.4 | 51.4 m |

* The study area is very flat: mean slope in the polygon 0.21 deg (0.31 deg at 250 m); the steepest cell anywhere in the
  domain is 2.4 deg (500 m) / 4.7 deg (250 m). Coarse cell means understate native slopes.
* A plane fit to the domain rises 0.97 m/km northward and 0.22 m/km eastward: about 32 m across the domain, steepest ascent
  toward NNE (bearing 13 deg). Mean elevation by southern / middle / northern third of the domain: 174.4 / 180.2 / 195.9 m
  (N minus S = 21.5 m); within the polygon 172.7 / 177.6 / 181.8 m (9.0 m).
* The cells above the 90th percentile (204.1 m) are centred 11.0 km NNE (bearing 30 deg) of the polygon centroid; only 0.2 %
  of them lie inside the polygon. The domain maximum (241.3 m) is 19.3 km NE of the centroid **on the domain edge**.
* So there is no mountain wall in the domain: the whole relief is a few tens of metres. Terrain rises toward the
  north-north-east and keeps rising at the domain edge; whether Siwalik or higher terrain lies beyond cannot be read from this DEM
  window (it extends only a small margin past the domain). This is a statement about the data we hold, not a claim about
  the region beyond.

## 7. Border: Belauri touches the India border

* The municipality polygon touches the Nepal ADM0 boundary (polygon-to-border distance 0; centroid-to-border 3.14 km, nearest
  point WSW, bearing 255 deg). 37 % of the 72.5 km perimeter lies within 50 m of that border (south 19.1 %, west 18.0 %,
  east 0.3 %, north 0 %). Bounding-box sides at distance 0: west, east, south.
* Caution: the ADM0 line and the ADM3 polygon come from the same lineage (Open Data Nepal), so their coincidence along the
  border says nothing about the true border position.
* The India ADM0 outline in geoBoundaries is coarse (2014 data, 2,082 vertices against 66,082 for Nepal). Near the
  domain the Nepal border points lie on average 5.2 km (max 11.0 km) from the Indian outline, and 12.2 % of the Belauri
  polygon falls inside that Indian outline. It cannot be used to decide which country a point near the border lies in.
* Origin classes: a point is "international" only if it lies inside a neighbour ADM0 polygon. The external patch centre
  (80.2836 E, 28.6836 N, 2.92 km from the municipality edge) lies in **no** ADM0 polygon: outside the Nepal ADM0 polygon (whose agreement with the true border is not independently verified), but the
  coarse Indian outline does not contain it either. `prepare` therefore labels it `external_outside_home` (outside the
  home country, near the shared border, containment in a neighbour not asserted). This is deliberately not a claim that any emitter is Indian.

## 8. OSM content

Study area (manifest counts): roads 1,025 km inside the polygon (2,207 km in the domain, 8,206 segments); 50,510 buildings
and 1,020 residential-landuse polygons in the domain (6.89 km^2 built-up in the polygon); 98 facilities (schools and health)
in the domain, 60 inside. **Completeness is unknown**: absent roads or buildings are mapping gaps, not zero sources. Fig. 1 shows
visibly sparser OSM roads in parts of the northern municipality.

## 9. Synthetic source hypotheses

Road length, built-up area, facility count (all clipped to the polygon, group `local`) and a Gaussian external patch
(centre 4 km beyond the municipality edge toward the border, sigma 1 km, group `external`) are normalised shapes f with
sum f*A = 1 at both resolutions. They are **synthetic spatial priors on real geometry**: the geometry is real, the emission
pattern is assumed. They are not an emissions inventory and carry no emission rates. At 500 m the share of source mass
inside the polygon is road 0.949, built-up 0.977, facility 0.968 (boundary cells count by area fraction) and external patch 0.0017.

## 10. What each layer does in the simulations

| Layer | Role |
|---|---|
| municipality polygon | study mask (local vs external origin) and metric weights; never a PDE wall |
| roads, buildings/landuse, facilities | source shapes and candidate-site constraints |
| external patch | external source of the transboundary tests |
| DEM, hillshade, slope | **context only: not used by the PDE** |

Terrain dynamics are **unresolved**: no terrain-following flow, no orographic blocking, no slope winds, no valley
channelling, no terrain-dependent mixing depth. The relief here is small (tens of metres), so this is a modest omission for
the domain itself, but it is an assumption, not a result. To go further one needs a mesoscale or local wind field (WRF or a CFD
simulation, or measured winds) as input to the advection term; a variable mixing depth requires reformulating the
conservation law (depth-integrated form with h(x, y, t)), not just a new input array.

## 11. Blocked / incomplete: what is not verified

* **No official survey-department polygon was obtained.** The study polygon is geoBoundaries ADM3, derived from Open Data
  Nepal / MoFAGA 2019 government data. The independent check is the OSM administrative relation (IoU 0.9384, Hausdorff
  238 m). If the requirement is read as "authoritative boundary from the Survey Department of Nepal", it is only **partly met**:
  the boundary is verified by agreement between geoBoundaries (government-derived) and OpenStreetMap (one independent source; the OSM relation and the Nominatim hierarchy are both OpenStreetMap), not against an official survey product.
* The ADM2 geometric coverage check fails (71.7 %); explained in 3, not resolved.
* Indian boundary geometry is coarse (see 7); the external patch cannot be classified as inside India.
* OSM completeness is unknown (8); the OSM layers come from an Overpass mirror (5).
* The DEM is a surface model with 30 m pixels averaged to 500/250 m; no vertical-accuracy validation against ground points was done.
* The Nepal ADM0 border position itself was not verified independently of the same data lineage.
* Terrain effects on wind and mixing are not modelled (10); no real meteorology or emissions data were used.


## Where the northern relief actually is (wider bounded DEM strip)

The prepared DEM window only covers the simulation domain (about +-16 km of the centroid), which shows tens of metres of
relief and no mountain wall. To locate the higher terrain, `scripts/geo_relief_profile.py` samples a thin north-south and
east-west strip of the same Copernicus GLO-30 DSM (30 m pixels sampled every 500 m, bounded public tile reads, run RUN-0021,
`results/geography/relief_profile.json` and `relief_profile.png`). Elevations are DSM heights a.s.l.; they are context only and
do not enter the PDE.

* Median elevation within 8 km of the municipality centroid: 178 m.
* Terrain north of the centroid first exceeds that level by 50 m at 28.5 km, by 100 m at
  31.5 km, by 200 m at 34.5 km, by 300 m at
  35.5 km and by 500 m at 37.0 km.
* The highest point on the strip is 1947 m at 77.5 km north of the centroid.
* The municipality spans about 16 km north-south, so the first substantial relief begins roughly 20 km beyond its northern
  edge and well outside the simulated domain. The simulations therefore see an essentially flat surface; any
  mountain-valley circulation, blocking or channelling by the higher terrain to the north is NOT represented and would need a
  wind field from a mesoscale/CFD model or measurements (see the terrain-dynamics section).
