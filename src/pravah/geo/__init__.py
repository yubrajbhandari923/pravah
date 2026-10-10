"""Belauri geography (optional extra ``pip install pravah[geo]``).

GIS libraries (geopandas, shapely, pyproj, rasterio, osmnx, requests) are imported lazily inside the functions that
need them, so ``import pravah.geo`` and the non-geo CLI never import them.

* ``regionio``  offline pure functions (CRS, grid, rasterisation, DEM aggregation, normalisation, candidate sites)
* ``fetch``     bounded, cached, polite network access (geoBoundaries, Nominatim, OSM via osmnx, Copernicus GLO-30)
* ``prepare``   offline: cached raw files -> processed arrays + ``manifest.json``; ``load_region`` for later stages
"""
