"""Fetch a small ERA5 hourly wind extract around Belauri (Nepal) with the Copernicus CDS API.

Run this ON YOUR OWN MACHINE (needs your CDS account); it downloads a few hundred kilobytes.  Nothing here is tested against the live API:
the request format follows the CDS documentation for 'reanalysis-era5-single-levels' (verify at
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download if a request is rejected).

One-time setup:
  1. Log in at https://cds.climate.copernicus.eu , open the dataset page above, and accept the licence at the bottom of the download tab.
  2. Copy your API key from your profile page into ~/.cdsapirc :
         url: https://cds.climate.copernicus.eu/api
         key: <your-personal-access-token>
  3. pip install "cdsapi>=0.7"    (any Python; this script does not need the pravah environment)

Usage:
  python scripts/fetch_era5.py --year 2023 --months 1 7 --out data/era5/belauri_era5_10m_wind_2023.nc

ERA5 is 0.25 deg (~28 km) hourly reanalysis.  It cannot resolve local terrain-driven flow around Belauri; it is intended only as a bounded external
meteorology INPUT layer, entering through pravah.winds.segments_from_series, with that limitation disclosed.  Licence: Copernicus (CC-BY 4.0 style,
cite 'Hersbach et al. (2023), ERA5 hourly data on single levels from 1940 to present, C3S CDS, DOI 10.24381/cds.adbb2d47').
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path

# Belauri centre is about 28.69 N, 80.30 E; a +-1 deg box gives a 9 x 9 grid of 0.25 deg ERA5 cells (N, W, S, E order for CDS).
AREA = [29.75, 79.25, 27.75, 81.25]
VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind"]  # 10 m is the near-surface level; 100 m variants exist as 100m_*


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2023)
    ap.add_argument("--months", type=int, nargs="+", default=[1, 7], help="months to fetch (default: January and July)")
    ap.add_argument("--out", default="data/era5/belauri_era5_10m_wind.nc")
    a = ap.parse_args()
    import cdsapi  # imported late so --help works without it

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": ["reanalysis"],
        "variable": VARIABLES,
        "year": [str(a.year)],
        "month": [f"{m:02d}" for m in a.months],
        "day": [f"{d:02d}" for d in range(1, 32)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": AREA,
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", request, str(out))
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    meta = {"dataset": "reanalysis-era5-single-levels", "doi": "10.24381/cds.adbb2d47", "request": request, "file": out.name, "sha256": sha,
            "bytes": out.stat().st_size, "download_date": date.today().isoformat(),
            "notes": "ERA5 0.25 deg hourly; 10 m wind components in m/s; time in UTC; cannot resolve local mountain-induced flow"}
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out} ({meta['bytes']} bytes, sha256 {sha[:12]}...) and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
