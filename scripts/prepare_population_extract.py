"""Cut a gridded population raster (people per cell, EPSG:4326) down to the Belauri simulation grids and record its provenance.

Input (downloaded separately, untrusted data; read with `python -I`): WorldPop Global 2 R2025A, Nepal, 2025, 100 m, constrained (people per 100 m cell):
  https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/NPL/v1/100m/constrained/npl_pop_2025_CN_100m_R2025A_v1.tif

Writes to data/population/ (small, tracked): belauri_worldpop_2025_CN_<res>m.npy (people per simulation cell, shape (ny, nx), one per prepared
mesh level) and belauri_worldpop_2025_CN.json (provenance + totals).  Allocation is by source-cell centre (shortcut in population.counts_from_lonlat: fine
for 100 m source cells on 250/500 m cells); people outside the grid and in nodata cells are reported, not imputed.  A population ESTIMATE, not a census.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pravah import population as P
from pravah.belauri import resolve_manifest
from pravah.geo.prepare import load_region

ROOT = Path(__file__).resolve().parent.parent
URL = "https://data.worldpop.org/GIS/Population/Global_2015_2030/R2025A/2025/NPL/v1/100m/constrained/npl_pop_2025_CN_100m_R2025A_v1.tif"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tif", required=True)
    ap.add_argument("--manifest", default="data/belauri/manifest.json")
    ap.add_argument("--out", default="data/population")
    a = ap.parse_args()
    tif = Path(a.tif)
    sha = hashlib.sha256(tif.read_bytes()).hexdigest()
    region = load_region(resolve_manifest(a.manifest))
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    totals: dict[str, dict[str, float]] = {}
    for res, lvl in sorted(region.levels.items()):
        counts, rep = P.counts_from_geotiff(str(tif), lvl.grid)
        np.save(out / f"belauri_worldpop_2025_CN_{int(res)}m.npy", counts)
        sf = np.asarray(lvl.arrays["study_fraction"], dtype=float)
        totals[f"{int(res)}"] = {**{k: float(v) for k, v in rep.items()}, "grid_total": float(np.nansum(counts)),
                                 "study_polygon_total": float(np.nansum(counts * sf)), "cells": int(counts.size)}
    rec = P.dataset_record(
        source="WorldPop Global 2 (R2025A), Nepal, 2025, constrained, 100 m", url=URL, version="R2025A v1 (file dated 2025-08-04)",
        license="CC BY 4.0 (WorldPop; verify on the dataset page before redistribution)",
        attribution="WorldPop, University of Southampton (www.worldpop.org); cite the R2025A release", original_resolution="3 arc-seconds (~100 m)",
        crs="EPSG:4326", units="people/cell", sha256=sha, resampling="allocation by source-cell centre onto the simulation cells (counts conserved inside the grid)",
        limitations="Modelled estimate (top-down disaggregation of census-based totals), not census ground truth; constrained to building footprints; "
                    "residential location only (no movement, indoor or time-activity exposure); sum over the study polygon uses the cell area fraction")
    rec["totals_by_level"] = totals
    (out / "belauri_worldpop_2025_CN.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps(totals, indent=2))


if __name__ == "__main__":
    main()
