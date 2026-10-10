"""Where does relief actually occur north (and in other directions) of Belauri? Bounded Copernicus GLO-30 strips.

    python scripts/geo_relief_profile.py --out results/geography [--north-km 80]

Reads thin windows (one pixel wide N-S and E-W strips through the municipality centroid, decimated) from the public
Copernicus GLO-30 COG tiles (no credentials), writes relief_profile.json/.png, and prints distances and elevations.
Context only: the DEM is a surface model and is not used by the PDE.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = "https://copernicus-dem-30m.s3.amazonaws.com"


def tile_url(lat: int, lon: int) -> str:
    n = f"Copernicus_DSM_COG_10_{'N' if lat >= 0 else 'S'}{abs(lat):02d}_00_{'E' if lon >= 0 else 'W'}{abs(lon):03d}_00_DEM"
    return f"{BASE}/{n}/{n}.tif"


def strip(points: list[tuple[float, float]]) -> np.ndarray:
    """Sample the DEM at (lon, lat) points (nearest 30 m pixel); NaN where nodata. Opens each needed tile once."""
    import rasterio
    out = np.full(len(points), np.nan)
    by_tile: dict[tuple[int, int], list[int]] = {}
    for i, (lon, lat) in enumerate(points):
        by_tile.setdefault((math.floor(lat), math.floor(lon)), []).append(i)
    for (la, lo), idx in by_tile.items():
        with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif"):
            with rasterio.open(tile_url(la, lo)) as ds:
                vals = list(ds.sample([points[i] for i in idx]))
                nod = ds.nodata
                for i, v in zip(idx, vals):
                    z = float(v[0])
                    out[i] = np.nan if (nod is not None and z == nod) else z
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default=str(ROOT / "data/belauri/manifest.json"))
    ap.add_argument("--out", default=str(ROOT / "results/geography"))
    ap.add_argument("--north-km", type=float, default=80.0)
    ap.add_argument("--step-m", type=float, default=500.0)
    a = ap.parse_args()
    man = json.loads(Path(a.manifest).read_text())
    lon0, lat0 = man["study"]["centroid_lonlat"] if "centroid_lonlat" in man["study"] else (80.3436, 28.7004)
    kx, ky = 111.320 * math.cos(math.radians(lat0)), 110.574  # km per degree (local, for sampling positions only)
    n = int(a.north_km * 1000 / a.step_m)
    ns = [(lon0, lat0 + (k * a.step_m / 1000) / ky) for k in range(-n // 4, n + 1)]
    ew = [(lon0 + (k * a.step_m / 1000) / kx, lat0) for k in range(-n // 2, n // 2 + 1)]
    zs_ns, zs_ew = strip(ns), strip(ew)
    d_ns = np.array([(p[1] - lat0) * ky for p in ns])   # km north of centroid
    d_ew = np.array([(p[0] - lon0) * kx for p in ew])
    base = float(np.nanmedian(zs_ns[np.abs(d_ns) < 8]))
    res = {"centroid_lonlat": [lon0, lat0], "source": "Copernicus GLO-30 DSM (public S3 COG tiles), 30 m, nearest pixel, step "
           f"{a.step_m:.0f} m", "median_elevation_within_8km_m": base,
           "north": {"km": d_ns.tolist(), "z_m": [None if np.isnan(z) else z for z in zs_ns]},
           "east_west": {"km": d_ew.tolist(), "z_m": [None if np.isnan(z) else z for z in zs_ew]}}
    for thr in (50, 100, 200, 300, 500):  # first distance north where terrain exceeds base + thr
        k = np.where((d_ns > 0) & (zs_ns > base + thr))[0]
        res[f"first_km_north_above_base_plus_{thr}m"] = float(d_ns[k[0]]) if k.size else None
    res["max_elevation_north_m"] = float(np.nanmax(zs_ns[d_ns > 0]))
    res["km_north_of_max"] = float(d_ns[np.nanargmax(np.where(d_ns > 0, zs_ns, np.nan))])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "relief_profile.json").write_text(json.dumps(res, indent=2) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(d_ns, zs_ns, lw=1)
    ax[0].axvspan(-8, 8, color="C1", alpha=0.15, label="~municipality extent (N-S)")
    ax[0].set(xlabel="km north of municipality centroid", ylabel="elevation (m, DSM)", title="North-south relief profile")
    ax[0].legend()
    ax[1].plot(d_ew, zs_ew, lw=1)
    ax[1].set(xlabel="km east of centroid", title="East-west relief profile")
    fig.suptitle("Real geography (Copernicus GLO-30 DSM); context only - not used by the PDE")
    fig.tight_layout()
    fig.savefig(out / "relief_profile.png", dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k not in ("north", "east_west")}, indent=1))


if __name__ == "__main__":
    main()
