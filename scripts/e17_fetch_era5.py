"""E17: fetch the ERA5 single-level winds for the Belauri box from the Copernicus CDS and write the manifest.

  uv run python scripts/e17_fetch_era5.py            # skips months already on disk; rewrites the manifest
The CDS key is read by cdsapi itself from ~/.cdsapirc; this script never reads, prints or stores it.
Raw files go to RAW (gitignored, shared with the main checkout) and are never modified after download.
"""
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PRAVAH_E17_RAW", root / ".cache/e17/raw"))  # set it in a worktree
MANIFEST = root / "docs/research/data/e17-manifest.json"
DATASET = "reanalysis-era5-single-levels"
# proposed box 79.80-80.82 E, 28.34-29.24 N, padded by >= one 0.25 deg cell and snapped to the 0.25 deg grid; [N, W, S, E]
AREA = [29.5, 79.5, 28.0, 81.25]
MONTHS = [(2021, 7), (2021, 11), (2021, 12), (2022, 1), (2022, 2)]
VARIABLES = ["10m_u_component_of_wind", "10m_v_component_of_wind", "boundary_layer_height"]
LICENCE = ("CC-BY 4.0 (replaced the 'Licence to use Copernicus Products' in the CDS on 2025-07-02, ECMWF forum announcement "
           "13464). Attribution (dataset References section / former licence 5.1.2-5.1.3 wording): \"Contains modified Copernicus "
           "Climate Change Service information [2021-2022]. Neither the European Commission nor ECMWF is responsible for any "
           "use that may be made of the Copernicus information or data it contains.\" Cite: Hersbach, H. et al. (2023): ERA5 "
           "hourly data on single levels from 1940 to present. Copernicus Climate Change Service (C3S) Climate Data Store "
           "(CDS), DOI: 10.24381/cds.adbb2d47.")


WORLDPOP_URL = "https://data.worldpop.org/GIS/Population/Global_2000_2020_1km_UNadj/2020/NPL/npl_ppp_2020_1km_Aggregated_UNadj.tif"
WORLDPOP = dict(
    dataset="WorldPop unconstrained individual countries 2000-2020 UN adjusted (1km resolution), Nepal, 2020",
    licence="CC BY 4.0 (https://www.worldpop.org/faq). Cite worldpop.org; dataset DOI 10.5258/SOTON/WP00671 [from memory, verify].",
    use="Population weights / candidate and source masks only; covers Nepal only (India part of the box is NoData).")


def request(y, m):
    days = (dt.date(y + m // 12, m % 12 + 1, 1) - dt.date(y, m, 1)).days
    return dict(product_type=["reanalysis"], variable=VARIABLES, year=[f"{y}"], month=[f"{m:02d}"],
                day=[f"{d:02d}" for d in range(1, days + 1)], time=[f"{h:02d}:00" for h in range(24)],
                area=AREA, data_format="netcdf", download_format="unarchived")


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    import cdsapi
    RAW.mkdir(parents=True, exist_ok=True)
    old = json.loads(MANIFEST.read_text()) if MANIFEST.exists() else {"files": []}
    known = {f["file"]: f for f in old["files"]}
    c = None
    files = []
    for y, m in MONTHS:
        name = f"era5_sl_belauri_{y}{m:02d}.nc"
        p, req = RAW / name, request(y, m)
        if not p.exists():
            c = c or cdsapi.Client(quiet=True)
            tmp = p.with_suffix(".part")
            c.retrieve(DATASET, req, str(tmp))  # licence not accepted -> cdsapi raises; we stop, never accept anything
            tmp.rename(p)
            p.chmod(0o444)  # immutable raw product
            retrieved = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        else:
            retrieved = known.get(name, {}).get("retrieved_utc", "unknown (file pre-existed)")
        files.append(dict(file=name, request=req, retrieved_utc=retrieved, bytes=p.stat().st_size, sha256=sha256(p)))
        print(name, files[-1]["bytes"], files[-1]["sha256"][:12], flush=True)
    wp = RAW / "worldpop" / WORLDPOP_URL.rsplit("/", 1)[1]
    if not wp.exists():  # plain HTTPS, no registration
        import urllib.request
        wp.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(WORLDPOP_URL, wp)
        wp.chmod(0o444)
        known[wp.name] = dict(retrieved_utc=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"))
    wpe = dict(WORLDPOP, url=WORLDPOP_URL, file=f"worldpop/{wp.name}", bytes=wp.stat().st_size, sha256=sha256(wp),
               retrieved_utc=known.get(wp.name, old.get("worldpop", {})).get("retrieved_utc", "2026-11-14T14:23:19Z"))
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(dict(
        dataset=DATASET, provider="Copernicus Climate Change Service (C3S) Climate Data Store, via cdsapi",
        raw_location="<project>/.cache/e17/raw/ (gitignored; immutable after download)", licence=LICENCE,
        note="Synthetic-tracer forcing only (E17): 'synthetic tracer under ERA5 winds, not validation'.",
        files=files, worldpop=wpe), indent=1) + "\n")


if __name__ == "__main__":
    sys.exit(main())
