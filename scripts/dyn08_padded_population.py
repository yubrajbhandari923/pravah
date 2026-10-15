"""DYN-08 input: WorldPop people-per-cell on the PADDED Belauri grids (configs/belauri_padded_geography), by exact conservative regrid
of the standard 250 m extract (``regrid_counts``).  Local, offline, writes only data/population/belauri_padded_worldpop_<res>m.npy and
belauri_padded_worldpop.json.  The padded 500 m / 250 m cell edges fall on standard 250 m edges (offsets 500 m in x, 750 m in y), so the
regrid is exact (checked here: totals conserved, 2 x 2 sum of the padded 250 m field equals the padded 500 m field).

    .venv/bin/python scripts/dyn08_padded_population.py [--check-only]

A modelled 2025 population ESTIMATE (WorldPop R2025A, constrained), not a census.  The study total is the sum over the padded study mask
(counts x study_fraction, fractional people kept), NOT the grid total (184 029.57): see docs/dyn08_plan.md section 5 (EXP-026 amendment).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from pravah.geo.prepare import load_region
from pravah.population import mask_to_study, regrid_counts

ROOT = Path(__file__).resolve().parent.parent


def build(std_manifest: str = "data/belauri/manifest.json", pad_manifest: str = "configs/belauri_padded_geography/manifest.json",
          src_dir: str = "data/population") -> tuple[dict[float, np.ndarray], dict]:
    std, pad = load_region(ROOT / std_manifest), load_region(ROOT / pad_manifest)
    src = np.load(ROOT / src_dir / "belauri_worldpop_2025_CN_250m.npy")
    out: dict[float, np.ndarray] = {}
    rec: dict = {"source_extract": "belauri_worldpop_2025_CN_250m.npy", "method": "regrid_counts (conservative area overlap) from the standard 250 m extract"}  # noqa
    for res in (500.0, 250.0):
        counts, rep = regrid_counts(src, std.level(250.0).grid, pad.level(res).grid)
        sf = np.asarray(pad.level(res).arrays["study_fraction"], dtype=float)
        out[res] = counts
        rec[f"{int(res)}"] = {"grid": pad.level(res).grid.__dict__ | {}, **{k: float(v) for k, v in rep.items()}, "grid_total": float(counts.sum()),
                              "study_polygon_total": float(mask_to_study(counts, sf, float((counts * sf).sum())).sum()),
                              "nan_cells_in_study": int(np.isnan(counts[sf > 0]).sum()),
                              "sha256": hashlib.sha256(np.ascontiguousarray(counts).tobytes()).hexdigest()}
    agg = out[250.0].reshape(136 // 2, 2, 136 // 2, 2).sum(axis=(1, 3))
    rec["check_250_aggregates_to_500_max_abs"] = float(np.abs(agg - out[500.0]).max())
    rec["check_conserved_abs"] = float(abs(out[500.0].sum() - np.nansum(src)))
    if rec["check_250_aggregates_to_500_max_abs"] > 1e-9 or rec["check_conserved_abs"] > 1e-6:
        raise AssertionError(f"padded population regrid is not exact: {rec}")
    return out, rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-only", action="store_true", help="compute and verify; write nothing")
    ap.add_argument("--out", default="data/population")
    a = ap.parse_args()
    arrays, rec = build()
    print(json.dumps(rec, indent=1, default=str))
    if a.check_only:
        return
    o = ROOT / a.out
    for res, arr in arrays.items():
        np.save(o / f"belauri_padded_worldpop_{int(res)}m.npy", arr)
    (o / "belauri_padded_worldpop.json").write_text(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    main()
