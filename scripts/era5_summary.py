"""ERA5 2023 (Jan, Jul) 10 m wind at the Belauri centroid: descriptive tables + figure, and a bounded plumbing demo.
    python scripts/era5_summary.py [--out results/phase3-era5]
DESCRIPTIVE ONLY: two months of one year at 0.25 deg are not a climatology and say nothing about causes.
Directions are meteorological FROM bearings (deg clockwise from north), 16 sectors of 22.5 deg centred on 0, 22.5, ...
WSW sector = the one centred on configs/belauri_forward.json wind.reference (from_west_southwest, 247.5 deg), i.e. 236.25-258.75."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pravah import era5  # noqa: E402
from pravah import forward as F  # noqa: E402
from pravah.config import Boundary, Domain, GaussianSource, Scenario, Times, Transport  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
NC = ROOT / "data/era5/belauri_era5_10m_wind.nc"
STAMP = "ERA5 2023, 0.25 deg, descriptive only"
NAMES = "N NNE NE ENE E ESE SE SSE S SSW SW WSW W WNW NW NNW".split()
BINS = [0, 1, 2, 3, 5, np.inf]
BIN_LABELS = ["<1", "1-2", "2-3", "3-5", ">=5"]
MONTHS = {1: "Jan", 7: "Jul"}


def sector(deg):
    return (np.floor(((np.asarray(deg) + 11.25) % 360.0) / 22.5)).astype(int)


def runs(flag):
    """Lengths (hours) of maximal runs of equal values in a 1-D array."""
    idx = np.flatnonzero(np.diff(flag) != 0)
    return np.diff(np.concatenate([[0], idx + 1, [len(flag)]]))


def stats(x):
    return {"n": int(len(x)), "mean": float(np.mean(x)), "median": float(np.median(x)), "max": int(np.max(x))}


def demo(w, u, v, t):
    """Plumbing check: hourly-uniform wind, first 24 h of each month, small synthetic domain, pulse source."""
    d = Domain(30_000.0, 30_000.0, 30, 30)
    out = {}
    for m in MONTHS:
        sel = np.flatnonzero((t.astype("datetime64[M]") == np.datetime64(f"2023-{m:02d}")))[:24]
        ts = np.arange(24) * 3600.0
        segs = era5.to_segments(d, ts, u[sel], v[sel])
        src = GaussianSource("s", 15_000.0, 15_000.0, 1500.0, profile=((0.0, 1.0), (1800.0, 0.0)))
        scn = Scenario(domain=d, sources=(src,), transport=Transport(segs, 100.0, 1 / 21600, 500.0),
                       boundary=Boundary("open"), times=Times(24 * 3600.0, 300.0, 3600.0))
        res = F.simulate(scn, [1e6])
        b, c = res.budget, res.concentration.values
        assert b is not None
        emitted, removed, lost = (float(x.sum()) for x in (b.emission, b.removal, b.outflow - b.inflow))
        closure = abs(500.0 * d.dx * d.dy * c[-1].sum() - (emitted - removed - lost)) / emitted
        xs, ys = (np.arange(30) + 0.5) * d.dx, (np.arange(30) + 0.5) * d.dy
        chk = []
        for k in (1, 2):  # output k at t = k h; the pulse mass is still inside the box then
            ck = c[k]
            cx, cy = (ck.sum(0) * xs).sum() / ck.sum() - 15_000.0, (ck.sum(1) * ys).sum() / ck.sum() - 15_000.0
            px, py = (u[sel[:k]].sum() * 3600.0, v[sel[:k]].sum() * 3600.0)  # hold-last wind integral
            ang = float(np.degrees(np.arctan2(cx * py - cy * px, cx * px + cy * py)))
            chk.append({"t_h": k, "centroid_shift_m": [cx, cy], "wind_integral_m": [px, py], "angle_between_deg": ang,
                        "length_ratio": float(np.hypot(cx, cy) / max(np.hypot(px, py), 1e-9))})
        out[MONTHS[m]] = {"closure_rel": closure, "step_residual_rel": float(res.diagnostics["max_rel_budget_residual"]),
                          "min_conc": float(res.diagnostics["min_concentration"]), "checks": chk}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "results/phase3-era5"))
    out = Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    meta = json.loads(NC.with_suffix(".json").read_text())
    sha = hashlib.sha256(NC.read_bytes()).hexdigest()
    assert sha == meta["sha256"], "ERA5 file does not match its json checksum"
    lon0, lat0 = json.loads((ROOT / "data/belauri/manifest.json").read_text())["study"]["centroid_lonlat"]
    ref = json.loads((ROOT / "configs/belauri_forward.json").read_text())["wind"]
    wsw_from = ref["scenarios"][ref["reference"]]["from_deg"]
    wsw = int(sector(wsw_from))
    w = era5.load_era5_wind(str(NC))
    u, v = era5.sample_at(w, lon0, lat0)
    spd, frm = np.hypot(u, v), era5.from_deg(u, v)
    mon = w.times.astype("datetime64[M]").astype(int) % 12 + 1
    hour = w.times.astype("datetime64[h]").astype(int) % 24
    rose, diur, pers, head = [], [], [], {}
    for m, name in MONTHS.items():
        s = mon == m
        sec, sp = sector(frm[s]), spd[s]
        cnt = np.zeros((16, len(BIN_LABELS)), int)
        for a, b in zip(sec, np.digitize(sp, BINS[1:-1])):
            cnt[a, b] += 1
        for i in range(16):
            rose.append({"month": name, "sector": NAMES[i], "centre_deg": i * 22.5, **{f"n_{l}": int(cnt[i, j]) for j, l in enumerate(BIN_LABELS)},
                         "n": int(cnt[i].sum()), "percent": 100.0 * cnt[i].sum() / s.sum()})
        for h in range(24):
            x = sp[hour[s] == h]
            diur.append({"month": name, "hour_utc": h, "n": len(x), "mean_speed": x.mean(), "median_speed": np.median(x)})
        in_wsw = sec == wsw
        in_wide = (frm[s] >= 225.0) & (frm[s] < 270.0)
        # runs of constant 16-sector wind; and runs of consecutive WSW / non-WSW hours.  Series is hourly and gap-free inside a month.
        r_sec, r_flag = runs(sec), runs(in_wsw.astype(int))
        first_wsw = int(in_wsw[0])
        wsw_runs = r_flag[0 if first_wsw else 1::2]
        pers.append({"month": name, "hours": int(s.sum()), "wsw_sector": NAMES[wsw], "wsw_share_pct": 100.0 * in_wsw.mean(),
                     "wide_225_270_share_pct": 100.0 * in_wide.mean(), "sector_runs": stats(r_sec), "wsw_runs": stats(wsw_runs)})
        top = np.argsort(-cnt.sum(1))[:3]
        head[name] = {"dominant_sectors": [(NAMES[i], round(100.0 * cnt[i].sum() / s.sum(), 1)) for i in top],
                      "wsw_share_pct": round(100.0 * in_wsw.mean(), 1), "mean_speed": round(float(sp.mean()), 2),
                      "calm_lt_0.5_hours": int((sp < 0.5).sum())}
    pd.DataFrame(rose).to_csv(out / "wind_rose_tables.csv", index=False)
    pd.DataFrame(diur).to_csv(out / "diurnal_speed.csv", index=False)
    (out / "persistence.json").write_text(json.dumps(pers, indent=1))
    dm = demo(w, u, v, w.times)
    (out / "summary.json").write_text(json.dumps({
        "stamp": STAMP, "point_lonlat": [lon0, lat0], "wsw_definition": f"16-sector bin centred on {wsw_from} deg (configs/belauri_forward.json)",
        "headline": head, "demo_plumbing_check": dm,
        "provenance": {"file": NC.name, "sha256": sha, "download_date": meta["download_date"], "doi": meta["doi"]}}, indent=1))
    fig = plt.figure(figsize=(11, 4.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.15, 0.9, len(BIN_LABELS)))
    r = pd.DataFrame(rose)
    for k, name in enumerate(MONTHS.values()):
        ax = fig.add_subplot(1, 3, k + 1, projection="polar")
        ax.set_theta_zero_location("N"); ax.set_theta_direction(-1)
        bottom = np.zeros(16)
        rr = r[r.month == name]
        for j, l in enumerate(BIN_LABELS):
            h = 100.0 * rr[f"n_{l}"].values / rr.n.sum()
            ax.bar(np.radians(rr.centre_deg), h, width=np.radians(20), bottom=bottom, color=colors[j], label=l + " m/s")
            bottom += h
        ax.set_title(f"{name} 2023: wind FROM (% of hours)", fontsize=9)
    fig.axes[0].legend(loc="lower left", bbox_to_anchor=(-0.2, -0.3), fontsize=7, ncol=2)
    ax = fig.add_subplot(1, 3, 3)
    dd = pd.DataFrame(diur)
    for name, s in dd.groupby("month"):
        ax.plot(s.hour_utc, s.mean_speed, label=name)
    ax.set(xlabel="hour UTC (Nepal = UTC+5:45)", ylabel="mean 10 m speed (m/s)"); ax.legend()
    fig.suptitle(f"Belauri centroid, {STAMP}", fontsize=9)
    fig.tight_layout()
    fig.savefig(out / "era5_summary.png", dpi=150)
    print(json.dumps({"headline": head, "persistence": pers, "demo": dm}, indent=1))


if __name__ == "__main__":
    main()
