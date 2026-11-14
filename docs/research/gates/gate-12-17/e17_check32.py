"""Gate-12-17, E17 (local, read-only raw ERA5 and the 32^2 pilot field cache):
(1) independent box-mean wind: own NetCDF read (raw h5netcdf via xarray, no sortby), own bilinear weights on the
    equirectangular cell centres, compared with region.model_winds; sign convention via the July FROM-direction;
(2) the 'point' spec: direction/speed error vs the box mean on the coolA sweep test hours;
(3) Courant numbers of the 64^2 sweep (dt 240 s, dx 1562.5 m) for the box mean and the point wind;
(4) 32^2 pilot transport floors: forecast RMSE with the TRUE strengths under each model wind (oracle/point/dir30), vs the
    committed pilot rows (map_grow, k = 8) and persistence.
Usage: PRAVAH_E17_RAW=<main>/.cache/e17/raw uv run python e17_check32.py PILOT_OUT
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

root = Path(__file__).resolve().parents[4]
sys.path[:0] = [str(root / "scripts"), str(root / "src")]
import e17_regional as E  # noqa: E402
from pravah.data import region as R  # noqa: E402


def main():
    pilot = Path(sys.argv[1])
    files = sorted(E.RAW.glob("era5_sl_belauri_*.nc"))
    u, v, t = [], [], []
    for f in files:
        ds = xr.open_dataset(f, engine="h5netcdf")
        ds = ds.squeeze(drop=True)
        lat, lon = ds.latitude.values, ds.longitude.values
        u.append(ds.u10.transpose(..., "latitude", "longitude").values)
        v.append(ds.v10.transpose(..., "latitude", "longitude").values)
        t.append(ds["valid_time" if "valid_time" in ds else "time"].values)
    u, v, t = np.concatenate(u), np.concatenate(v), np.concatenate(t).astype("datetime64[s]")
    print("raw latitude order:", "descending" if lat[0] > lat[-1] else "ascending", "; lon", lon.min(), "-", lon.max())
    # own bilinear weights at the 64^2 cell centres (works for either latitude order)
    nx = 64
    L, R0 = 1e5, 6371008.8
    x = (np.arange(nx) + .5) * L / nx - L / 2
    la = 28.79 + np.degrees(x / R0)
    lo = 80.31 + np.degrees(x / (R0 * np.cos(np.radians(28.79))))
    LA, LO = np.meshgrid(la, lo, indexing="ij")  # [y-index? no: rows = lat] -> we only need the box mean

    def weights(grid, pts):
        g = grid if grid[0] < grid[-1] else grid[::-1]
        i = np.clip(np.searchsorted(g, pts) - 1, 0, len(g) - 2)
        f = (pts - g[i]) / (g[i + 1] - g[i])
        if grid[0] > grid[-1]:
            i = len(g) - 2 - i
            return i + 1, i, f  # (lower index in raw order, upper index, fraction toward upper)
        return i, i + 1, f

    i0, i1, fy = weights(lat, LA.ravel())
    j0, j1, fx = weights(lon, LO.ravel())

    def interp(a):
        return ((1 - fy) * (1 - fx) * a[:, i0, j0] + (1 - fy) * fx * a[:, i0, j1] + fy * (1 - fx) * a[:, i1, j0]
                + fy * fx * a[:, i1, j1]).mean(1)

    mine = np.stack([interp(u), interp(v)], 1)
    e = R.read_era5(files)
    ref = R.model_winds(e, R.BELAURI, (64, 64))[0]
    order = np.argsort(t)
    print("(1) box mean, own vs region.model_winds: max |diff| %.2e m/s over %d hours; times identical %s" % (
        np.abs(mine[order] - ref).max(), len(ref), bool(np.array_equal(t[order], e["time"]))))
    jul = (e["time"] >= np.datetime64("2021-07-01")) & (e["time"] < np.datetime64("2021-08-01"))
    th = np.degrees(np.arctan2(-ref[:, 0], -ref[:, 1])) % 360
    print("    July: share of hours with FROM direction in 67-112 deg (easterly): %.2f; mean u10 %.2f m/s (negative = "
          "westward flow = easterly wind)" % (((th[jul] >= 67) & (th[jul] <= 112)).mean(), ref[jul, 0].mean()))
    hrs = (e["time"].astype("datetime64[h]").astype(int) % 24)
    cool = ~jul
    for h0, lab in ((9, "09-12 UTC = 14:45-17:45 NPT"), (18, "18-21 UTC = 23:45-02:45 NPT")):
        m = cool & (hrs >= h0) & (hrs < h0 + 3)
        print(f"    cool season {lab}: vector-mean FROM direction %.0f deg" % (
            np.degrees(np.arctan2(-ref[m, 0].mean(), -ref[m, 1].mean())) % 360))
    # (2) point spec on the coolA sweep test hours
    tt = E.case_times("sweep", "coolA")
    cfg = E.make_cfg("sweep", "coolA")
    te = slice((cfg.split.validation[1]) * 24, None)
    wb, wp = E.case_winds("sweep", "coolA"), E.case_winds("sweep", "coolA", E.SPECS[1])
    s0, s1 = np.hypot(*wb[te].T), np.hypot(*wp[te].T)
    d = np.abs((np.degrees(np.arctan2(-wp[te, 0], -wp[te, 1])) - np.degrees(np.arctan2(-wb[te, 0], -wb[te, 1])) + 180) % 360 - 180)
    print("(2) point vs box mean, coolA test hours: mean |dir diff| %.1f deg (median %.1f, hours with box speed < 1 m/s: "
          "%.0f %%, their mean |dir diff| %.1f deg); mean speed ratio %.2f, ratio of mean speeds %.2f; vector RMS diff %.2f m/s "
          "vs mean box speed %.2f" % (d.mean(), np.median(d), 100 * (s0 < 1).mean(), d[s0 < 1].mean(), (s1 / s0).mean(),
                                       s1.mean() / s0.mean(), np.sqrt(((wp[te] - wb[te]) ** 2).sum(1).mean()), s0.mean()))
    # (3) Courant
    dx = 1e5 / 64
    for nm, w in (("box mean", wb), ("point", wp)):
        print("(3) %s: max per-axis Courant at dt 240 s = %.2f (VanLeer limit 1; the solver raises above it)" % (
            nm, np.abs(w).max() * 240 / dx))
    # (4) pilot floors
    rows = pd.read_csv(root / "reports/quarto/tables/e17-local-rows.csv")
    out = []
    for K in E.SCALES["pilot"]["Ks"]:
        for seed in E.SCALES["pilot"]["seeds"]:
            job = ("pilot", "coolA", K, seed)
            ft = E.field_tasks(*job)
            cfg, drv = E.drivers(*job)
            sc = E.B.generate_scenario(cfg, seed, K, truth_field=np.load(pilot / "fields" / f"{ft[0][0]}.npy"), drivers=drv)
            issue, a_ref, cbg = E.X.issue_times(sc), cfg.sources.reference_rate_ug_s, cfg.sources.background_ug_m3
            ut = np.tile(sc.truth.strengths_ug_s / a_ref, (len(issue), 1))
            w = np.ones(cfg.model.grid) / np.prod(cfg.model.grid)
            for sp in E.specs_for("coolA"):
                F = np.stack([np.load(pilot / "fields" / f"{n}.npy") for n, tk in ft[1:] if tk[6] == sp.name])
                fl = E.X.score(ut, F, sc.truth.field, issue, sc.truth.strengths_ug_s, a_ref, cbg, w)["forecast_rmse"]
                q = rows[(rows.case == "coolA") & (rows.K == K) & (rows.seed == seed) & (rows.spec == sp.name) & (rows.k == 8)
                         & (rows.estimator == "map_grow")]
                pers = rows[(rows.case == "coolA") & (rows.K == K) & (rows.seed == seed) & (rows.strategy == "persistence")].forecast_rmse.iloc[0]
                out.append(dict(K=K, seed=seed, spec=sp.name, true_strength_rmse=fl, persistence=pers,
                                map_grow_dopt=q[q.strategy == "dopt"].forecast_rmse.mean(),
                                map_grow_random=q[q.strategy == "random"].forecast_rmse.mean()))
    o = pd.DataFrame(out)
    o["floor_share_dopt"] = o.true_strength_rmse / o.map_grow_dopt
    print("(4) 32^2 pilot coolA, k = 8: forecast RMSE with TRUE strengths under each model wind vs estimated (map_grow)")
    print(o.groupby("spec")[["true_strength_rmse", "map_grow_dopt", "map_grow_random", "persistence", "floor_share_dopt"]]
          .median().round(4).to_string())


if __name__ == "__main__":
    main()
