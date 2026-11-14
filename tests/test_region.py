"""E17 regional forcing (src/pravah/data/region.py): small synthetic fixtures, no network, no real data."""
import numpy as np
import pytest
import xarray as xr
from PIL import Image, TiffImagePlugin

from pravah import benchmarks as B
from pravah.data import region as R
from pravah.experiment_config import ExperimentConfig

BOX = R.Box(28.79, 80.31, 100_000.0)


def _era5(tmp_path, name, times, u_fn, v_fn, tname="valid_time"):
    lat = np.arange(29.5, 27.99, -.25)       # descending, as ERA5 ships it
    lon = np.arange(79.5, 81.26, .25)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    t = np.asarray(times, "datetime64[ns]")
    u = np.stack([u_fn(LA, LO, k) for k in range(len(t))])
    v = np.stack([v_fn(LA, LO, k) for k in range(len(t))])
    ds = xr.Dataset({"u10": ((tname, "latitude", "longitude"), u), "v10": ((tname, "latitude", "longitude"), v)},
                    coords={tname: t, "latitude": lat, "longitude": lon})
    p = tmp_path / name
    ds.to_netcdf(p, engine="h5netcdf")
    return p


def test_projection_roundtrip_and_box_extent():
    la, lo = BOX.latlon(np.array([0., 1e5]), np.array([0., 1e5]))
    assert np.allclose(la, [28.340, 29.240], atol=2e-3) and np.allclose(lo, [79.797, 80.823], atol=2e-3)
    x, y = BOX.xy(la, lo)
    assert np.allclose(x, [0, 1e5]) and np.allclose(y, [0, 1e5])


def test_read_sorts_lat_and_time_and_rejects_duplicates(tmp_path):
    t = R.hourly_times("2021-11-01", 1)
    a = _era5(tmp_path, "b.nc", t[12:], lambda la, lo, k: la * 0 + k, lambda la, lo, k: la)
    b = _era5(tmp_path, "a.nc", t[:12], lambda la, lo, k: la * 0 + k, lambda la, lo, k: la, tname="time")
    e = R.read_era5([a, b])
    assert (np.diff(e["lat"]) > 0).all() and (np.diff(e["time"]).astype(int) == 3600).all()
    assert e["v10"][0, :, 0] == pytest.approx(e["lat"])           # field flipped together with latitude
    with pytest.raises(ValueError, match="duplicated"):
        R.read_era5([a, a])


def test_bilinear_is_exact_for_linear_fields_at_misaligned_centres(tmp_path):
    # u linear in lon, v linear in lat: bilinear interpolation must reproduce them exactly at the (off-grid) cell centres
    t = R.hourly_times("2021-11-01", 1)[:3]
    p = _era5(tmp_path, "x.nc", t, lambda la, lo, k: 2 * (lo - 80) + k, lambda la, lo, k: -3 * (la - 28.8))
    e = R.read_era5([p])
    w, ug, vg = R.model_winds(e, BOX, (8, 8))
    la, lo = BOX.cell_centres((8, 8))
    assert np.allclose(ug[1], 2 * (lo - 80) + 1) and np.allclose(vg[0], -3 * (la - 28.8))
    assert np.allclose(w[:, 0], ug.mean((1, 2))) and w.shape == (3, 2)


def test_missing_times_invalid_values_and_box_outside_grid(tmp_path):
    t = R.hourly_times("2021-11-01", 1)[:4]
    e = R.read_era5([_era5(tmp_path, "x.nc", t, lambda la, lo, k: la * 0 + 1, lambda la, lo, k: la * 0)])
    with pytest.raises(ValueError, match="missing"):
        R.model_winds(e, BOX, (4, 4), times=R.hourly_times("2021-11-02", 1)[:2])
    bad = dict(e, u10=e["u10"].copy())
    bad["u10"][2, 0, 0] = np.nan
    with pytest.raises(ValueError, match="invalid"):
        R.model_winds(bad, BOX, (4, 4))
    bad["u10"][2, 0, 0] = 99.0
    assert not R.quality(bad["u10"], bad["v10"]).all()
    with pytest.raises(ValueError, match="inside"):
        R.model_winds(e, R.Box(28.79, 80.31, 400_000.0), (4, 4))


def test_time_zone_is_display_only():
    t = np.datetime64("2021-11-01T00:00", "s")
    assert R.to_npt(t) == np.datetime64("2021-11-01T05:45", "m")
    assert len(R.hourly_times("2021-11-01", 2)) == 48


def test_wind_rose_uses_from_direction():
    # ux > 0 blows toward the east, i.e. FROM the west (270 deg, sector 12 of 16); uy < 0 from the north (sector 0)
    r = R.wind_rose(np.array([[3., 0.], [0., -1.5]]))
    assert r[12].sum() == .5 and r[0].sum() == .5 and r.sum() == pytest.approx(1)
    assert r[12, 3] == .5 and r[0, 1] == .5     # speed bins [0,1) [1,2) [2,3) [3,5) [5,inf)


def _tif(tmp_path, a, x0, y0, s):
    ifd = TiffImagePlugin.ImageFileDirectory_v2()
    ifd[33550] = (s, s, 0.0)
    ifd[33922] = (0.0, 0.0, 0.0, x0, y0, 0.0)
    ifd[42113] = "-99999"
    p = tmp_path / "pop.tif"
    Image.fromarray(a.astype(np.float32), "F").save(p, tiffinfo=ifd)
    return p


def test_worldpop_aggregation_nodata_and_masks(tmp_path):
    s = 0.05
    a = np.ones((20, 24), np.float32)
    a[:, :6] = -99999                            # western strip outside the country raster
    a[2, 20] = 100.0
    lat, lon = BOX.latlon(np.array([0, 1e5]), np.array([0, 1e5]))
    p = _tif(tmp_path, a, lon[0] - .1, lat[1] + .05, s)
    c, la, lo = R.read_worldpop(p)
    assert np.isnan(c[:, :6]).all() and np.diff(la).max() < 0 and np.allclose(np.diff(lo), s)
    pop, vf = R.aggregate_counts(c, la, lo, BOX, (4, 4))
    inside = (lambda x, y: (x >= 0) & (x < 1e5) & (y >= 0) & (y < 1e5))(*BOX.xy(*np.meshgrid(la, lo, indexing="ij")))
    assert pop.sum() == pytest.approx(np.nansum(np.where(inside, c, 0)))   # counts conserved inside the box
    assert vf[0].max() < .5 and vf[-1].min() == 1                         # west column mostly NoData
    assert (R.candidate_mask(vf, [(0, 0), (3, 3)]) == [False, True]).all()
    m = R.source_mask(pop)
    assert m[pop.argmax() // 4, pop.argmax() % 4] and not m[0].any()


def test_masked_source_draw_respects_mask_and_default_unchanged():
    cfg = ExperimentConfig.default("T")
    allowed = np.zeros(cfg.sources.lattice, bool)
    allowed[6:, :] = True
    cells, a = B.draw_sources(cfg, 0, 2, allowed=allowed)
    assert all(c[0] >= 6 for c in cells)
    c0, a0 = B.draw_sources(cfg, 0, 2)
    c1, a1 = B.draw_sources(cfg, 0, 2, allowed=np.ones(cfg.sources.lattice, bool))
    assert c0 == c1 and np.array_equal(a0, a1)
