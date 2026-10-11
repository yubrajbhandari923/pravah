import numpy as np
import pytest

from pravah import population as P
from pravah.geo.regionio import GridSpec

G = GridSpec(500000.0, 3000000.0, 4, 4, 1000.0, 1000.0, 32645)


def test_density_roundtrip():
    c = np.arange(16.0).reshape(4, 4)
    d = P.counts_to_density(c, G)
    assert d[1, 1] == c[1, 1] / 1.0  # 1 km^2 cells
    g2 = GridSpec(0, 0, 4, 4, 500.0, 500.0, 1)
    assert P.counts_to_density(np.ones((4, 4)), g2)[0, 0] == 4.0
    assert np.allclose(P.density_to_counts(P.counts_to_density(c, g2), g2), c)


def test_regrid_coarser_finer_exact():
    c = np.arange(16.0).reshape(4, 4)
    coarse = GridSpec(G.x0, G.y0, 2, 2, 2000.0, 2000.0, G.epsg)
    out, r = P.regrid_counts(c, G, coarse)
    assert out[0, 0] == c[:2, :2].sum() and r["total_out"] == r["total_in"] == 120 and r["lost_outside"] == 0
    fine = GridSpec(G.x0, G.y0, 8, 8, 500.0, 500.0, G.epsg)
    out, r = P.regrid_counts(c, G, fine)
    assert out.sum() == 120 and out[0, 0] == c[0, 0] / 4 and r["lost_outside"] == 0


def test_regrid_shifted_reports_loss():
    c = np.ones((4, 4)) * 10
    sh = GridSpec(G.x0 + 500.0, G.y0 + 500.0, 4, 4, 1000.0, 1000.0, G.epsg)
    out, r = P.regrid_counts(c, G, sh)
    # dst covers 3.5x3.5 km of the 4x4 km source -> 12.25/16 of the people
    assert r["total_out"] == pytest.approx(160 * 12.25 / 16) and r["lost_outside"] == pytest.approx(160 - 122.5)
    assert out[0, 0] == pytest.approx(10.0)


def test_regrid_nan_excluded_and_reported():
    c = np.ones((4, 4))
    c[0, 0] = np.nan
    out, r = P.regrid_counts(c, G, G)
    assert r["nodata_cells"] == 1 and r["total_in"] == 15 and out[0, 0] == 0 and r["lost_outside"] == 0
    with pytest.raises(ValueError):
        P.regrid_counts(c, G, GridSpec(0, 0, 4, 4, 1000.0, 1000.0, 1))


@pytest.mark.parametrize("name", list(P.FIXTURES))
def test_fixtures(name):
    f = P.FIXTURES[name]
    g = GridSpec(0, 0, 20, 15, 500.0, 500.0, 32645)
    a = f(g, 12345, 3)
    assert a.shape == g.shape and a.sum() == 12345 and a.min() >= 0 and np.all(a == np.round(a))
    assert np.array_equal(a, f(g, 12345, 3))
    assert np.array_equal(f(g, 7, 1).sum(), 7)


def test_fixture_shapes():
    g = GridSpec(0, 0, 20, 20, 500.0, 500.0, 1)
    assert P.uniform(g, 400).std() == 0
    assert P.one_town(g, 10000).argmax() in (np.ravel_multi_index((j, i), g.shape) for j in range(8, 12) for i in range(8, 12))
    assert not np.array_equal(P.one_town(g, 10000, 0), P.one_town(g, 10000, 1))


def test_rmse_hand_computed():
    e = np.arange(16.0).reshape(4, 4)
    n = np.zeros((4, 4))
    n[2, 3] = 50
    assert P.pop_weighted_rmse(e, n) == pytest.approx(e[2, 3])
    assert P.pop_weighted_rmse(e, np.full((4, 4), 7.0)) == pytest.approx(P.area_weighted_rmse(e, 1e6))
    assert P.area_weighted_rmse(e, 1e6) == pytest.approx(np.sqrt((e**2).mean()))
    sw = np.zeros((4, 4))
    sw[0, :] = 1
    assert P.area_weighted_rmse(e, 1.0, sw) == pytest.approx(np.sqrt((e[0] ** 2).mean()))
    # time: e(t) = t * ones, t in [0,2] -> trapezoid of t^2 over [0,1,2] = 0.5*1+... = (0+1)/2+(1+4)/2 = 3 ; /2
    et = np.stack([np.full((4, 4), t) for t in (0.0, 1.0, 2.0)])
    assert P.pop_weighted_rmse(et, np.ones((4, 4)), times=np.array([0.0, 1.0, 2.0])) == pytest.approx(np.sqrt(1.5))
    with pytest.raises(ValueError):
        P.pop_weighted_rmse(et, np.ones((4, 4)))


def test_nan_counts_and_err():
    n = np.ones((4, 4))
    n[0, 0] = np.nan
    e = np.ones((4, 4))
    e[0, 0] = np.nan
    assert P.pop_weighted_rmse(e, n) == 1.0
    e[1, 1] = np.nan
    with pytest.raises(ValueError):
        P.pop_weighted_rmse(e, n)


def test_mixed_weight():
    n = np.arange(1.0, 17.0).reshape(4, 4)
    a = np.full((4, 4), 2.0)
    assert np.allclose(P.mixed_weight(n, a, 0), 1 / 16) and np.allclose(P.mixed_weight(n, 2.0, 1), n / n.sum())
    w = P.mixed_weight(n, a, 0.3)
    assert w.sum() == pytest.approx(1.0)
    with pytest.raises(ValueError):
        P.mixed_weight(n, a, 1.5)


def test_mixed_weight_with_field_mse():
    xr = pytest.importorskip("xarray")
    from pravah.metrics import field_mse
    t, y, x = np.array([0.0, 1.0]), np.arange(4.0), np.arange(4.0)
    mk = lambda v: xr.DataArray(np.full((2, 4, 4), v), dims=("time", "y", "x"), coords={"time": t, "y": y, "x": x})
    n = np.ones((4, 4))
    assert field_mse(mk(2.0), mk(0.0), P.mixed_weight(n, 1.0, 0.5)) == pytest.approx(4.0)


def test_dataset_record():
    kw = dict(source="s", url="u", version="v", license="l", attribution="a", original_resolution="100 m", crs="EPSG:4326",
              units="people/cell", sha256="0" * 64, resampling="none", limitations="x")
    r = P.dataset_record(**kw)
    assert r["units"] == "people/cell" and r["download_date"]
    with pytest.raises(ValueError):
        P.dataset_record(**{**kw, "units": "people"})


def test_lonlat_counts_and_geotiff(tmp_path):
    pytest.importorskip("pyproj")
    g = GridSpec(0, 0, 2, 2, 1, 1, 32645)
    from pyproj import Transformer
    x, y = Transformer.from_crs("EPSG:32645", "EPSG:4326", always_xy=True).transform(500000.0, 3000000.0)
    # 4 source cells of 0.1 deg are far bigger than dst: all mass lost outside, none invented
    out, r = P.counts_from_lonlat(np.ones((2, 2)), x, y, 0.1, 0.1, g)
    assert out.sum() == r["total_out"] and r["total_in"] == 4
    rio = pytest.importorskip("rasterio")
    from rasterio.transform import from_origin
    p = tmp_path / "a.tif"
    with rio.open(p, "w", driver="GTiff", height=2, width=2, count=1, dtype="float64", crs="EPSG:4326",
                  transform=from_origin(x, y, 0.1, 0.1), nodata=-1) as ds:
        ds.write(np.array([[1.0, -1.0], [1.0, 1.0]]), 1)
    out, r = P.counts_from_geotiff(str(p), g)
    assert r["nodata_cells"] == 1 and r["total_in"] == 3
