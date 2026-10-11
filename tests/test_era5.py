"""ERA5 ingestion: synthetic NetCDF (no dependence on the real file), convention check, one skip-if-missing real-file test."""
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from pravah import era5
from pravah import forward as F
from pravah.config import Boundary, Domain, GaussianSource, Scenario, Times, Transport

REAL = Path(__file__).resolve().parents[1] / "data/era5/belauri_era5_10m_wind.nc"


def _write(path, lat=(29.0, 28.75, 28.5), units="m s**-1", bad=False):
    t = np.datetime64("2023-01-01T00") + np.arange(4) * np.timedelta64(1, "h")
    lon = np.array([80.0, 80.25, 80.5])
    u = np.broadcast_to(1.0 + 2.0 * (lon - 80.0), (4, len(lat), 3)).astype("float32").copy()  # u = 1 + 2 dlon
    v = np.broadcast_to(np.array(lat)[:, None] - 28.5, (4, len(lat), 3)).astype("float32").copy()  # v = lat - 28.5
    if bad:
        u[1, 0, 0] = np.nan
    a = {"units": units}
    xr.Dataset({"u10": (("valid_time", "latitude", "longitude"), u, a), "v10": (("valid_time", "latitude", "longitude"), v, a)},
               coords={"valid_time": t, "latitude": list(lat), "longitude": lon}).to_netcdf(path)
    return str(path)


def test_load_sample_box(tmp_path):
    w = era5.load_era5_wind(_write(tmp_path / "a.nc"))
    assert w.u.shape == (4, 3, 3) and np.all(np.diff(w.lat) > 0)  # latitude flipped to ascending
    u, v = era5.sample_at(w, 80.1, 28.6)  # exact for the linear fields
    assert np.allclose(u, 1.2) and np.allclose(v, 0.1)
    bu, bv = era5.box_mean(w, 80.0, 28.5, 80.5, 29.0)
    assert np.allclose(bu, 1.5) and np.allclose(bv, 0.25)
    with pytest.raises(ValueError):
        era5.sample_at(w, 79.0, 28.6)


def test_validation(tmp_path):
    with pytest.raises(ValueError, match="non-finite"):
        era5.load_era5_wind(_write(tmp_path / "b.nc", bad=True))
    with pytest.raises(ValueError, match="units"):
        era5.load_era5_wind(_write(tmp_path / "c.nc", units="knots"))


def test_from_deg():
    assert np.allclose(era5.from_deg([1.0, -1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]), [270.0, 90.0, 180.0, 0.0])


def test_eastward_wind_carries_plume_east():
    d = Domain(20_000.0, 20_000.0, 20, 20)
    segs = era5.to_segments(d, np.array([0.0, 3600.0]), np.array([2.0, 2.0]), np.array([0.0, 0.0]))
    assert len(segs) == 2 and segs[0].u == pytest.approx(2.0) and segs[0].v == pytest.approx(0.0, abs=1e-9)
    scn = Scenario(domain=d, sources=(GaussianSource("s", 5000.0, 10000.0, 1000.0),), transport=Transport(segs, 100.0, 0.0, 500.0),
                   boundary=Boundary("open"), times=Times(3600.0, 100.0, 900.0))
    c = F.simulate(scn, [1e6]).concentration.values[-1]
    xc = (c.sum(axis=0) * (np.arange(20) + 0.5) * d.dx).sum() / c.sum()
    yc = (c.sum(axis=1) * (np.arange(20) + 0.5) * d.dy).sum() / c.sum()
    assert xc > 5000.0 + 3000.0 and abs(yc - 10000.0) < 100.0


@pytest.mark.skipif(not REAL.exists(), reason="real ERA5 file not present")
def test_real_file():
    w = era5.load_era5_wind(str(REAL))
    assert w.u.shape == (1488, 9, 9) and w.v.shape == (1488, 9, 9)
    assert str(w.times[0])[:10] == "2023-01-01" and str(w.times[-1])[:13] == "2023-07-31T23"
    assert np.abs(w.u).max() < 40 and np.abs(w.v).max() < 40
    u, v = era5.sample_at(w, 80.354, 28.700)
    assert np.isfinite(u).all() and u.min() >= w.u.min() and u.max() <= w.u.max()
