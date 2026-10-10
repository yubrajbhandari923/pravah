import json

import matplotlib
import numpy as np
import pandas as pd
import pytest
import xarray as xr

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pravah import io, plotting  # noqa: E402
from pravah.config import Domain, GaussianSource  # noqa: E402
from pravah.observations import (ObservationSpec, SensorLayout, candidate_hull, interp_weights, observe,  # noqa: E402
                                 sample)

DOM = Domain(lx=1000.0, ly=800.0, nx=10, ny=8)
TIMES = np.arange(0.0, 601.0, 100.0)


def make_field(fn):
    x = (np.arange(DOM.nx) + 0.5) * DOM.dx
    y = (np.arange(DOM.ny) + 0.5) * DOM.dy
    T, Y, X = np.meshgrid(TIMES, y, x, indexing="ij")
    return xr.DataArray(fn(X, Y, T), dims=("time", "y", "x"), coords={"time": TIMES, "y": y, "x": x})


def affine(x, y, t):
    return 3.0 + 0.01 * x - 0.02 * y + 0.005 * t


def rand_layout(n, seed=0):
    rng = np.random.default_rng(seed)
    (x0, x1), (y0, y1) = candidate_hull(DOM)
    return SensorLayout(tuple(f"s{i}" for i in range(n)), tuple(rng.uniform(x0, x1, n)), tuple(rng.uniform(y0, y1, n)))


def test_affine_exact_at_arbitrary_points():
    lay = rand_layout(20)
    ts = [0.0, 33.3, 250.0, 599.0, 600.0]
    got = sample(make_field(affine), lay, ts)
    exp = affine(np.array(lay.x)[None], np.array(lay.y)[None], np.array(ts)[:, None])
    assert got.dims == ("time", "sensor")
    np.testing.assert_allclose(got.values, exp, rtol=1e-12)
    assert list(got["sensor"].values) == list(lay.ids)
    np.testing.assert_allclose(got["x"].values, lay.x)


def test_constant_exact_and_weights_sum_to_one():
    lay = rand_layout(10, 1)
    np.testing.assert_allclose(sample(make_field(lambda x, y, t: 7.0 + 0 * x), lay, [10.0, 400.0]).values, 7.0)
    x = (np.arange(DOM.nx) + 0.5) * DOM.dx
    y = (np.arange(DOM.ny) + 0.5) * DOM.dy
    idx, w = interp_weights(x, y, lay)
    assert idx.shape == w.shape == (10, 4)
    np.testing.assert_allclose(w.sum(axis=1), 1.0)


def test_cell_centre_returns_cell_value():
    f = make_field(lambda x, y, t: np.sin(x / 100) * np.cos(y / 70) + t)
    lay = SensorLayout(("a", "b"), (DOM.dx * 2.5, DOM.dx / 2), (DOM.dy * 3.5, DOM.ly - DOM.dy / 2))
    got = sample(f, lay, [100.0, 300.0])
    np.testing.assert_allclose(got.values[:, 0], f.isel(time=[1, 3], x=2, y=3).values, atol=1e-12)
    np.testing.assert_allclose(got.values[:, 1], f.isel(time=[1, 3], x=0, y=DOM.ny - 1).values, atol=1e-12)


@pytest.mark.parametrize("x,y,t", [(10.0, 400.0, 100.0), (990.0, 400.0, 100.0), (500.0, 10.0, 100.0),
                                  (500.0, 790.0, 100.0), (500.0, 400.0, -1.0), (500.0, 400.0, 601.0)])
def test_rejects_out_of_range(x, y, t):
    with pytest.raises(ValueError, match="outside"):
        sample(make_field(affine), SensorLayout(("a",), (x,), (y,)), [t])


def test_rejects_non_monotone_times_and_duplicate_ids():
    with pytest.raises(ValueError, match="increasing"):
        sample(make_field(affine), rand_layout(2), [200.0, 100.0])
    with pytest.raises(ValueError, match="unique"):
        SensorLayout(("a", "a"), (1.0, 2.0), (1.0, 2.0))
    with pytest.raises(ValueError, match="increasing"):
        ObservationSpec((0.0, 0.0), 1.0, 0)
    with pytest.raises(ValueError):
        SensorLayout(("a",), (float("nan"),), (1.0,))
    with pytest.raises(ValueError):
        SensorLayout(("a",), (1.0, 2.0), (1.0,))


def test_negative_noise_rejected():
    with pytest.raises(ValueError, match="noise_std"):
        ObservationSpec((0.0,), -0.1, 0)
    with pytest.raises(ValueError, match="noise_std"):
        ObservationSpec((0.0,), (1.0, -1.0), 0)


def test_layout_subset_select():
    lay = rand_layout(5)
    assert lay.subset(["s3", "s1"]).ids == ("s3", "s1")
    assert lay.select([0, 4]).x == (lay.x[0], lay.x[4])
    with pytest.raises(ValueError):
        lay.subset(["nope"])


def test_candidate_hull():
    assert candidate_hull(DOM) == ((50.0, 950.0), (50.0, 750.0))


def test_noise_reproducible_and_statistics():
    f = make_field(lambda x, y, t: 0.0 * x)
    lay = rand_layout(50)
    times = tuple(np.arange(0.0, 601.0, 10.0)[:61])
    a = observe(f, lay, ObservationSpec(times, 2.0, 1))
    b = observe(f, lay, ObservationSpec(times, 2.0, 1))
    c = observe(f, lay, ObservationSpec(times, 2.0, 2))
    np.testing.assert_array_equal(a.values.values, b.values.values)
    assert not np.allclose(a.values.values, c.values.values)
    assert a.values.shape == (61, 50)
    assert abs(a.values.values.std() - 2.0) < 0.1  # n = 3050
    assert abs(a.values.values.mean()) < 0.2
    assert (a.values.values < 0).any()  # negatives retained
    assert a.layout is lay and a.noise_std.shape == (50,)


def test_per_sensor_noise_and_zero_noise():
    f = make_field(affine)
    lay = rand_layout(3)
    spec = ObservationSpec((0.0, 100.0), (0.0, 1.0, 5.0), 3)
    o = observe(f, lay, spec)
    np.testing.assert_array_equal(o.noise_std, [0.0, 1.0, 5.0])
    np.testing.assert_allclose(o.values.values[:, 0], sample(f, lay, [0.0, 100.0]).values[:, 0])
    with pytest.raises(ValueError):
        observe(f, lay, ObservationSpec((0.0,), (1.0, 2.0), 0))


def test_run_dir(tmp_path):
    d = tmp_path / "run"
    io.make_run_dir(d)
    (d / "keep.txt").write_text("x")
    with pytest.raises(FileExistsError):
        io.make_run_dir(d)
    io.make_run_dir(d, overwrite=True)
    assert (d / "keep.txt").read_text() == "x"
    io.make_run_dir(tmp_path / "empty")
    io.make_run_dir(tmp_path / "empty")  # empty existing dir is fine


def test_write_json_nan_to_null(tmp_path):
    p = tmp_path / "a.json"
    io.write_json(p, {"a": float("nan"), "b": [np.inf, np.float64(1.5), np.int64(2)], "c": np.array([1.0, np.nan])})
    assert json.loads(p.read_text()) == {"a": None, "b": [None, 1.5, 2], "c": [1.0, None]}


def test_provenance_keys():
    pv = io.provenance({"a": 1}, {"design": 3}, {"note": "x"})
    for k in ("config", "seeds", "versions", "python", "platform", "git_revision", "git_dirty", "timestamp_utc",
              "extra"):
        assert k in pv
    assert {"numpy", "scipy", "xarray", "fipy", "pravah"} <= set(pv["versions"])
    json.dumps(pv, allow_nan=False)


def test_save_field_and_csv(tmp_path):
    f = make_field(affine)
    io.save_field(tmp_path / "f.nc", f.rename("c"))
    assert float(xr.open_dataset(tmp_path / "f.nc")["c"].sum()) == pytest.approx(float(f.sum()))
    io.write_csv(tmp_path / "t.csv", pd.DataFrame({"a": [1, 2]}))
    assert (tmp_path / "t.csv").read_text().startswith("a")


def test_plotting_smoke():
    f = make_field(lambda x, y, t: np.exp(-((x - 500) ** 2 + (y - 400) ** 2) / 1e5) * (1 + t / 600))
    lay = rand_layout(3)
    srcs = [GaussianSource("S1", 300.0, 300.0, 50.0)]
    obs = observe(f, lay, ObservationSpec(tuple(TIMES), 0.05, 0))
    nt = len(TIMES)
    budget = xr.Dataset({k: ("time", np.random.default_rng(0).random(nt)) for k in
                         ("emission", "removal", "inflow", "outflow", "mass_change", "residual")},
                        coords={"time": TIMES})
    df = pd.DataFrame({"method": ["a"] * 4 + ["b"] * 4, "n_sensors": [1, 1, 2, 2] * 2,
                       "error": np.random.default_rng(1).random(8)})
    figs = [plotting.plot_sources(srcs, DOM, lay), plotting.plot_field_snapshots(f, layout=lay),
            plotting.plot_traces(obs.values, sample(f, lay, TIMES)), plotting.plot_mass_budget(budget),
            plotting.plot_layout(rand_layout(20, 5), lay, srcs, DOM), plotting.plot_compare_fields(f, 0.9 * f),
            plotting.plot_error_vs_budget(df)]
    anim = plotting.animate_field(f)
    assert anim._func is not None  # noqa: SLF001
    anim._func(1)  # noqa: SLF001
    for fig in figs:
        assert fig.axes
    plt.close("all")
