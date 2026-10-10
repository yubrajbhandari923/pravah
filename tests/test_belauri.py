"""Tests for the Belauri forward study (Stage E): real prepared geography, synthetic emissions and meteorology.

Skipped when data/belauri/manifest.json is missing.  Real FiPy solves on the 500 m grid cropped to a small buffer, short
duration (7200 s, dt 100 s); no solver mocks."""
from __future__ import annotations

import dataclasses
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pravah.attribution import contributions
from pravah.belauri import (Study, block_average, plotting_only_check, run_belauri_forward, select_points)
from pravah.cli import main
from pravah.config import (Boundary, Domain, GaussianSource, InflowBasis, Initial, Scenario, SolverOptions, Times, Transport,
                           WindSegment, load_config, to_dict)
from pravah.forward import FIPY, simulate
from pravah.inverse import ResponseCache
from pravah.geo.prepare import Region, load_region
from pravah.geo.scenarios import (LOCAL_ORIGIN_TOL, BelauriCase, belauri_scenario, build_wind, crop_offset_m, crop_region,
                                  eddy_wind, numerical_diffusion_estimate, study_weights_of, toward_vector, uniform_wind)
from pravah.sources import check_origin_tags, source_fields

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "data" / "belauri" / "manifest.json"
pytestmark = pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing (run geo prepare)")

AMP = {"road": 1.5e6, "builtup": 1.0e6, "external": 5.0e5}
ONSET = ((0.0, 0.0), (1800.0, 1.0))
BUF = 1000.0  # small crop buffer for the tests
T, DT, OUT = 7200.0, 100.0, 900.0
WSW = uniform_wind([(0.0, 247.5)], 1.5)


@pytest.fixture(scope="module")
def region() -> Region:
    return load_region(MANIFEST)


def case(region: Region, wind: Any = WSW, res: float = 500.0, buf: float | None = BUF, **kw: Any) -> BelauriCase:
    args: dict[str, Any] = dict(duration_s=T, dt_s=DT, output_interval_s=OUT, K=100.0, lam=1.0 / 21600.0, H=500.0,
                                amplitudes=AMP, external_profile=ONSET, crop_buffer_m=buf)
    args.update(kw)
    return belauri_scenario(region, res, wind, **args)


def test_tags_sum_to_total_and_local_plus_external_equals_combined(region: Region) -> None:
    c = case(region)
    resp = ResponseCache().get(c.scenario)
    z = np.zeros_like(resp.base)
    tags = contributions(resp, c.theta, {"initial": z, "background": z})
    total = sum((np.asarray(v.values) for v in tags.values()), np.zeros_like(resp.base))
    assert np.abs(total - np.asarray(resp.field(c.theta).values)).max() <= 1e-10
    local, ext = np.asarray(tags["local"].values), np.asarray(tags["external_sources"].values)
    direct = FIPY(c.scenario, c.theta).concentration.values
    assert np.abs(local + ext - direct).max() <= 1e-10
    assert local.max() > 1e-3 and ext.max() > 1e-3  # both tags are non-trivial (not a vacuous identity)
    # local-only and external-only runs are independent solves of the two sub-problems
    th_loc = np.where([lab["group"] == "local" for lab in c.labels], c.theta, 0.0)
    only_local = FIPY(c.scenario, th_loc).concentration.values
    assert np.abs(only_local - local).max() <= 1e-10


def test_study_mask_area_and_origin_tags(region: Region) -> None:
    area_km2 = float(region.manifest["study"]["area_km2"])
    for res in (500.0, 250.0):
        a = float(region.study_mask(res).weights.sum()) * res * res / 1e6
        assert a == pytest.approx(area_km2, rel=1e-9)
    c = case(region)
    assert float(study_weights_of(c).sum()) * 500.0**2 / 1e6 == pytest.approx(area_km2, rel=1e-9)
    check_origin_tags(c.scenario, tol=LOCAL_ORIGIN_TOL)  # passes at the documented tolerance
    with pytest.raises(ValueError, match="tagged local"):
        check_origin_tags(c.scenario, tol=0.01)  # the library default is too strict for polygon-clipped rasters
    assert {s.name: s.group for s in c.scenario.sources} == {"road": "local", "builtup": "local", "external_patch": "external"}
    assert LOCAL_ORIGIN_TOL < 0.1


@pytest.mark.parametrize("res", [500.0, 250.0])
def test_raster_source_normalisation_on_both_resolutions(region: Region, res: float) -> None:
    c = belauri_scenario(region, res, WSW, groups=("road", "builtup", "facility", "external"), duration_s=T, dt_s=50.0,
                         output_interval_s=OUT, K=100.0, lam=1 / 21600, H=500.0,
                         amplitudes={**AMP, "facility": 1e5}, external_profile=ONSET, crop_buffer_m=BUF)
    f = source_fields(c.scenario)
    d = c.scenario.domain
    assert [float((fj * d.dx * d.dy).sum()) for fj in f] == pytest.approx([1.0] * 4, abs=1e-12)


def test_crop_region_keeps_absolute_positions(region: Region) -> None:
    c_std = belauri_scenario(region, 500.0, WSW, duration_s=T, dt_s=DT, output_interval_s=OUT, K=100.0, lam=1 / 21600, H=500.0,
                             amplitudes=AMP, external_profile=ONSET)
    c_small = case(region, buf=BUF)
    assert c_small.scenario.domain.nx < c_std.scenario.domain.nx and c_small.scenario.domain.ny < c_std.scenario.domain.ny
    off = crop_offset_m(region, c_small.region)
    assert off[0] % 500.0 == 0.0 and off[1] % 500.0 == 0.0 and off[0] > 0

    def centroid_utm(c: BelauriCase, name: str) -> tuple[float, float]:
        f = source_fields(c.scenario)[[s.name for s in c.scenario.sources].index(name)]
        d = c.scenario.domain
        x, y = (np.arange(d.nx) + 0.5) * d.dx, (np.arange(d.ny) + 0.5) * d.dy
        cx, cy = (f.sum(0) * x).sum() / f.sum(), (f.sum(1) * y).sum() / f.sum()
        return c.region.georef.x0 + float(cx), c.region.georef.y0 + float(cy)

    for name in ("road", "builtup"):
        assert centroid_utm(c_small, name) == pytest.approx(centroid_utm(c_std, name), abs=1e-6)
    gs, gc = c_std.scenario.sources[-1], c_small.scenario.sources[-1]
    assert isinstance(gs, GaussianSource) and isinstance(gc, GaussianSource)
    assert (gs.x + c_std.region.georef.x0, gs.y + c_std.region.georef.y0) == pytest.approx(
        (gc.x + c_small.region.georef.x0, gc.y + c_small.region.georef.y0), abs=1e-6)
    ws, wc = study_weights_of(c_std), study_weights_of(c_small)
    assert wc.sum() == pytest.approx(ws.sum(), rel=1e-12)
    dx = 500.0
    i0, j0 = int(off[0] / dx), int(off[1] / dx)
    assert np.array_equal(ws[j0:j0 + wc.shape[0], i0:i0 + wc.shape[1]], wc)
    # a crop that would cut the external patch is refused
    with pytest.raises(ValueError, match="external patch"):
        crop_region(region, 0.0)
    # both levels cropped consistently
    assert c_small.region.level(250.0).grid.nx == 2 * c_small.region.level(500.0).grid.nx


def measured_keff_excess(nx: int, dt: float, T: float = 3600.0) -> float:
    """K_eff/K - 1 from the decay of the m = 1 Fourier mode of 1 + cos(kx) (periodic, lam = 0, u = 1 along x, K = 100): the
    measurement style of scripts/verification.py (study numerical_diffusion), written against simulate()."""
    L, K, u = 10000.0, 100.0, 1.0
    sc = Scenario(domain=Domain(L, L, nx, 4), transport=Transport((WindSegment(0.0, u, 0.0),), K, 0.0, 500.0),
                  boundary=Boundary("periodic"), times=Times(T, dt, T))
    x = (np.arange(nx) + 0.5) * L / nx
    k = 2 * math.pi / L
    c = simulate(sc, [0, 0, 0], SolverOptions("exponential"), initial_field=np.tile(1 + np.cos(k * x), (4, 1)))
    a = abs(np.mean(c.concentration.values[-1] * np.exp(-1j * k * x)[None, :]))
    return float(-math.log(a / 0.5) / (k * k * T) / K - 1.0)


def test_numerical_diffusion_estimate_reproduces_documented_values() -> None:
    for dt, documented in ((60.0, 0.606), (10.0, 0.361)):  # docs/specification.md, 50 x 50 cells of 200 m, u = 1, K = 100
        est = numerical_diffusion_estimate(1.0, 200.0, dt, 100.0)
        assert est == pytest.approx(documented, rel=0.03)
        assert est == pytest.approx(measured_keff_excess(50, dt), rel=0.03)
    assert numerical_diffusion_estimate(0.0, 500.0, 100.0, 100.0) == 0.0
    # Belauri mesh: the excess is large at 500 m / 1.5 m/s and decreases with dx and dt
    e500, e250 = numerical_diffusion_estimate(1.5, 500.0, 100.0, 100.0), numerical_diffusion_estimate(1.5, 250.0, 50.0, 100.0)
    assert e500 > e250 > 0 and e500 > 3.0


def test_budget_closes_and_no_negative_values(region: Region) -> None:
    res = FIPY(case(region).scenario, case(region).theta)
    assert res.diagnostics["max_rel_budget_residual"] < 1e-9
    assert res.diagnostics["n_negative"] == 0
    b = res.budget
    assert b is not None and float(b["emission"].sum()) > 0 and float(b["outflow"].sum()) > 0


def test_streamfunction_demo_flow_conserves_a_constant_field(region: Region) -> None:
    c0 = 0.07
    centre = (12000.0, 13000.0)
    base = case(region, eddy_wind(247.5, 1.2, centre, 4000.0, 1500.0), lam=0.0, amplitudes={k: 0.0 for k in AMP})
    sc = dataclasses.replace(base.scenario, initial=Initial("uniform", c0), boundary=Boundary("open", c0))
    assert sc.transport.wind[0].psi is not None
    res = FIPY(sc, base.theta)
    assert float(np.abs(res.concentration.values - c0).max()) <= 1e-10
    assert res.diagnostics["max_rel_budget_residual"] <= 1e-9
    # the eddy is real: the face speeds differ from the uniform wind's
    uni = case(region, uniform_wind([(0.0, 247.5)], 1.2))
    assert to_dict(uni.scenario) != to_dict(base.scenario)


def test_wind_reversal_flips_inflow_faces_in_the_budget(region: Region) -> None:
    rev = uniform_wind([(0.0, 247.5), (T / 2, 67.5)], 1.5)
    bases = (InflowBasis("w", ("west",)), InflowBasis("e", ("east",)))
    out: dict[str, np.ndarray] = {}
    for name in ("w", "e"):
        amps = {**AMP, "w": 0.05 if name == "w" else 0.0, "e": 0.05 if name == "e" else 0.0}
        c = case(region, rev, amplitudes=amps, bases=bases)
        res = FIPY(c.scenario, c.theta)
        assert res.diagnostics["max_rel_budget_residual"] < 1e-9
        assert res.budget is not None
        inflow = np.asarray(res.budget["inflow"].values)
        n = len(inflow) // 2
        out[name] = np.array([inflow[:n].sum(), inflow[n:].sum()])
    # west faces are inflow before the reversal (wind toward ENE) and outflow after it; east faces the opposite
    assert out["w"][0] > 0 and out["e"][1] > 0
    # the other half carries no inflow of that basis (only the small diffusive loss through the other inflow faces, <= 0)
    assert out["w"][1] <= 1e-9 and out["e"][0] <= 1e-9


def test_wind_helpers_and_points(region: Region) -> None:
    u, v = toward_vector(247.5, 1.5)  # from WSW: toward ENE (+x, +y)
    assert u > 0 and v > 0 and math.hypot(u, v) == pytest.approx(1.5)
    assert toward_vector(0.0, 2.0) == pytest.approx((0.0, -2.0))  # from the north: toward the south
    f = build_wind("r", {"kind": "reversal", "from_deg": 247.5, "second_from_deg": 67.5}, 1.5, 7200.0, (0.0, 0.0))
    segs = f(region.domain(500.0), (0.0, 0.0))
    assert [s.t_start for s in segs] == [0.0, 3600.0] and segs[1].u == pytest.approx(-segs[0].u)
    with pytest.raises(ValueError, match="unknown kind"):
        build_wind("x", {"kind": "tornado", "from_deg": 0.0}, 1.0, 100.0, (0.0, 0.0))
    pts = select_points(region.level(500.0), (10609.0, 14416.0))
    assert len(pts) == 5 and len({p["label"] for p in pts}) == 5
    w = region.level(500.0).arrays["study_fraction"]
    for p in pts:
        assert w[int(p["y"] // 500), int(p["x"] // 500)] >= 0.5  # all five lie inside the polygon
    a = np.arange(16.0).reshape(1, 4, 4)
    assert block_average(a, 2)[0, 0, 0] == pytest.approx((0 + 1 + 4 + 5) / 4)


def test_plotting_only_layers_do_not_change_the_scenario(region: Region) -> None:
    chk = plotting_only_check(region, lambda r: case(r))
    assert chk["scenario_identical"] is True
    assert set(chk["layers"]) >= {"dem_mean", "slope_deg", "hillshade"}


def test_smoke_run_end_to_end_writes_all_files(tmp_path: Path) -> None:
    out = tmp_path / "run"
    rc = main(["benchmark", "--config", str(ROOT / "configs" / "belauri_forward_smoke.json"), "--out", str(out)])
    assert rc == 0
    expect = ["config.json", "provenance.json", "timings.json", "runs.csv", "wind_summary.csv", "consistency.json", "arrival_times.csv",
              "outer_domain.csv", "outer_domain_arrival.csv", "mesh_sensitivity.csv", "mesh_sensitivity_arrival.csv", "what_changes.csv",
              "what_changes.json", "points.json", "series/from_west_southwest.csv", "fields/reversal.nc",
              "figures/overview_map.png", "figures/arrival_traces.png", "figures/regional_summaries.png", "figures/what_changes.png",
              "figures/contributions_channelled_demo.png"]
    for name in expect:
        assert (out / name).stat().st_size > 0, name

    def no_nan(token: str) -> None:
        raise ValueError(f"non-finite JSON constant {token}")
    for p in out.glob("*.json"):
        json.loads(p.read_text(), parse_constant=no_nan)
    prov = json.loads((out / "provenance.json").read_text())
    ex = prov["extra"]
    assert ex["manifest_sha256"] and ex["resolution_m"] == 500.0 and prov["seeds"] == {}
    cons = json.loads((out / "consistency.json").read_text())
    for c in cons.values():
        assert c["max_rel_err_direct_vs_tag_sum"] < 1e-10 and c["max_rel_budget_residual"] < 1e-9
        assert c["n_negative_direct"] == 0 and c["max_courant"] < 0.4
    wc = json.loads((out / "what_changes.json").read_text())
    assert wc["plotting_only_layers"]["changes_any_number"] is False
    assert wc["plotting_only_layers"]["check"]["max_abs_change"] == 0.0
    with pytest.raises(FileExistsError):
        run_belauri_forward(load_config(ROOT / "configs" / "belauri_forward_smoke.json"), out)


def test_study_object_uses_manifest_hash_and_synthetic_amplitudes() -> None:
    st = Study(load_config(ROOT / "configs" / "belauri_forward_smoke.json"))
    assert len(st.manifest_sha256) == 64 and st.amplitudes == AMP
    assert st.region.georef.x0 > st.full.georef.x0  # the smoke standard domain is a crop
