"""EXP-015 stage 2 script: metrics, UTM candidate mapping across meshes, seed disjointness, cache key vs dt (no FiPy solves).
The end-to-end smoke profile (configs/convergence_stage2_smoke.json, ~3 min) is run by hand, see the script docstring."""
import copy
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from pravah import geodesign as G
from pravah.config import SolverOptions, load_config
from pravah.inverse import ResponseCache, cache_key

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "data" / "belauri" / "manifest.json"
spec = importlib.util.spec_from_file_location("convergence_stage2", ROOT / "scripts/convergence_stage2.py")
cs = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(cs)  # type: ignore[union-attr]

XY = {"a": (0.0, 0.0), "b": (1000.0, 0.0), "c": (0.0, 2000.0), "d": (5000.0, 5000.0)}


def test_compare_self_and_shift():
    s = cs.compare_layouts(["a", "b", "c"], ["c", "a", "b"], XY, (500.0, 1000.0))
    assert s["jaccard"] == 1 and s["mean_disp_m"] == 0 and s["max_disp_m"] == 0 and s["frac_within_500m"] == 1
    t = cs.compare_layouts(["a", "b"], ["a", "d"], XY, (500.0, 1000.0))  # b -> d optimal (only remaining match)
    assert t["jaccard"] == 1 / 3 and t["max_disp_m"] == pytest.approx(np.hypot(4000, 5000)) and t["frac_within_500m"] == 0.5


@pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing")
class TestWithGeography:
    @pytest.fixture(scope="class")
    def cfg(self):
        c = load_config(ROOT / "configs/convergence_stage2_smoke.json")
        c["region"]["manifest"] = str(MANIFEST)
        return c

    @pytest.fixture(scope="class")
    def table(self, cfg):
        c = copy.deepcopy(cfg)
        return G.build_candidate_table(G.load_geography(c), c)

    def test_mapping_consistent_across_meshes(self, cfg, table):
        geo = G.load_geography(cfg)
        m = cs.mapping_table(geo, table, [500.0, 250.0])
        assert (m["ix_250"] // 2 == m["ix_500"]).all() and (m["iy_250"] // 2 == m["iy_500"]).all()  # nested grids
        assert (m["centre_offset_m_250"] <= 250 * 2 ** 0.5 / 2 + 1e-6).all()
        # Setups on both meshes see the SAME sites in the same order, in the same UTM coordinates
        cache = ResponseCache()
        a, b = (cs.make_setup(cfg, r, dt, table, cache) for r, dt in ((500.0, 100.0), (250.0, 50.0)))
        assert a.cands.ids == b.cands.ids and np.array_equal(a.cands.x, b.cands.x)
        assert a.cands.meta and b.cands.meta and a.cands.meta["utm_x"] == b.cands.meta["utm_x"] == a.table["utm_x"].tolist()[: len(a.cands.ids)]
        with pytest.raises(AssertionError):
            cs.cell_lookup(geo.region.level(500.0).grid, np.array([-1e9]), np.array([0.0]))

    def test_replicate_seeds_disjoint(self, cfg):
        d = cs.draws_of(cfg)  # runs check_ensembles
        assert {"design", "validation", "test", "rep1", "rep2"} <= set(d)
        bad = copy.deepcopy(cfg)
        bad["stage2"]["replicates"]["seed0"] = 101  # replicate scenario seed == design seed
        with pytest.raises(ValueError, match="seeds must differ"):
            cs.draws_of(bad)
        bad = copy.deepcopy(cfg)
        bad["stage2"]["replicates"]["noise_seed0"] = 1000  # noise range overlaps the design ensemble
        with pytest.raises(ValueError, match="overlap"):
            cs.draws_of(bad)

    def test_cache_key_changes_with_dt_and_mesh(self, cfg, table):
        cache = ResponseCache()
        d = G.draw_cases(cfg, 1, 101, "d")[0]
        key = {mesh: cache_key(cs.make_setup(cfg, *mesh, table, cache).scenario(d).scenario, SolverOptions(), 1e6)
               for mesh in ((500.0, 100.0), (500.0, 50.0), (250.0, 100.0))}
        assert len(set(key.values())) == 3


@pytest.mark.skipif(not MANIFEST.exists(), reason="data/belauri/manifest.json is missing")
def test_workers_identical_to_sequential(tmp_path):
    """Tiny config (cheap meshes, 1 case per ensemble, 2 replicates, replicates also on the reference mesh): --workers 3 gives the
    same CSVs as --workers 1, and the extra refmesh baseline exists."""
    import json
    import subprocess
    import sys
    cfg = {"extends": str(ROOT / "configs/convergence_stage2_smoke.json"), "name": "s2_tiny",
           "ensembles": {k: {"n_cases": 1} for k in ("design", "validation", "test")},
           "stage2": {"meshes": [[500, 100], [500, 50]], "reference": [500, 50], "budgets": [1, 2], "map_budget": 2,
                      "replicates": {"n": 2, "n_cases": 1}, "replicates_on_reference": True}}
    (tmp_path / "c.json").write_text(json.dumps(cfg))
    for w in (1, 3):
        r = subprocess.run([sys.executable, str(ROOT / "scripts/convergence_stage2.py"), "--config", str(tmp_path / "c.json"),
                            "--out", str(tmp_path / f"w{w}"), "--workers", str(w)], cwd=ROOT, capture_output=True, text=True)
        assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    names = ["stability.csv", "layouts.csv", "replicate_baseline.csv", "replicate_baseline_refmesh.csv", "test_scores.csv"]
    for n in names:
        assert (tmp_path / "w1" / n).read_bytes() == (tmp_path / "w3" / n).read_bytes(), n
    assert not list((tmp_path / "w3").rglob(".*.tmp"))
