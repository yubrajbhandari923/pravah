"""Fast tests of the pure-numpy core of scripts/exp027b_sensitivity.py (synthetic layouts x cases matrices; no PDE data)."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import exp027b_sensitivity as S  # noqa: E402


def synth(L: int = 20, P: int = 24, shape_sd: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    base = rng.lognormal(0, 0.5, (L, 1)) * rng.lognormal(0, 0.3, (1, P))
    ja = base * rng.lognormal(0, 0.05, (L, P))
    jm = 1.3 * base * np.exp(shape_sd * rng.standard_normal((L, 1))) * rng.lognormal(0, 0.05, (L, P))
    return ja, jm


def test_matches_registered_statistics() -> None:
    import exp027b as X
    ja, jm = synth(shape_sd=0.1)
    assert S.E_of(ja, jm) == pytest.approx(X.stats(ja, jm)[1])
    assert S.jr_of(ja, jm) == pytest.approx(float(np.median(jm.mean(1) / ja.mean(1))))


def test_bootstrap_reproduces_registered_cell_table() -> None:
    import exp027b as X

    class Fake:
        def __init__(self, J: np.ndarray) -> None:
            self.J = J

        def j(self, f: str, ids: tuple[str, ...]) -> np.ndarray:
            return self.J[int(ids[0])]
    ja, jm = synth(L=12, P=18, shape_sd=0.1)
    menu = {("0.05", "3"): [(str(i),) for i in range(12)]}
    df = X.cell_table(menu, {"ref": Fake(ja), "main": Fake(jm)}, boot=60, seed=5)
    be, bj = S.boot_stats(ja, jm, S.boot_idx(np.random.default_rng(5), 60, 18))
    r = df[df.mesh == "main"].iloc[0]
    assert np.percentile(be, 2.5) == pytest.approx(r.E_lo95) and np.percentile(be, 97.5) == pytest.approx(r.E_hi95)


def test_intervals_and_classification() -> None:
    ja, jm = synth(shape_sd=0.15)
    r = S.summarize(ja, jm, 600, seed=1, types=("pct", "basic", "bca"))
    for t in ("pct", "basic", "bca"):
        assert r[f"E_lo95_{t}"] < r[f"E_hi95_{t}"]
    # basic = reflected percentile
    be, _ = S.boot_stats(ja, jm, S.boot_idx(np.random.default_rng(1), 600, ja.shape[1]))
    lo, hi = np.percentile(be, [2.5, 97.5])
    assert r["E_lo95_basic"] == pytest.approx(2 * r["E"] - hi)
    assert S.classify(0.06, 0.2) == "INADEQUATE" and S.classify(0.0, 0.05) == "ADEQUATE" and S.classify(0.03, 0.08) == "INCONCLUSIVE"


def test_influence_identifies_outlier_case() -> None:
    ja, jm = synth(shape_sd=0.02)
    jm = jm.copy()
    jm[:, 7] *= np.linspace(0.3, 3.0, jm.shape[0])  # one wild case
    inf = S.influence(ja, jm)
    assert int(np.argmax(np.abs(inf))) == 7


def test_debiasing_reduces_pure_noise_error() -> None:
    # no true shape error; only case noise: plain E is positive, the split-half cross-fit is smaller
    rng = np.random.default_rng(3)
    L, P = 25, 24
    base = rng.lognormal(0, 0.4, (L, 1)) * rng.lognormal(0, 0.6, (1, P))
    ja = base
    jm = base * rng.lognormal(0, 0.25, (L, P))
    assert S.debiased_E(ja, jm, 100, np.random.default_rng(0)) < S.E_of(ja, jm)
    lo, hi = S.debiased_interval(ja, jm, 80, np.random.default_rng(1))
    assert lo <= hi


def test_analyse_cell_rows() -> None:
    ja, jm = synth(shape_sd=0.1)
    P = ja.shape[1]
    o = argparse.Namespace(seeds=3, boot_reg=40, boot_big=200, boot=100, boot_deb=40, splits=10, top=2)
    draw_of = np.repeat([1, 2], P // 2)
    keep = np.arange(ja.shape[0]) % 2 == 0
    rows = S.analyse_cell(("0.05", "3"), {"ref": ja, "main": jm, "control": jm * 1.1}, draw_of, [f"c{i}" for i in range(P)], keep, o)
    kinds = {r["analysis"] for r in rows}
    assert {"A1", "A2", "A3", "A4", "A6"} <= kinds
    seeds = [r for r in rows if r["variant"].endswith("seeds") and r["mesh"] == "main"][0]
    assert 0 <= seeds["frac_inadequate"] <= 1 and seeds["E_lo95_min"] <= seeds["E_lo95_max"]
    a4 = [r for r in rows if r["analysis"] == "A4" and r["mesh"] == "main"][0]
    assert a4["n_layouts"] == int(keep.sum())


def test_assemble_menu_provenance() -> None:
    extra = [("ref", {"0.05": {"3": ["a", "b", "c"]}}), ("control", {"0.05": {"3": ["c", "b", "d"]}})]
    greedy = {(nm, (0, 1)): {"0.05": {"3": ["a", "b", "c"]}} for nm in ("ref", "main", "control")}
    menu = S.assemble(extra, greedy, [np.array([0, 1])], ["ref", "main", "control"])
    m = menu[("0.05", "3")]
    assert m[("a", "b", "c")] == {"ref", "main", "control"} and m[("b", "c", "d")] == {"control"}
