"""EXP-015 stage 1 script: aggregation, nested-grid assertion, resumability (no FiPy solves)."""
import csv
import importlib.util
from pathlib import Path

import numpy as np
import pytest

spec = importlib.util.spec_from_file_location("convergence_stage1", Path(__file__).resolve().parents[1] / "scripts/convergence_stage1.py")
cs = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(cs)  # type: ignore[union-attr]


def test_block_mean_conserves_mass():
    a = np.random.default_rng(0).random((3, 8, 12))
    b = cs.block_mean(a, 4)
    assert b.shape == (3, 2, 3)
    assert np.allclose(a.sum(axis=(1, 2)) / 16, b.sum(axis=(1, 2)))  # coarse cell has 16x the area: mean x area equal


def _centres(res, n0=4, shift=0.0):
    k = round(500 / res)
    return res * (np.arange(n0 * k) + 0.5) + shift, res * (np.arange((n0 + 1) * k) + 0.5) + shift


def test_nested_assertion():
    cs.assert_nested({500.0: _centres(500), 250.0: _centres(250), 125.0: _centres(125)})
    with pytest.raises(AssertionError):
        cs.assert_nested({500.0: _centres(500), 250.0: _centres(250, shift=10.0)})
    with pytest.raises(AssertionError):
        cs.assert_nested({500.0: _centres(500), 300.0: _centres(250)})


def test_resume_skips_finished(tmp_path):
    pl = cs.plan("smoke")
    calls = []

    def fake(key):
        calls.append(key)
        return dict(res_m=key[0], dt_s=key[1], scheme=key[2], group=key[3], wall_s=1, peak_rss_gb=0.1)

    csvp = tmp_path / "stage1_runs.csv"
    n = cs.execute(pl, csvp, fake)
    assert n == len(pl) == len(calls) == len(list(csv.DictReader(csvp.open())))
    assert cs.execute(pl, csvp, fake) == 0 and len(calls) == n
    rows = list(csv.DictReader(csvp.open()))
    csvp.write_text(csvp.read_text().replace(rows[-1]["scheme"], rows[-1]["scheme"], 1))
    lines = csvp.read_text().splitlines()
    csvp.write_text("\n".join(lines[:-1]) + "\n")  # drop the last row: only that key is rerun
    assert cs.execute(pl, csvp, fake) == 1


def test_orders_declare_non_monotone():
    assert cs.order_status(0.1, 0.2)[1].startswith("non-monotone")
    assert abs(cs.order_status(0.4, 0.1)[0] - 2.0) < 1e-12
    assert cs.richardson(1.0, 0.5, 0.75)[1].startswith("non-monotone")
    assert abs(cs.richardson(1.0, 0.5, 0.25)[0] - 1.0) < 1e-12
