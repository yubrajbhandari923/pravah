"""EXP-021 script: second returned-mass definition, not-yet-exited flag, margin guard, closure of a smoke case."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("pwc", ROOT / "scripts/prescribed_winds_converged.py")
pwc = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
spec.loader.exec_module(pwc)  # type: ignore[union-attr]

t = np.arange(0.0, 2000.0, 10.0)
hump = lambda c: np.exp(-((t - c) / 40.0) ** 2)  # noqa: E731
Z = np.zeros_like(t)


def test_two_humps_second_definition_and_flags():
    kw = dict(lam=0.0, H=1.0, dA=1.0, emitted=1.0, a_min=1e-3, exit_frac=0.02, floor_frac=0.02)
    two = pwc.passage_returns(t, hump(300) + 0.5 * hump(1200), Z, **kw)
    # the floor (2 % of P1) clips the foot of the later hump: a known low bias of about 9 % for a 0.5-height hump
    assert two["ret_mass_frac"] == pytest.approx(0.5, rel=0.12) and two["ret_eul_after_exit"] == pytest.approx(0.5, rel=0.05)
    assert not two["not_yet_exited"] and 300 < two["t_exit_s"] < 500 and not two["end_inside"]
    one = pwc.passage_returns(t, hump(300), Z, **kw)
    assert one["ret_mass_frac"] < 1e-6
    cut = pwc.passage_returns(t[:35], hump(300)[:35], Z[:35], **kw)  # plume still in the box when the run stops
    assert cut["not_yet_exited"] and np.isnan(cut["ret_mass_frac"]) and cut["end_inside"]


def _cfg():
    cfg = json.loads((ROOT / "configs/prescribed_winds_converged.json").read_text())
    cfg.update(cfg["profiles"]["smoke"])
    return cfg


def test_margin_guard_and_geometry():
    cfg = _cfg()
    pwc.check_margins(cfg)
    cfg["domain"]["half_widths_m"] = [15000.0, 60000.0]  # source 3 km from the west boundary
    with pytest.raises(ValueError):
        pwc.check_margins(cfg)
    with pytest.raises(ValueError):
        pwc.geometry(_cfg(), 22501.0)


def test_smoke_cases_close_and_reference_independent_of_geometry_shift():
    cfg = _cfg()
    r = pwc.run_case(cfg, "channel_step", 22500.0, None)
    assert r["closure_rel"] < 1e-8 and r["min_conc"] > -1e-9 * r["max_conc"]
    assert sum(r[f"leave_{s}_frac"] for s in pwc.SIDES) + r["emitted_frac_removed"] + r["remaining_ug"] / r["emitted_ug"] == pytest.approx(1.0, abs=0.02)
