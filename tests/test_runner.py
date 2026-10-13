"""pravah.runner: the shared guards keep the exact messages and file format the three population scripts used before."""
import json

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pravah import runner  # noqa: E402


def test_say_none_and_callback():
    got: list[str] = []
    runner.say(None, "x")
    runner.say(got.append, "y")
    assert got == ["y"]


def test_save_fig_writes_and_records(tmp_path):
    names: list[str] = []
    runner.save_fig(tmp_path, plt.figure(), "a.png", names)
    assert (tmp_path / "figures" / "a.png").exists() and names == ["a.png"]


def test_check_frozen(tmp_path):
    (tmp_path / "provenance.json").write_text(json.dumps({"frozen_config_hash": "h1"}))
    assert runner.check_frozen(tmp_path, "h1") == "h1"
    with pytest.raises(RuntimeError, match=r"freeze guard: frozen config hash changed \(h1 -> h2\)"):
        runner.check_frozen(tmp_path, "h2")


def test_guard_out_new_resume_and_mismatch(tmp_path):
    out = tmp_path / "o"
    rec = {"config_hash": "a", "profile": "smoke"}
    runner.guard_out(out, rec)
    assert json.loads((out / "run_guard.json").read_text()) == rec
    runner.guard_out(out, rec)  # resume with the same record is allowed
    with pytest.raises(RuntimeError, match="different frozen config or run selection"):
        runner.guard_out(out, {**rec, "profile": "full"})
    (out / "run_guard.json").unlink()
    (out / "stray.txt").write_text("x")
    with pytest.raises(RuntimeError, match="no run_guard.json"):
        runner.guard_out(out, rec)
