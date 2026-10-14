"""DYN-02 (EXP-030): historical source inversion M0 / M1 / M2 on synthetic dynamic episodes (toy geometry).

Stages (the test stage refuses to run without a hyperparameter freeze made by the tune stage):
    python -m pravah.studies.toy.dynamic_study --stage tune --out DIR   # validation seeds 6000-6999 only
    python -m pravah.studies.toy.dynamic_study --stage test --out DIR   # test seeds 7000-7999, hyperparameters frozen

Everything is synthetic.  Truth uses 12 fine bins (30 min); the inverse uses 6 bins (60 min), so half of all change
points fall between inverse bins.  The statistical unit is the episode.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from pravah import dynamic as dyn
from pravah.config import Boundary, Domain, GaussianSource, InflowBasis, Scenario, Times, Transport, WindSegment
from pravah.forward import simulate
from pravah.geo.regionio import GridSpec
from pravah.inverse import ResponseCache, Responses
from pravah.io import write_json
from pravah.population import pop_weighted_rmse, separated_settlements

DUR = 21600.0
N_INV, N_FINE = 6, 12
SRC_XY = ((3000.0, 6500.0), (6500.0, 3000.0), (9000.0, 6500.0))
WINDS = {"steady": ((0.0, 1.0, 0.3),), "rotation": ((0.0, 1.0, 0.3), (10800.0, 0.0, 1.0)),
         "reversal": ((0.0, 1.0, 0.2), (10800.0, -1.0, -0.2))}
FAMILIES = ("constant", "activation", "shutdown", "ramp", "opposing", "pulse", "influx_end", "compensating")
LAYOUTS = {"L4": ((6, 5), (6, 12), (17, 5), (17, 12)),  # (ix, iy) cell indices
           "L9": tuple((x, y) for x in (4, 12, 20) for y in (3, 9, 15))}
NOISE = (0.02, 0.10)  # fraction of the episode's domain-mean clean concentration
ALPHAS = (1e-6, 1e-4, 1e-2, 1.0, 100.0)  # grids widened twice on validation only (edge selections); final
BETAS = (0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0)
MODELS = ("M0", "M1", "M2")
SEEDS = {"tune": 6000, "test": 7100}  # 7000-7047 burned by invalid RUN-0060 (emission units bug)
N_SEEDS = {"tune": 2, "test": 2}  # seeds per (family, wind)


def scenario(wind: str) -> Scenario:
    srcs = tuple(GaussianSource(f"s{i}", x, y, 600.0) for i, (x, y) in enumerate(SRC_XY))
    bases = (InflowBasis("B_wsw", sides=("west", "south")), InflowBasis("B_ene", sides=("east", "north")))
    return Scenario(domain=Domain(lx=12000.0, ly=9000.0, nx=24, ny=18), sources=srcs,
                    transport=Transport(wind=tuple(WindSegment(*w) for w in WINDS[wind])),
                    boundary=Boundary(mode="open", bases=bases), times=Times(duration=DUR, dt=120.0, output_interval=600.0))


def population() -> np.ndarray:
    return separated_settlements(GridSpec(0.0, 0.0, 24, 18, 500.0, 500.0, 32645), 100_000, seed=0).astype(float)


def make_history(family: str, rng: np.random.Generator) -> np.ndarray:
    """(5, N_FINE) true history: rows s0, s1, s2 (ug/s), B_wsw, B_ene (ug/m^3)."""
    h = np.zeros((5, N_FINE))
    h[:3] = rng.uniform(0.6e5, 1.6e5, 3)[:, None]  # ug/s, the project's reference scale (configs/reference.json)
    k = int(rng.integers(3, 9))
    bamp = rng.uniform(0.05, 0.15)  # ug/m^3
    if family == "activation":
        h[0, :k] = 0.0
    elif family == "shutdown":
        h[1, k:] = 0.0
    elif family == "ramp":
        h[2] = h[2, 0] * np.linspace(1.0, 3.0, N_FINE)
    elif family == "opposing":
        h[0, k:] *= 3.0
        h[1, k:] *= 0.3
    elif family == "pulse":
        h[3, k:k + int(rng.integers(2, 4))] = bamp
    elif family == "influx_end":
        h[4, :k] = bamp
    elif family == "compensating":
        h[0, k:] *= 0.3
        h[3, k:] = bamp
    elif family != "constant":
        raise ValueError(family)
    return h


def episodes(stage: str) -> list[dict]:
    out = []
    for i, (w, f, _) in enumerate(itertools.product(WINDS, FAMILIES, range(N_SEEDS[stage]))):
        out.append({"id": i, "wind": w, "family": f, "seed": SEEDS[stage] + i})
    return out


def _cells(layout: str) -> tuple[np.ndarray, np.ndarray]:
    ix, iy = zip(*LAYOUTS[layout])
    return np.array(iy), np.array(ix)


_CACHE = ResponseCache(os.environ.get("PRAVAH_RESPONSE_CACHE", ".cache/responses"))  # key covers scenario, solver, params


def build_responses(wind: str) -> Responses:
    return _CACHE.get(dyn.expand(scenario(wind), N_INV))


def scale_check(R: Responses) -> float:
    """Guard (RUN-0060 lesson): typical emission and inflow signals must be of comparable size, else one kind of
    parameter is invisible and the study is meaningless.  Returns the local/boundary rms ratio of the typical truth."""
    loc = float(np.sqrt(np.mean(R.g[:3] ** 2))) * 1.1e5
    bnd = float(np.sqrt(np.mean(R.g[3:] ** 2))) * 0.1
    ratio = loc / bnd
    if not 0.1 <= ratio <= 10.0:
        raise RuntimeError(f"emission/inflow signal ratio {ratio:.3g} outside [0.1, 10]: check amplitude units")
    return ratio


def truth_field(wind: str, hist: np.ndarray) -> np.ndarray:
    sc = dyn.truth_scenario(scenario(wind), hist, dyn.bin_edges(DUR, N_FINE))
    return np.asarray(simulate(sc, np.ones(5)).concentration.values)


def run_episode(ep: dict, R: Responses, counts: np.ndarray,
                hp: dict[str, dict[str, list[tuple[float, float]]]]) -> list[dict]:
    """One row per (layout, noise, model, candidate hyperparameter); the truth run is shared by all of them."""
    rng = np.random.default_rng(ep["seed"])
    hist = make_history(ep["family"], rng)
    tf = truth_field(ep["wind"], hist)
    t = np.asarray(R.times)
    scale = float(tf.mean())
    binned = hist.reshape(5, N_INV, N_FINE // N_INV).mean(axis=2)
    rows = []
    for layout, frac in itertools.product(LAYOUTS, NOISE):
        iy, ix = _cells(layout)
        sd = frac * scale
        y = tf[1:, iy, ix] + np.random.default_rng(ep["seed"] * 10 + 1).standard_normal((len(t) - 1, len(iy))) * sd
        A = np.moveaxis(R.g[:, 1:, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
        sig = np.full(y.size, sd)
        cands = [("oracle", np.nan, np.nan)] + [(m, a, b) for m in MODELS for a, b in hp[f"{layout}|{frac}"][m]]
        for model, a, b in cands:
            th = binned if model == "oracle" else dyn.fit_history(A, y.reshape(-1), sig, 5, N_INV, model, alpha=a, beta=b).theta
            err = np.tensordot(th.reshape(-1), R.g, axes=1) - tf
            rows.append({"episode": ep["id"], "wind": ep["wind"], "family": ep["family"], "layout": layout, "noise": frac,
                         "model": model, "alpha": a, "beta": b,
                         "jpop": pop_weighted_rmse(err, counts, None, t),
                         "area": float(np.sqrt(np.trapezoid((err ** 2).mean(axis=(1, 2)), t) / (t[-1] - t[0]))),
                         "src_local": float(np.sqrt(np.mean((th[:3] - binned[:3]) ** 2))),
                         "src_bnd": float(np.sqrt(np.mean((th[3:] - binned[3:]) ** 2)))})
    return rows


def _all_rows(stage: str, hp) -> list[dict]:
    counts = population()
    cache = {w: build_responses(w) for w in WINDS}
    for w, R in cache.items():
        print(f"scale check {w}: local/boundary signal ratio {scale_check(R):.2f}")
    rows: list[dict] = []
    for ep in episodes(stage):
        rows += run_episode(ep, cache[ep["wind"]], counts, hp)
    return rows


def tune() -> dict:
    """Per (layout, noise, model) pick the hyperparameters with the lowest mean validation J_pop (seeds 6000-6999)."""
    grid = {"M0": [(a, 0.0) for a in ALPHAS], "M1": [(a, 0.0) for a in ALPHAS], "M2": list(itertools.product(ALPHAS, BETAS))}
    hp = {f"{lay}|{fr}": grid for lay in LAYOUTS for fr in NOISE}
    rows = [r for r in _all_rows("tune", hp) if r["model"] != "oracle"]
    scores: dict[str, dict] = {}
    for r in rows:
        scores.setdefault(f"{r['layout']}|{r['noise']}", {}).setdefault(r["model"], {}).setdefault((r["alpha"], r["beta"]), []).append(r["jpop"])
    mean = {k: {m: {c: float(np.mean(v)) for c, v in d.items()} for m, d in dm.items()} for k, dm in scores.items()}
    frozen = {k: {m: [min(d, key=d.get)] for m, d in dm.items()} for k, dm in mean.items()}
    return {"frozen": frozen, "validation_jpop": {k: {m: {f"{a}|{b}": s for (a, b), s in d.items()} for m, d in dm.items()}
                                                   for k, dm in mean.items()}}


def _hash(frozen: dict) -> str:
    return hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest()[:16]


def boot_ci(d: np.ndarray, n: int = 5000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def summarise(rows: list[dict]) -> list[dict]:
    out = []
    for lay, fr, metric in itertools.product(LAYOUTS, NOISE, ("jpop", "src_local", "src_bnd")):
        def vec(m):
            sel = sorted((r for r in rows if r["layout"] == lay and r["noise"] == fr and r["model"] == m),
                         key=lambda r: r["episode"])
            return np.array([r[metric] for r in sel])
        for a, b in (("M2", "M1"), ("M2", "M0"), ("M1", "M0")):
            d = vec(a) - vec(b)
            lo, hi = boot_ci(d)
            out.append({"layout": lay, "noise": fr, "metric": metric, "contrast": f"{a}-{b}", "n": len(d),
                        "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "frac_a_better": float((d < 0).mean()),
                        "mean_a": float(vec(a).mean()), "mean_b": float(vec(b).mean()), "mean_oracle": float(vec("oracle").mean())})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("tune", "test"), required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn02"))
    ap.add_argument("--freeze", default=None, help="path to frozen_hparams.json (test stage)")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "tune":
        r = tune()
        write_json(out / "frozen_hparams.json", {"hash": _hash(r["frozen"]), **r})
        print("frozen", _hash(r["frozen"]))
        return
    fz = json.loads(Path(a.freeze or out / "frozen_hparams.json").read_text())
    if fz["hash"] != _hash(fz["frozen"]):
        raise RuntimeError("freeze guard: hyperparameter file does not match its hash")
    hp = {k: {m: [tuple(c) for c in v] for m, v in d.items()} for k, d in fz["frozen"].items()}
    rows = _all_rows("test", hp)
    with open(out / "episodes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summ = summarise(rows)
    write_json(out / "summary.json", {"frozen_hash": fz["hash"], "n_episodes": len(episodes("test")), "summary": summ,
                                      "git_note": "synthetic only; test seeds 7000-7999"})
    for s in summ:
        if s["metric"] == "jpop" and s["contrast"] != "M1-M0":
            print(s["layout"], s["noise"], s["contrast"], f"{s['mean_diff']:+.4f} [{s['ci_lo']:+.4f},{s['ci_hi']:+.4f}]")


if __name__ == "__main__":
    main()
