"""DYN-06 (EXP-034): can a changing local source be told apart from a transboundary influx event?  (code + smoke only)

Everything is linear, y = A theta + noise, so two complementary analyses per (wind, layout, noise):
  (1) analytic distinguishability of two scenarios (d in noise-sd units, raw and after the best nonnegative rescaling of b, whitened
      singular values, principal-angle cosines of the changing-parameter column spaces);
  (2) noisy attribution: simulate scenario a with noise replicates, fit M2 (dynamic.fit_history), classify which group changed.

Scenarios S1-S7 are defined in ``build``.  Noise streams use rng seeds 9000-9999 ONLY (replicate r -> seed 9000 + r, so at most 1000
replicates); seeds 7000-8999 are never used.  Hyperparameters (alpha, beta) default to documented PLACEHOLDERS, not tuned values.
Exploratory / plumbing code: nothing here is a confirmatory result.  Everything is synthetic.

    python -m pravah.studies.toy.dynamic_separability --smoke
    python -m pravah.studies.toy.dynamic_separability --out DIR [--reps 200] [--workers 4]
"""
from __future__ import annotations

import argparse
import csv
import functools
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pravah import dynamic as dyn
from pravah.studies.toy import dynamic_study as ds
from pravah.studies.toy.dynamic_study import LAYOUTS, N_INV, NP, REF, WINDS

DEFAULT_HP = (1e3, 1e1)  # (alpha, beta): documented placeholders for M2, NOT frozen/tuned values; pass frozen ones via ``hp``
SEED0, MAX_REPS = 9000, 1000  # replicate r uses rng seed 9000 + r  (range 9000-9999 only)
TIE_REL = 1e-3  # two group contributions within this relative gap are 'ambiguous' (counted as a coin flip = 0.5 error)
SPLIT = N_INV // 2  # change at bin 3
LOCAL_BOUNDARY = {"local": (0, 1, 2), "boundary": (3, 4)}
NOISE = ds.NOISE


@dataclass
class Op:
    A: np.ndarray  # (n_obs, 30) native units; rows time-major then sensor (dynamic_study.run_episode's construction)
    Ax: np.ndarray  # A * repeat(REF, 6): operator on REF-unit histories
    sigma: float  # per-observation noise sd
    rb: np.ndarray  # (n_obs,) temporal bin of each row
    live: np.ndarray  # (5, 6) bool: whitened column norm above 1e-6 of the max
    R: object = None  # Responses (None for synthetic fixtures)


def make_op(wind: str, cells, noise: float, R=None) -> Op:
    R = ds.build_responses(wind) if R is None else R
    iy, ix = ds._cells(tuple(cells))
    A = np.moveaxis(R.g[:, 1:, iy, ix], 0, -1).reshape(-1, R.g.shape[0])
    sigma = noise * ds.reference_scale(R)
    t = np.asarray(R.times)[1:]
    rb = np.repeat(np.minimum(((t - 1e-6) // (ds.DUR / N_INV)).astype(int), N_INV - 1), len(iy))
    return _op(A, sigma, rb, R)


def _op(A, sigma, rb, R=None) -> Op:
    Ax = A * np.repeat(REF, N_INV)
    sens = dyn.sensitivity(Ax, np.full(len(A), sigma), NP, N_INV)
    return Op(A, Ax, float(sigma), rb, sens > 1e-6 * sens.max(), R)


# ---------------------------------------------------------------- analytic distinguishability
def _basis(M: np.ndarray, cols: np.ndarray) -> np.ndarray:
    U, s, _ = np.linalg.svd(M[:, cols], full_matrices=False)
    return U[:, s > 1e-9 * max(s.max(), 1e-300)] if s.size and s.max() > 0 else U[:, :0]


def _changing_cols(op: Op, x: np.ndarray) -> np.ndarray:
    """Columns of the parameters whose history is non-constant (all constant -> every nonzero parameter); dead columns dropped."""
    par = np.where(np.ptp(x, axis=1) > 0)[0]
    if not len(par):
        par = np.where(x.any(axis=1))[0]
    cols = np.array([j * N_INV + b for j in par for b in range(N_INV) if op.live[j, b]], dtype=int)
    return cols


def rescale_distance(u: np.ndarray, v: np.ndarray, sigma: float) -> tuple[float, float]:
    """(min_{c>=0} ||u - c v|| / sigma, c*)."""
    vv = float(v @ v)
    c = max(0.0, float(u @ v) / vv) if vv > 0 else 0.0
    return float(np.linalg.norm(u - c * v) / sigma), c


def analytic(op_a: Op, op_b: Op, xa: np.ndarray, xb: np.ndarray) -> dict:
    """Scenarios given as REF-unit histories (5, 6).  Cross-wind pairs use the mean of the two noise sds, so d is symmetric."""
    u, v = op_a.Ax @ xa.ravel(), op_b.Ax @ xb.ravel()
    sig = 0.5 * (op_a.sigma + op_b.sigma)
    d_after, c = rescale_distance(u, v, sig)
    sv = np.linalg.svd(op_a.Ax / op_a.sigma, compute_uv=False)
    Qa, Qb = _basis(op_a.Ax, _changing_cols(op_a, xa)), _basis(op_b.Ax, _changing_cols(op_b, xb))
    cs = np.linalg.svd(Qa.T @ Qb, compute_uv=False) if Qa.shape[1] and Qb.shape[1] else np.array([np.nan])
    return {"d_raw": float(np.linalg.norm(u - v) / sig), "d_after_rescale": d_after, "c_star": c,
            "sv_max": float(sv[0]), "sv_min": float(sv[-1]), "n_sv_gt1": int((sv > 1).sum()),
            "cos_max": float(np.max(cs)), "cos_min": float(np.min(cs))}


# ---------------------------------------------------------------- scenarios
def _base() -> np.ndarray:
    x = np.zeros((NP, N_INV))
    x[:3] = 1.0  # locals at their REF amplitude, boundary off
    return x


def _pattern(j: int, bins) -> np.ndarray:
    p = np.zeros((NP, N_INV))
    p[j, list(bins)] = 1.0
    return p


def _chg_obs(op: Op, x: np.ndarray) -> float:
    """Observed sensor-mean concentration change: mean over rows in bins >= 3 minus mean over rows in bins < 3 (linear in x)."""
    return float((op.Ax[op.rb >= SPLIT].mean(0) - op.Ax[op.rb < SPLIT].mean(0)) @ x.ravel())


def _pick_b(op: Op, bins) -> int | None:
    """B_wsw (3) if live in all requested bins, else B_ene (4), else None (wind offers no live window)."""
    return next((j for j in (3, 4) if op.live[j, list(bins)].all()), None)


def _matched(op: Op, xa: np.ndarray, xb0: np.ndarray, j: int, bins) -> tuple[np.ndarray, bool]:
    """xb0 plus boundary j on ``bins`` with the amplitude (REF units) that makes b's observed change equal a's."""
    p = _pattern(j, bins)
    den = _chg_obs(op, p)
    c = (_chg_obs(op, xa) - _chg_obs(op, xb0)) / den if abs(den) > 1e-300 else np.nan
    ok = bool(np.isfinite(c) and c > 0)
    return xb0 + (c if ok else 0.0) * p, ok


def build(name: str, op: Op, op_b: Op | None = None) -> dict:
    """Scenario pair for ``op`` (op_b only for S3).  Returns dict(name, op_a, op_b, xa, xb, groups, valid, match_ok).
    xa is the scenario simulated for the attribution (the 'event'); xb its look-alike.  Histories in REF units, change at bin 3.
      S1 a: s0 x3 from bin 3                       b: boundary 0 -> c (bins 3-5), c matched to a's observed sensor-mean change
      S2 a: s0 off from bin 3                      b: boundary on bins 0-2 then off, locals constant, c matched
      S3 a: wind A, constant emissions (op)        b: wind B, s1 0 -> 1 at bin 3 (op_b)   [cross-wind; no attribution]
      S4 a: s0 x3 and s1 -> 0.3 from bin 3         b: a + boundary pulse (bins 3-4, 1 REF)
      S5 a: boundary 1 REF on bins 0-2 then off    b: locals constant           (a = influx ends, plume persists)
      S6 a: s0 x3 from bin 3                       b: s2 x3 from bin 3          (local vs local; groups s0 / s2)
    The boundary is B_wsw if live in the required bins, else B_ene; no live window -> valid False."""
    xa, ok = _base(), True
    sc = dict(name=name, op_a=op, op_b=op, groups=LOCAL_BOUNDARY, valid=True, match_ok=True)
    xb = _base()
    if name == "S1":
        xa[0, SPLIT:] = 3.0
        j = _pick_b(op, range(SPLIT, N_INV))
        if j is not None:
            xb, ok = _matched(op, xa, xb, j, range(SPLIT, N_INV))
    elif name == "S2":
        xa[0, SPLIT:] = 0.0
        j = _pick_b(op, range(SPLIT))
        if j is not None:
            xb, ok = _matched(op, xa, xb, j, range(SPLIT))
    elif name == "S3":
        xb[1, :SPLIT] = 0.0
        sc.update(op_b=op_b, groups=None)
        j = 0
    elif name == "S4":
        xa[0, SPLIT:], xa[1, SPLIT:] = 3.0, 0.3
        j = _pick_b(op, (3, 4))
        xb = xa + (_pattern(j, (3, 4)) if j is not None else 0)
    elif name == "S5":
        j = _pick_b(op, range(SPLIT))
        xb = _base()
        xa = xb + (_pattern(j, range(SPLIT)) if j is not None else 0)
    elif name == "S6":
        xa[0, SPLIT:] = 3.0
        xb[2, SPLIT:] = 3.0
        sc["groups"] = {"s0": (0,), "s2": (2,)}
        j = 0
    else:
        raise ValueError(name)
    sc.update(xa=xa, xb=xb, valid=j is not None, match_ok=ok)
    return sc


# ---------------------------------------------------------------- noisy attribution
def _contrib(op: Op, xx: np.ndarray, groups: dict) -> np.ndarray:
    """Per group: sum_j (mean_{b>=3} x_j - mean_{b<3} x_j) * s_j, s_j = sensor-time mean concentration of a constant unit-REF j."""
    s = op.Ax.reshape(len(op.Ax), NP, N_INV).sum(2).mean(0)
    d = xx[:, SPLIT:].mean(1) - xx[:, :SPLIT].mean(1)
    return np.array([(d * s)[list(g)].sum() for g in groups.values()])


@functools.lru_cache(maxsize=1)
def _counts() -> np.ndarray:
    return ds.population()


def attribute(op: Op, x: np.ndarray, groups: dict = LOCAL_BOUNDARY, hp: tuple[float, float] = DEFAULT_HP, reps: int = 200) -> dict:
    """Simulate y = Ax + N(0, sigma^2) (replicate r: seed 9000 + r), fit M2 (zero prior mean), label which group changed.
    truth label = group with the largest |true contribution| ('none' if all zero); estimate label 'none' if every |contribution| < sigma,
    'ambiguous' if the top two are within TIE_REL.  conf_rate = (wrong + 0.5 * ambiguous) / fitted.  Statistical unit: the replicate."""
    if not 0 < reps <= MAX_REPS:
        raise ValueError("reps must be in 1..1000 (noise seeds 9000-9999 only)")
    names = list(groups)
    ct = np.abs(_contrib(op, x, groups))
    truth = names[int(ct.argmax())] if ct.max() > 0 else "none"
    y0, sig = op.Ax @ x.ravel(), np.full(len(op.Ax), op.sigma)
    loc_true = None
    if op.R is not None:
        xl = x.copy(); xl[3:] = 0
        loc_true = np.tensordot((xl * REF[:, None]).ravel(), op.R.g, axes=1)
        norm0 = ds.jpop(op.R, np.zeros(NP * N_INV), loc_true, _counts())[0]
    labels, jp = [], []
    for r in range(reps):
        y = y0 + np.random.default_rng(SEED0 + r).standard_normal(len(y0)) * op.sigma
        try:
            th = dyn.fit_history(op.A, y, sig, NP, N_INV, "M2", alpha=hp[0], beta=hp[1], ref=REF).theta
        except dyn.NotConverged:
            continue
        ce = np.abs(_contrib(op, th / REF[:, None], groups))
        top = np.sort(ce)[::-1]
        labels.append("none" if top[0] < op.sigma else "ambiguous" if len(top) > 1 and top[0] - top[1] < TIE_REL * top[0]
                      else names[int(ce.argmax())])
        if loc_true is not None:
            tl = th.copy(); tl[3:] = 0
            jp.append(ds.jpop(op.R, tl, loc_true, _counts())[0] / norm0 if norm0 > 0 else np.nan)
    n = len(labels)
    lab = np.array(labels)
    wrong = float(((lab != truth) & (lab != "ambiguous")).sum() + 0.5 * (lab == "ambiguous").sum())
    return {"truth": truth, "n_rep": n, "n_fail": reps - n, "conf_rate": wrong / n if n else np.nan,
            "frac_none": float((lab == "none").mean()) if n else np.nan, "frac_ambiguous": float((lab == "ambiguous").mean()) if n else np.nan,
            "jpop_local_rel": float(np.median(jp)) if jp else np.nan}


# ---------------------------------------------------------------- registered controls
def fixture_confounded(seed: int = 9999, n_rows: int = 72) -> Op:
    """Synthetic operator where B_wsw's bin-b column equals s0's (identical in REF units): perfectly confounded.  Seed in 9000-9999."""
    rng = np.random.default_rng(seed)
    Ax = rng.uniform(0.0, 5.0, (n_rows, NP * N_INV))
    Ax[:, 3 * N_INV:4 * N_INV] = Ax[:, 0:N_INV]
    return _op(Ax / np.repeat(REF, N_INV), 1.0, np.repeat(np.arange(N_INV), n_rows // N_INV))


def confounded_pair() -> tuple[np.ndarray, np.ndarray, dict]:
    """(xa, xb, groups): a = s0 on at 3 REF from bin 3, b = boundary (identical column to s0) at 1 REF from bin 3; groups s0 vs B_wsw."""
    xa, xb = np.zeros((NP, N_INV)), np.zeros((NP, N_INV))  # no baseline: rescaling the whole of b must be able to match a
    xa[0, SPLIT:] = 3.0
    xb[3, SPLIT:] = 1.0
    return xa, xb, {"s0": (0,), "B_wsw": (3,)}


def fixture_no_variation(op: Op, hp=DEFAULT_HP, reps: int = 200) -> dict:
    """Constant emissions, nothing changes: truth label is 'none'; frac_none is the rate at which the estimator also reports no change."""
    return attribute(op, _base(), LOCAL_BOUNDARY, hp, reps)


# ---------------------------------------------------------------- l4 layouts for S7
def l4_ref_cells(wind: str) -> tuple[tuple[int, int], ...]:
    """3 L4 interior cells plus one near the upwind boundary of the wind's FIRST segment: |u| >= |v| -> west (ix 1, u>0) or east (ix 22) side
    at iy 9; else south (iy 1, v>0) or north (iy 16) at ix 12.  The inflow bases are W+S (B_wsw) and E+N (B_ene)."""
    _, u, v = WINDS[wind][0]
    ref = (1 if u > 0 else 22, 9) if abs(u) >= abs(v) else (12, 1 if v > 0 else 16)
    return LAYOUTS["L4"][:3] + (ref,)


# ---------------------------------------------------------------- driver
_KEYS = ("d_raw", "d_after_rescale", "c_star", "sv_max", "sv_min", "n_sv_gt1", "cos_max", "cos_min")
_ATT = ("truth", "n_rep", "n_fail", "conf_rate", "frac_none", "frac_ambiguous", "jpop_local_rel")


def evaluate(sc: dict, meta: dict, hp=DEFAULT_HP, reps: int = 200) -> dict:
    row = {**meta, "scenario": meta.get("scenario", sc["name"]), "valid": sc["valid"], "match_ok": sc["match_ok"]}
    nan = float("nan")
    row.update({k: nan for k in _KEYS + _ATT})
    if not sc["valid"]:
        return row
    row.update(analytic(sc["op_a"], sc["op_b"], sc["xa"], sc["xb"]))
    if sc["groups"] is not None and reps:
        row.update(attribute(sc["op_a"], sc["xa"], sc["groups"], hp, reps))
    return row


def _wind_rows(wind: str, noises, hp, reps: int) -> list[dict]:
    rows = []
    R = ds.build_responses(wind)
    lays = {"L4": LAYOUTS["L4"], "L9": LAYOUTS["L9"], "L4_interior": LAYOUTS["L4"], "L4_ref": l4_ref_cells(wind)}
    for fr in noises:
        for lay, cells in lays.items():
            op = make_op(wind, cells, fr, R)
            if lay.startswith("L4_"):  # S7: the S1 pair at equal budget 4, interior vs external-reference layout
                rows.append(evaluate(build("S1", op), dict(scenario="S7", wind=wind, layout=lay, noise=fr), hp, reps))
                continue
            for s in ("S1", "S2", "S4", "S5", "S6"):
                rows.append(evaluate(build(s, op), dict(wind=wind, layout=lay, noise=fr), hp, reps))
    return rows


def _pair_rows(wa: str, wb: str, noises, hp) -> list[dict]:
    rows = []
    for fr in noises:
        for lay in ("L4", "L9"):
            sc = build("S3", make_op(wa, LAYOUTS[lay], fr), make_op(wb, LAYOUTS[lay], fr))
            rows.append(evaluate(sc, dict(wind=f"{wa}|{wb}", layout=lay, noise=fr), hp, 0))
    return rows


def _task(args):
    kind, a = args
    return _wind_rows(*a) if kind == "wind" else _pair_rows(*a)


def run_grid(winds=tuple(WINDS), noises=NOISE, hp=DEFAULT_HP, reps: int = 200, workers: int = 1,
             s3_pairs=(("rotation_a", "steady_a"), ("rotation_b", "steady_b"))) -> list[dict]:
    tasks = [("wind", (w, noises, hp, reps)) for w in winds]
    tasks += [("pair", (a, b, noises, hp)) for a, b in s3_pairs if a in winds or b in winds]
    if workers <= 1:
        return [r for t in tasks for r in _task(t)]
    with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
        return [r for rs in ex.map(_task, tasks) for r in rs]


def summarise(rows: list[dict]) -> dict:
    """Per scenario x layout x noise: fraction of winds (pairs for S3) with d_after_rescale > 3, median d, confusion range,
    and cannot_separate = True when d_after_rescale < 2 for every valid wind."""
    out: dict = {}
    keys = sorted({(r["scenario"], r["layout"], r["noise"]) for r in rows})
    for k in keys:
        rs = [r for r in rows if (r["scenario"], r["layout"], r["noise"]) == k and r["valid"]]
        if not rs:
            out["|".join(map(str, k))] = {"n_valid_winds": 0}
            continue
        d = np.array([r["d_after_rescale"] for r in rs])
        cf = np.array([r["conf_rate"] for r in rs], dtype=float)
        cf = cf[np.isfinite(cf)]
        out["|".join(map(str, k))] = {"n_valid_winds": len(rs), "frac_winds_d_gt3": float((d > 3).mean()), "median_d_after_rescale": float(np.median(d)),
                                      "median_d_raw": float(np.median([r["d_raw"] for r in rs])),
                                      "conf_rate_range": [float(cf.min()), float(cf.max())] if len(cf) else None,
                                      "cannot_separate": bool((d < 2).all())}
    return out


def write_csv(rows: list[dict], path: Path) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def _print(rows: list[dict]) -> None:
    print(f"{'scen':5}{'wind':24}{'layout':12}{'noise':6}{'valid':6}{'d_raw':>8}{'d_resc':>8}{'cosmax':>8}{'cosmin':>8}{'conf':>6}{'none':>6}{'jpopL':>7}")
    for r in rows:
        print(f"{r['scenario']:5}{r['wind']:24}{r['layout']:12}{r['noise']:<6}{str(r['valid']):6}{r['d_raw']:8.2f}{r['d_after_rescale']:8.2f}"
              f"{r['cos_max']:8.3f}{r['cos_min']:8.3f}{r['conf_rate']:6.2f}{r['frac_none']:6.2f}{r['jpop_local_rel']:7.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="reversal wind, noise 0.1, 20 replicates, plumbing only")
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn06"))
    ap.add_argument("--reps", type=int, default=200)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=DEFAULT_HP[0])
    ap.add_argument("--beta", type=float, default=DEFAULT_HP[1])
    a = ap.parse_args()
    hp = (a.alpha, a.beta)
    if a.smoke:
        rows = _wind_rows("reversal", (0.10,), hp, 20)
        rows = [r for r in rows if r["layout"] in ("L4", "L4_interior", "L4_ref")]
        rows += [r for r in _pair_rows("rotation_a", "steady_a", (0.10,), hp) if r["layout"] == "L4"]  # S3 is cross-wind
        _print(rows)
        return
    rows = run_grid(hp=hp, reps=a.reps, workers=a.workers)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out / "separability.csv")
    import json
    (out / "summary.json").write_text(json.dumps({"hp": hp, "reps": a.reps, "noise_seeds": "9000-9999", "summary": summarise(rows)}, indent=1))
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
