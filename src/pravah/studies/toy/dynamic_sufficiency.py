"""DYN-05 / P0 (EXP-033): how many independent design episodes does a stable sensor design need?

Design objective of a layout S (empirical-risk design, ridge estimator with prior x ~ N(0, I) in x = theta/REF units):
    J_D(S) = (1/N) sum_e  E_noise || M ( theta_hat_e(S) - x_e ) ||^2,   M = sqrt(time x population weights) * G * REF
For the ridge estimator the noise expectation is closed form and depends on the design episodes ONLY through the per-wind
second-moment matrices C_w = mean(x_e x_e^T):
    J_w(S; C) = tr( Q H^-1 [ C + H - I ] H^-1 ),   H = I + sum_{c in S} A_c^T A_c / sigma_w^2,   Q = M^T M   (Lambda = I)
so exhaustive search is exact and cheap for any ensemble.  An 'ensemble' is N episodes (wind uniform over the 6 winds, family uniform
over the 8, random amplitudes - the dynamic_study generator) drawn from independent streams SeedSequence([5, N, r, e]); the
REFERENCE objective uses 4000 episodes from SeedSequence([6, e]).  Selection regret = J_ref(S_N) - J_ref(S*), S* = reference optimum.
Real-estimator check: the chosen layouts are evaluated with the actual nonnegative ridge estimator (dynamic_study run_episode, M1z)
on 96 fresh test episodes (seeds 8400-8495).

    python -m pravah.studies.toy.dynamic_sufficiency --stage design --out DIR      # no test data touched
    python -m pravah.studies.toy.dynamic_sufficiency --stage test --out DIR        # evaluates the hash-frozen layouts
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path

import numpy as np

from pravah.io import write_json
from pravah.studies.toy import dynamic_study as ds

NS = (6, 12, 24, 48, 96, 192)
R_ENS = 16
KS = (2, 3)
N_REF = 20000  # amended 2026-10-13 (D-018): A selects S*, an independent B measures every regret
CANDS = tuple((ix, iy) for ix in range(2, 22, 2) for iy in range(2, 16, 2))  # 70 cells, as DYN-04
TEST_SEED0 = 8400
CHUNK = 4096


def episode_moments(n: int, stream: tuple, lives: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Per-wind second moment (30x30, x units) and counts of n episodes drawn from SeedSequence([*stream, e])."""
    winds, fams = list(ds.WINDS), ds.FAMILIES
    C = {w: np.zeros((30, 30)) for w in winds}
    cnt = dict.fromkeys(winds, 0)
    for e in range(n):
        rng = np.random.default_rng(np.random.SeedSequence([*stream, e]))
        w, fam = winds[int(rng.integers(len(winds)))], fams[int(rng.integers(len(fams)))]
        hist, _ = ds.make_history(fam, rng, lives[w])
        x = (hist.reshape(ds.NP, ds.N_INV, -1).mean(axis=2) / ds.REF[:, None]).reshape(-1)
        C[w] += np.outer(x, x)
        cnt[w] += 1
    return C, cnt


def wind_operators(noise: float) -> dict[str, dict]:
    """Per wind: candidate Gram matrices A_c^T A_c / sigma^2 (70,30,30) and Q = M^T M (30,30), in x units."""
    counts = ds.population()
    out = {}
    for w in ds.WINDS:
        R = ds.build_responses(w)
        s = noise * ds.reference_scale(R)
        g = R.g * np.repeat(ds.REF, ds.N_INV)[:, None, None, None]  # per-x-unit response
        Ac = np.stack([g[:, 1:, iy, ix].reshape(30, -1).T for ix, iy in CANDS])  # (70, 36, 30)
        wt = np.sqrt(np.outer(ds._time_weights(np.asarray(R.times)), (counts / counts.sum()).reshape(-1))).reshape(37, 18, 24)
        Mrows = (g * wt[None]).reshape(30, -1).T
        out[w] = {"G": np.einsum("cti,ctj->cij", Ac, Ac) / s ** 2, "Q": Mrows.T @ Mrows}
    return out


def layout_batches(k: int):
    it = itertools.combinations(range(len(CANDS)), k)
    while True:
        b = np.array(list(itertools.islice(it, CHUNK)))
        if not len(b):
            return
        yield b


def risks_all(ops: dict[str, dict], k: int, moments: list[tuple[dict, dict, int]]) -> np.ndarray:
    """(n_ensembles, n_layouts) objective J_D(S) for every layout of size k and every ensemble (C_w, counts_w, N)."""
    n_lay = int(np.prod([len(CANDS) - i for i in range(k)]) / np.prod(range(1, k + 1)))
    out = np.zeros((len(moments), n_lay))
    off = 0
    I = np.eye(30)
    for b in layout_batches(k):
        for w, op in ops.items():
            H = I + op["G"][b].sum(axis=1)
            Hi = np.linalg.inv(H)
            P = Hi @ op["Q"] @ Hi
            noise_term = np.einsum("bij,bji->b", P, H - I)
            for e, (C, cnt, n) in enumerate(moments):
                if cnt[w]:
                    out[e, off:off + len(b)] += (cnt[w] / n) * (np.einsum("bij,ji->b", P, C[w] / cnt[w]) + noise_term)
        off += len(b)
    return out


def layouts_of(k: int) -> np.ndarray:
    return np.array(list(itertools.combinations(range(len(CANDS)), k)))


def jaccard(a: tuple, b: tuple) -> float:
    return len(set(a) & set(b)) / len(set(a) | set(b))


def design(noises=ds.NOISE) -> dict:
    lives = {w: ds.live_mask(ds.build_responses(w)) for w in ds.WINDS}
    doc: dict = {"cands": [list(c) for c in CANDS], "Ns": list(NS), "R": R_ENS, "results": {}}
    for noise in noises:
        ops = wind_operators(noise)
        for k in KS:
            lay = layouts_of(k)
            ens = [("refA", episode_moments(N_REF, (6,), lives) + (N_REF,)), ("refB", episode_moments(N_REF, (7,), lives) + (N_REF,))]
            keys = [(n, r) for n in NS for r in range(R_ENS)]
            ens += [((n, r), (*episode_moments(n, (5, n, r), lives), n)) for n, r in keys]
            J = risks_all(ops, k, [m for _, m in ens])
            jA, jref, jn = J[0], J[1], J[2:]  # jref = evaluation reference B
            star = int(np.argmin(jA))
            starB = int(np.argmin(jref))
            rec = {"star": lay[star].tolist(), "j_star": float(jref[star]), "j_ref_median_layout": float(np.median(jref)),
                   "j_ref_worst": float(jref.max()), "selected": {}, "eps_set": {},
                   "starB": lay[starB].tolist(), "star_on_B_vs_best_B": float(jref[star] / jref[starB] - 1.0),
                   "starB_vs_starA_regret_on_B": float(jref[starB] / jref[star] - 1.0), "jaccard_starA_starB": jaccard(tuple(lay[star]), tuple(lay[starB]))}
            # layouts within 5% of the reference optimum are "equivalent" (reported; an acknowledged arbitrary band)
            rec["n_within_5pct_of_star"] = int((jref <= 1.05 * jref[star]).sum())
            for (n, r), row in zip(keys, jn):
                s = int(np.argmin(row))
                rec["selected"].setdefault(str(n), []).append({"layout": lay[s].tolist(), "regret": float(jref[s] - jref[star]),
                                                               "rel_regret": float(jref[s] / jref[star] - 1.0), "j_design": float(row[s])})
            doc["results"][f"{noise:g}|{k}"] = rec
    doc["hash"] = hashlib.sha256(json.dumps(doc["results"], sort_keys=True).encode()).hexdigest()[:16]
    return doc


def summarise(doc: dict) -> list[dict]:
    rows = []
    for key, rec in doc["results"].items():
        noise, k = key.split("|")
        star = tuple(rec["star"])
        for n, sel in rec["selected"].items():
            rel = np.array([s["rel_regret"] for s in sel])
            lays = [tuple(s["layout"]) for s in sel]
            cells = np.array([CANDS[i] for i in star])
            disp = []
            for s in lays:
                d = np.linalg.norm(np.array([CANDS[i] for i in s])[:, None] - cells[None], axis=2)
                disp.append(float(np.mean(d.min(axis=1))))
            jac = [jaccard(a, b) for a, b in itertools.combinations(lays, 2)]
            rows.append({"noise": float(noise), "k": int(k), "N": int(n), "mean_rel_regret": float(rel.mean()), "median_rel_regret": float(np.median(rel)),
                         "p90_rel_regret": float(np.percentile(rel, 90)), "frac_exact_ref_optimum": float(np.mean([s == star for s in lays])),
                         "mean_jaccard_between_replicates": float(np.mean(jac)), "mean_site_displacement_cells": float(np.mean(disp)),
                         "n_distinct_layouts": len(set(lays)), "n_within_5pct_of_star": rec["n_within_5pct_of_star"], "star_on_B_vs_best_B": rec["star_on_B_vs_best_B"],
                         "jaccard_starA_starB": rec["jaccard_starA_starB"]})
    return rows


def test(doc: dict, workers: int) -> list[dict]:
    """Real nonnegative ridge estimator (M1z, alpha = 1/(2*prior_var) = 0.5) on 96 fresh episodes for each distinct chosen layout."""
    eps = [{**e, "seed": TEST_SEED0 + e["id"], "stage": "sufficiency"} for e in ds.episodes("test")]
    lays: dict[str, tuple] = {}
    for key, rec in doc["results"].items():
        noise, _ = key.split("|")
        lays[f"{noise}|ref"] = tuple(rec["star"])
        for n, sel in rec["selected"].items():
            for r, s in enumerate(sel):
                lays[f"{noise}|N{n}|r{r}"] = tuple(s["layout"])
    names = {}
    for key, idx in lays.items():
        names.setdefault(tuple(sorted(idx)), f"L{len(names)}")
    layouts = {nm: tuple(CANDS[i] for i in idx) for idx, nm in names.items()}
    hp = {f"{nm}|{fr}": {"M1z": [(0.5, 0.0)]} for nm in layouts for fr in ds.NOISE}
    rows = ds._pool_rows(eps, hp, ("M1z",), workers, layouts)
    inv = {nm: idx for idx, nm in names.items()}
    out = []
    for r in rows:
        if r["model"] == "M1z":
            out.append({"layout": r["layout"], "sites": inv[r["layout"]], "noise": r["noise"], "episode": r["episode"], "jpop": r["jpop"]})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("design", "test"), required=True)
    ap.add_argument("--out", default=os.environ.get("RESEARCH_RUN_DIR", "results/dyn05"))
    ap.add_argument("--design", default=None, help="frozen design.json for the test stage (default OUT/design.json)")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1)))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.stage == "design":
        doc = design()
        write_json(out / "design.json", doc)
        write_json(out / "summary.json", {"hash": doc["hash"], "rows": summarise(doc)})
        for r in summarise(doc):
            print(r)
        return
    doc = json.loads(Path(a.design or out / "design.json").read_text())
    if doc["hash"] != hashlib.sha256(json.dumps(doc["results"], sort_keys=True).encode()).hexdigest()[:16]:
        raise RuntimeError("freeze guard: design.json does not match its hash")
    ds._prebuild(a.workers)
    rows = test(doc, a.workers)
    write_json(out / "test_rows.json", rows)
    print(len(rows), "rows")


if __name__ == "__main__":
    main()
