"""EXP-015 stage 2: stability of greedy_total sensor-design decisions across mesh and dt (real geography; synthetic emissions
and meteorology).

    python scripts/convergence_stage2.py --config configs/convergence_stage2.json [--out DIR] [--work DIR] [--estimate]

Order (each step resumes from files in --work; one mesh config at a time, response bases cached on disk by ResponseCache key =
hash of grid, dt, scheme, wind, physics, sources):
  1. candidate table built ONCE (stage2.candidate_mesh) in UTM metres; each mesh looks the sites up by cell (never reuses indices);
  2. greedy_total layouts per mesh config (design seed 101), reference config first;
  3. R replicate design ensembles on the candidate mesh (resampling baseline), layouts + their objective for the reference layout;
  4. only now (layouts frozen) the TEST ensemble on the reference mesh is built; metrics, tables, figures.
alpha and the noise scale are frozen inputs (alpha_selected.json / noise_levels.csv); nothing is selected here."""
from __future__ import annotations

import argparse
import copy
import gc
import itertools
import json
import resource
import sys
import time
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from pravah import design as D  # noqa: E402
from pravah import geodesign as G  # noqa: E402
from pravah.config import load_config  # noqa: E402
from pravah.inverse import ResponseCache  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
STAMP = "EXP-015 stage2"
RSS_LIMIT_GB = 2.0
T0 = time.perf_counter()


def say(msg: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1e9 if sys.platform == "darwin" else 1e6)
    print(f"[{time.perf_counter() - T0:7.0f}s rss {rss:.2f}GB] {msg}", flush=True)
    assert rss < RSS_LIMIT_GB, f"peak RSS {rss:.2f} GB exceeds {RSS_LIMIT_GB} GB"


def mname(res: float, dt: float) -> str:
    return f"r{res:g}_dt{dt:g}"


# ------------------------------------------------------------------------------------------------ pure metrics
def compare_layouts(ids_a: list[str], ids_b: list[str], xy: dict[str, tuple[float, float]], taus: tuple[float, ...]) -> dict[str, float]:
    """Jaccard of the site sets; optimal one-to-one assignment (equal budgets) of a's sites to b's by UTM distance [m]:
    mean/max displacement and the fraction of assigned pairs within each tau."""
    a, b = np.array([xy[i] for i in ids_a]), np.array([xy[i] for i in ids_b])
    assert len(a) == len(b), "layouts must have equal budgets"
    d = cdist(a, b)
    r, c = linear_sum_assignment(d)
    dm = d[r, c]
    out = {"jaccard": len(set(ids_a) & set(ids_b)) / len(set(ids_a) | set(ids_b)), "mean_disp_m": float(dm.mean()),
           "max_disp_m": float(dm.max())}
    out.update({f"frac_within_{t:g}m": float((dm <= t + 1e-9).mean()) for t in taus})
    return out


def cell_lookup(grid: Any, ux: np.ndarray, uy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cell (ix, iy) of UTM points on a mesh; AssertionError when a point lies outside the domain."""
    nx, ny = round(grid.lx / grid.dx), round(grid.ly / grid.dy)
    ix, iy = np.floor((ux - grid.x0) / grid.dx).astype(int), np.floor((uy - grid.y0) / grid.dy).astype(int)
    assert ((ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)).all(), "candidate site outside the mesh domain"
    return ix, iy


def mapping_table(geo: Any, table: Any, resolutions: list[float]) -> Any:
    """Per site and mesh: cell index, cell-centre offset [m], study_fraction and its category (inside > 0.5, partial, outside);
    'category_changed_vs_coarsest' flags sites whose study category differs from the coarsest mesh's."""
    ux, uy = table["utm_x"].to_numpy(float), table["utm_y"].to_numpy(float)
    out = pd.DataFrame({"id": table["id"], "utm_x": ux, "utm_y": uy})
    coarse = max(resolutions)
    cats: dict[float, np.ndarray] = {}
    for r in sorted(resolutions, reverse=True):
        lv = geo.region.level(r)
        ix, iy = cell_lookup(lv.grid, ux, uy)
        sf = np.asarray(lv.arrays["study_fraction"], float)[iy, ix]
        cx, cy = lv.grid.x0 + (ix + 0.5) * lv.grid.dx, lv.grid.y0 + (iy + 0.5) * lv.grid.dy
        cats[r] = np.where(sf > 0.5, "inside", np.where(sf > 0, "partial", "outside"))
        out[f"ix_{r:g}"], out[f"iy_{r:g}"] = ix, iy
        out[f"centre_offset_m_{r:g}"] = np.hypot(cx - ux, cy - uy)
        out[f"study_fraction_{r:g}"], out[f"study_category_{r:g}"] = sf, cats[r]
    out["category_changed_vs_coarsest"] = np.any([cats[r] != cats[coarse] for r in resolutions], axis=0)
    return out


# ------------------------------------------------------------------------------------------------ setup / ensembles
def make_setup(cfg: dict[str, Any], res: float, dt: float, table: Any, cache: ResponseCache) -> G.Setup:
    """geodesign.Setup on mesh (res, dt) with the ONE candidate table (patched in: Setup would rebuild it per mesh)."""
    c = copy.deepcopy(cfg)
    c["mesh"] = {"resolution_m": float(res), "dt_s": float(dt)}
    orig = G.build_candidate_table
    G.build_candidate_table = lambda geo, _cfg: table.copy()  # shortcut: module-level patch, restored below; no src edit
    try:
        return G.Setup(c, cache)
    finally:
        G.build_candidate_table = orig


def draws_of(cfg: dict[str, Any]) -> dict[str, tuple[list[G.CaseDraw], int, int]]:
    """All ensembles of the study (design, validation, test, replicates) for the disjointness guard."""
    s2 = cfg["stage2"]["replicates"]
    d = {r: (G.draw_cases(cfg, int(e["n_cases"]), int(e["seed"]), r[0]), int(e["seed"]), int(e["noise_seed"]))
         for r, e in cfg["ensembles"].items()}
    for r in range(1, int(s2["n"]) + 1):
        seed, ns = int(s2["seed0"]) + r - 1, int(s2["noise_seed0"]) + (r - 1) * int(s2["noise_seed_stride"])
        d[f"rep{r}"] = (G.draw_cases(cfg, int(s2["n_cases"]), seed, f"r{r}_"), seed, ns)
    G.check_ensembles(d)
    return d


def frozen_noise_scale(cfg: dict[str, Any]) -> float | None:
    f = cfg["stage2"]["noise_scale_file"]
    if f is None:
        return None
    t = pd.read_csv(ROOT / f)
    return float(t.loc[t["ensemble"] == "validation", "signal_scale_ug_m3"].iloc[0])


def check_alpha(cfg: dict[str, Any]) -> float:
    s2 = cfg["stage2"]
    if s2["alpha_file"] is not None:
        sel = json.loads((ROOT / s2["alpha_file"]).read_text())["alpha_by_noise_fraction"]
        for f in cfg["noise"]["fractions"]:
            assert float(sel[f"{f:g}"]) == float(s2["alpha"]), f"alpha {s2['alpha']} != frozen {sel}"
    return float(s2["alpha"])


def build_cases(st: G.Setup, draws: list[G.CaseDraw], noise_first: int, role: str, tag: str) -> list[D.DesignCase]:
    cases = []
    for i, d in enumerate(draws):
        t = time.perf_counter()
        cases.append(st.build(d, role, noise_first + i)[0])
        say(f"{tag} case {d.name} built in {time.perf_counter() - t:.0f} s")
    st.cache._mem.clear()  # keep RSS low: only the small DesignCase objects are needed afterwards
    st.cache.extra.clear()
    gc.collect()
    return cases


def greedy_layouts(st: G.Setup, cases: list[D.DesignCase], sigma: dict[float, float], alpha: float,
                   budgets: list[int]) -> dict[str, dict[str, list[str]]]:
    cons = st.constraints(G.SETTING_A, G.STANDARD)  # interior candidates only
    out: dict[str, dict[str, list[str]]] = {}
    for f in st.fractions:
        tr = D.greedy_trace(st.cands, max(budgets), cases, cons, st.opts(sigma[f], alpha), "total")  # budget b = prefix b
        out[f"{f:g}"] = {str(b): [st.cands.ids[i] for i in tr.order[:b]] for b in budgets}
    return out


def objective(st: G.Setup, cases: list[D.DesignCase], ids: list[str], sigma: float, alpha: float) -> float:
    idx = tuple(sorted(st.cands.ids.index(i) for i in ids))
    return float(D.score_layout(idx, cases, st.opts(sigma, alpha), "total")["value"])


# ------------------------------------------------------------------------------------------------ stages
class Run:
    def __init__(self, cfg: dict[str, Any], out: Path, work: Path) -> None:
        self.cfg, self.out, self.work = cfg, out, work
        self.s2 = cfg["stage2"]
        self.budgets = [int(b) for b in self.s2["budgets"]]
        self.fractions = [float(f) for f in cfg["noise"]["fractions"]]
        self.taus = tuple(float(t) for t in self.s2["tau_m"])
        self.ref = tuple(float(v) for v in self.s2["reference"])
        self.meshes = [tuple(float(v) for v in m) for m in self.s2["meshes"]]
        assert self.ref in self.meshes
        out.mkdir(parents=True, exist_ok=True)
        work.mkdir(parents=True, exist_ok=True)
        self.cache = ResponseCache(work / "cache")
        self.alpha = check_alpha(cfg)
        self.draws = draws_of(cfg)
        say("candidate table (once, UTM)")
        cm = [float(v) for v in self.s2["candidate_mesh"]]
        c0 = copy.deepcopy(cfg)
        c0["mesh"] = {"resolution_m": cm[0], "dt_s": cm[1]}
        geo = G.load_geography(c0)
        self.table: Any = G.build_candidate_table(geo, c0)
        self.xy = {i: (x, y) for i, x, y in zip(self.table['id'], self.table['utm_x'], self.table['utm_y'])}
        self.mapping = mapping_table(geo, self.table, sorted({m[0] for m in self.meshes}))
        self.mapping.to_csv(out / "candidate_mapping.csv", index=False)
        say(f"{len(self.table)} sites mapped on meshes {sorted({m[0] for m in self.meshes})}; "
            f"{int(self.mapping['category_changed_vs_coarsest'].sum())} change study category vs the coarsest mesh")
        self.cand_mesh = (cm[0], cm[1])
        self._scale = frozen_noise_scale(cfg)
        self._ref: tuple[G.Setup, list[D.DesignCase], dict[float, float]] | None = None

    def setup(self, mesh: tuple[float, ...]) -> G.Setup:
        return make_setup(self.cfg, mesh[0], mesh[1], self.table, self.cache)

    def scale(self, st: G.Setup) -> float:
        if self._scale is None:  # smoke: validation ensemble on the reference mesh defines the scale
            say("noise scale from the validation ensemble on the reference mesh")
            v = st.ensemble("validation", say)
            self._scale = v.scale
            st.cache._mem.clear()
            st.cache.extra.clear()
        return self._scale

    def sigma(self, scale: float) -> dict[float, float]:
        return {f: f * scale for f in self.fractions}

    def ref_design(self) -> tuple[G.Setup, list[D.DesignCase], dict[float, float]]:
        if self._ref is None:
            st = self.setup(self.ref)
            sg = self.sigma(self.scale(st))
            e = self.cfg["ensembles"]["design"]
            cases = build_cases(st, self.draws["design"][0], int(e["noise_seed"]), "design", f"ref {mname(*self.ref)} design")
            self._ref = (st, cases, sg)
        return self._ref

    def config_layouts(self) -> dict[Any, Any]:
        out = {}
        for mesh in [self.ref] + [m for m in self.meshes if m != self.ref]:  # reference first
            f = self.work / f"layouts_{mname(*mesh)}.json"
            if f.exists():
                say(f"{mname(*mesh)}: layouts found, skipped")
            else:
                if mesh == self.ref:
                    st, cases, sg = self.ref_design()
                else:
                    st = self.setup(mesh)
                    sg = self.sigma(self.scale(self.ref_design()[0]))
                    e = self.cfg["ensembles"]["design"]
                    cases = build_cases(st, self.draws["design"][0], int(e["noise_seed"]), "design", f"{mname(*mesh)} design")
                lay = greedy_layouts(st, cases, sg, self.alpha, self.budgets)
                f.write_text(json.dumps(lay))
                say(f"{mname(*mesh)}: layouts selected {lay}")
                del cases
                gc.collect()
            out[mesh] = json.loads(f.read_text())
        return out

    def replicates(self, ref_lay: dict[str, dict[str, list[str]]]) -> list[dict[str, Any]]:
        """Per replicate on the candidate mesh: greedy layouts and the objective of the REFERENCE layout on that replicate."""
        rp = self.s2["replicates"]
        st = None
        out = []
        for r in range(1, int(rp["n"]) + 1):
            f = self.work / f"rep{r}.json"
            if not f.exists():
                st = st or self.setup(self.cand_mesh)
                sg = self.sigma(self.scale(self.ref_design()[0]))
                draws, _, ns = self.draws[f"rep{r}"]
                cases = build_cases(st, draws, ns, "design", f"replicate {r}")
                lay = greedy_layouts(st, cases, sg, self.alpha, self.budgets)
                j_ref = {fk: {bk: objective(st, cases, ids, sg[float(fk)], self.alpha) for bk, ids in v.items()}
                         for fk, v in ref_lay.items()}
                f.write_text(json.dumps({"layouts": lay, "j_ref_layout": j_ref}))
                say(f"replicate {r}/{rp['n']} done")
            out.append(json.loads(f.read_text()))
        return out

    def evaluate(self, lays: dict[Any, Any], reps: list[dict[str, Any]]) -> None:
        """Layouts are frozen: build the TEST ensemble on the reference mesh, then metrics, tables, figures."""
        assert all((self.work / f"layouts_{mname(*m)}.json").exists() for m in self.meshes)
        st, dcases, sg = self.ref_design()
        scale = self.scale(st)
        te = self.cfg["ensembles"]["test"]
        tcases = build_cases(st, self.draws["test"][0], int(te["noise_seed"]), "test", "ref test")
        ids = st.cands.ids
        allowed = np.flatnonzero(st.constraints(G.SETTING_A, G.STANDARD).allowed)
        Jc: dict[tuple[str, tuple[str, ...]], float] = {}

        def J(f: float, layout: list[str]) -> float:
            k = (f"{f:g}", tuple(sorted(layout)))
            if k not in Jc:
                Jc[k] = objective(st, dcases, layout, sg[f], self.alpha)
            return Jc[k]

        def test_rmse(f: float, layout: list[str]) -> list[float]:
            idx = tuple(sorted(ids.index(i) for i in layout))
            o = st.opts(sg[f], self.alpha)
            return [D.fit_layout(c, idx, rep, o)["rmse"] for c in tcases for rep in range(c.n_noise)]

        tests, stab, eps_rows = [], [], []
        for f in self.fractions:
            fk = f"{f:g}"
            for b in self.budgets:
                bk = str(b)
                ref_ids = lays[self.ref][fk][bk]
                jref = J(f, ref_ids)
                sd = float(np.std([r["j_ref_layout"][fk][bk] for r in reps], ddof=1)) if len(reps) > 1 else float("nan")
                # single-site swaps of the reference layout (interior candidates), objective on the reference design cases
                refset = set(ref_ids)
                delta = np.array([J(f, [s if s != o else ids[c] for s in ref_ids]) - jref
                                  for o in ref_ids for c in allowed if ids[c] not in refset])
                eps_rows += [{"noise_fraction": f, "budget": b, "delta_j": float(v)} for v in np.sort(delta)]
                eps = int((delta <= sd).sum()) if np.isfinite(sd) else -1
                rt_ref = test_rmse(f, ref_ids)
                for mesh in self.meshes:
                    lay = lays[mesh][fk][bk]
                    rt = rt_ref if mesh == self.ref else test_rmse(f, lay)
                    tests += [{"config": mname(*mesh), "noise_fraction": f, "budget": b, "case_rep": i, "rmse": x} for i, x in enumerate(rt)]
                    stab.append({"res_m": mesh[0], "dt_s": mesh[1], "budget": b, "noise_fraction": f,
                                 **compare_layouts(lay, ref_ids, self.xy, self.taus),
                                 "j_ref_layout_c": J(f, lay), "j_ref_layout_ref": jref, "regret": J(f, lay) - jref,
                                 "test_rmse_c": float(np.mean(rt)), "test_rmse_ref": float(np.mean(rt_ref)),
                                 "test_rmse_delta": float(np.mean(rt) - np.mean(rt_ref)), "eps_sd": sd, "eps_set_size": eps,
                                 "n_swaps": len(delta), "tie_fraction": eps / len(delta) if eps >= 0 else float("nan"),
                                 "sites": ",".join(lay)})
        base = []
        main = lays[self.cand_mesh]
        for f in self.fractions:
            fk = f"{f:g}"
            for b in self.budgets:
                bk = str(b)
                rl = [r["layouts"][fk][bk] for r in reps]
                for (i, a), (j, c) in itertools.combinations(enumerate(rl, 1), 2):
                    base.append({"kind": "pair", "a": f"rep{i}", "b": f"rep{j}", "budget": b, "noise_fraction": f,
                                 **compare_layouts(a, c, self.xy, self.taus), "regret": J(f, a) - J(f, c)})
                for i, a in enumerate(rl, 1):
                    base.append({"kind": "vs_main", "a": f"rep{i}", "b": mname(*self.cand_mesh), "budget": b, "noise_fraction": f,
                                 **compare_layouts(a, main[fk][bk], self.xy, self.taus), "regret": J(f, a) - J(f, main[fk][bk])})
        o = self.out
        pd.DataFrame(stab).to_csv(o / "stability.csv", index=False)
        pd.DataFrame(base).to_csv(o / "replicate_baseline.csv", index=False)
        pd.DataFrame(tests).to_csv(o / "test_scores.csv", index=False)
        pd.DataFrame(eps_rows).to_csv(o / "epsilon_swaps.csv", index=False)
        self.write_layouts(lays, reps)
        self.figures(st, pd.DataFrame(stab), pd.DataFrame(base), pd.DataFrame(eps_rows), lays)
        self.readme(pd.DataFrame(stab), pd.DataFrame(base), scale)

    def write_layouts(self, lays: dict[Any, Any], reps: list[dict[str, Any]]) -> None:
        t = self.table.set_index("id")
        named = {mname(*m): v for m, v in lays.items()} | {f"rep{i}": r["layouts"] for i, r in enumerate(reps, 1)}
        rows, feats = [], []
        for name, lay in named.items():
            for fk, per_b in lay.items():
                for bk, ids in per_b.items():
                    rows.append({"config": name, "noise_fraction": float(fk), "budget": int(bk), "site_ids": ";".join(ids),
                                 "utm_x_m": ";".join(f"{t.at[i, 'utm_x']:.1f}" for i in ids),
                                 "utm_y_m": ";".join(f"{t.at[i, 'utm_y']:.1f}" for i in ids), "epsg": self.mapping_epsg()})
                    feats += [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [float(t.at[i, "lon"]), float(t.at[i, "lat"])]},
                                "properties": {"config": name, "noise_fraction": float(fk), "budget": int(bk), "order": k + 1, "site_id": i}}
                              for k, i in enumerate(ids)]
        pd.DataFrame(rows).to_csv(self.out / "layouts.csv", index=False)
        (self.out / "layouts.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}))

    def mapping_epsg(self) -> int:
        return G.load_geography(self.cfg).epsg

    # -------------------------------------------------------------------------------------------- figures / readme
    def figures(self, st: G.Setup, stab: Any, base: Any, eps: Any, lays: dict[Any, Any]) -> None:
        fd = self.out / "figures"
        fd.mkdir(exist_ok=True)
        f0 = self.fractions[0]
        mb = int(self.s2["map_budget"])
        # 1. layout maps per mesh
        fig, axes = plt.subplots(1, len(self.meshes), figsize=(4.2 * len(self.meshes), 4.6), squeeze=False, layout="constrained")
        t = self.table.set_index("id")
        inter = self.table[self.table["category"] == G.INTERIOR]
        ref_ids = lays[self.ref][f"{f0:g}"][str(mb)]
        for ax, mesh in zip(axes[0], self.meshes):
            xs, ys = st.geo.study.exterior.xy
            ax.plot(np.array(xs) / 1000, np.array(ys) / 1000, "k-", lw=0.8, label="study polygon")
            ax.scatter(inter["utm_x"] / 1000, inter["utm_y"] / 1000, s=8, c="0.8", label="interior candidates")
            ids = lays[mesh][f"{f0:g}"][str(mb)]
            ax.scatter(t.loc[ref_ids, "utm_x"] / 1000, t.loc[ref_ids, "utm_y"] / 1000, s=120, facecolors="none", edgecolors="C3", lw=1.5,
                       label=f"reference {mname(*self.ref)}")
            ax.scatter(t.loc[ids, "utm_x"] / 1000, t.loc[ids, "utm_y"] / 1000, s=35, c="C0", label="this mesh")
            ax.set(title=f"{mname(*mesh)}, budget {mb}", xlabel="UTM easting [km]", ylabel="UTM northing [km]", aspect="equal")
        axes[0, 0].legend(fontsize=6, loc="best")
        fig.text(0.01, 0.005, f"{STAMP}: greedy_total, noise {f0:g}, alpha {self.alpha:g}. {G.LABEL}", fontsize=7, style="italic")
        fig.savefig(fd / "layout_maps.png", dpi=150)
        plt.close(fig)
        # 2. displacement / regret vs the resampling distribution
        for f in self.fractions:
            fig, axes = plt.subplots(2, len(self.budgets), figsize=(3.6 * len(self.budgets), 6.5), squeeze=False, layout="constrained")
            for j, b in enumerate(self.budgets):
                bs, ss = base[(base.budget == b) & (base.noise_fraction == f)], stab[(stab.budget == b) & (stab.noise_fraction == f)]
                for i, (col, lab) in enumerate([("mean_disp_m", "mean site displacement [m]"), ("regret", "regret J_ref(c) - J_ref(ref) [ug$^2$/m$^6$]")]):
                    ax = axes[i, j]
                    groups = [("replicate pairs", bs[bs.kind == "pair"][col].to_numpy()), ("replicate vs main", bs[bs.kind == "vs_main"][col].to_numpy())]
                    if col == "regret":
                        groups[0] = ("replicate pairs |.|", np.abs(groups[0][1]))
                    ax.boxplot([g for _, g in groups], tick_labels=[n for n, _ in groups], showfliers=False)
                    for k, (_, g) in enumerate(groups, 1):
                        ax.scatter(np.full(len(g), k) + np.linspace(-0.15, 0.15, max(len(g), 1))[: len(g)], g, s=6, c="0.5", zorder=3)
                    for m, r in enumerate(ss.itertuples()):
                        if (r.res_m, r.dt_s) != self.ref:
                            ax.scatter(2.6 + 0.12 * m, getattr(r, col), marker="D", s=40, label=mname(r.res_m, r.dt_s), zorder=4)
                    ax.set(title=f"budget {b}", ylabel=lab)
                    ax.tick_params(axis="x", labelrotation=20, labelsize=7)
            axes[0, 0].legend(fontsize=6, title="mesh change vs reference", title_fontsize=6)
            fig.text(0.01, 0.005, f"{STAMP}: noise {f:g}; boxes = {int(base['a'].nunique())} replicate design ensembles (500 m, dt 100)", fontsize=7, style="italic")
            fig.savefig(fd / f"displacement_regret_vs_baseline_f{f:g}.png", dpi=150)
            plt.close(fig)
        # 3. objective ties
        fig, axes = plt.subplots(1, len(self.budgets), figsize=(3.6 * len(self.budgets), 3.8), squeeze=False, layout="constrained")
        for ax, b in zip(axes[0], self.budgets):
            e, s = eps[(eps.budget == b) & (eps.noise_fraction == f0)], stab[(stab.budget == b) & (stab.noise_fraction == f0)]
            ax.plot(np.arange(1, len(e) + 1), e["delta_j"], ".", ms=3, label="single-site swaps of reference layout")
            sd = float(s["eps_sd"].iloc[0])
            ax.axhline(sd, c="C3", ls="--", label="one replicate SD")
            for r in s.itertuples():
                if (r.res_m, r.dt_s) != self.ref:
                    ax.axhline(r.regret, c="0.5", lw=0.6, ls=":")
            ax.set(title=f"budget {b}: {int(s['eps_set_size'].iloc[0])}/{len(e)} swaps within SD", xlabel="swap rank",
                   ylabel="J(swap) - J(reference layout) [ug$^2$/m$^6$]", yscale="symlog")
        axes[0, 0].legend(fontsize=6)
        fig.text(0.01, 0.005, f"{STAMP}: noise {f0:g}; dotted lines = regret of the other mesh layouts", fontsize=7, style="italic")
        fig.savefig(fd / "objective_ties.png", dpi=150)
        plt.close(fig)

    def readme(self, stab: Any, base: Any, scale: float) -> None:
        def q(x: pd.Series) -> str:
            return f"{x.median():.3g} [{x.quantile(.25):.3g}, {x.quantile(.75):.3g}]" if len(x) else "n/a"
        lines = [f"# EXP-015 stage 2: decision stability across mesh and dt ({self.cfg['name']})", "",
                 "Real geography; synthetic emissions and meteorology. Numbers only, no interpretation. Reference = "
                 f"{mname(*self.ref)}. alpha frozen = {self.alpha:g}; noise sigma = fraction x {scale:.5g} ug/m3 (frozen validation signal scale), "
                 "identical on every mesh. Replicate baseline = replicate design ensembles on the "
                 f"{mname(*self.cand_mesh)} mesh; cells show median [IQR].", "",
                 "Success-criterion flags: `in_iqr` = mesh-induced value within the replicate-pair IQR; `monotone` = displacement of "
                 "the coarse mesh >= that of the refined mesh at fixed dt 100 (500 -> 250 m).", ""]
        for f in self.fractions:
            for b in self.budgets:
                bs = base[(base.budget == b) & (base.noise_fraction == f)]
                pr, vm = bs[bs.kind == "pair"], bs[bs.kind == "vs_main"]
                lines += [f"## noise {f:g}, budget {b}", "",
                          f"Replicate pairs (n={len(pr)}): mean disp [m] {q(pr.mean_disp_m)}; Jaccard {q(pr.jaccard)}; |regret| {q(pr.regret.abs())}. "
                          f"Replicate vs main (n={len(vm)}): mean disp [m] {q(vm.mean_disp_m)}; Jaccard {q(vm.jaccard)}; regret {q(vm.regret)}.", "",
                          "| config | Jaccard | mean disp [m] | max disp [m] | within 500 m | within 1000 m | regret | test RMSE c / ref | eps set | in_iqr (disp, |regret|) |",
                          "|---|---|---|---|---|---|---|---|---|---|"]
                s = stab[(stab.budget == b) & (stab.noise_fraction == f)]
                for r in s.itertuples():
                    lo, hi = pr.mean_disp_m.quantile(.25), pr.mean_disp_m.quantile(.75)
                    rlo, rhi = pr.regret.abs().quantile(.25), pr.regret.abs().quantile(.75)
                    ok = f"{lo <= r.mean_disp_m <= hi}, {rlo <= abs(r.regret) <= rhi}" if len(pr) else "n/a"
                    lines.append(f"| {mname(r.res_m, r.dt_s)} | {r.jaccard:.2f} | {r.mean_disp_m:.0f} | {r.max_disp_m:.0f} | "
                                 f"{r.frac_within_500m:.2f} | {r.frac_within_1000m:.2f} | {r.regret:.3g} | {r.test_rmse_c:.4g} / {r.test_rmse_ref:.4g} | "
                                 f"{r.eps_set_size}/{r.n_swaps} | {ok} |")
                co, fi = s[(s.res_m == 500) & (s.dt_s == 100)], s[(s.res_m == 250) & (s.dt_s == 100)]
                if len(co) and len(fi):
                    lines += ["", f"monotone (500 -> 250 m at dt 100): {bool(co.mean_disp_m.iloc[0] >= fi.mean_disp_m.iloc[0])}"]
                lines.append("")
        lines += ["Files: stability.csv, replicate_baseline.csv, layouts.csv/geojson, test_scores.csv, epsilon_swaps.csv, "
                  "candidate_mapping.csv, figures/. eps set = number of single-site swaps of the reference layout whose reference-mesh "
                  "design objective is within one SD (over replicates) of the reference layout's objective on the replicate ensembles."]
        (self.out / "README_stage2.md").write_text("\n".join(lines) + "\n")


def estimate(run: Run) -> None:
    """Time one reference-mesh design case (the full case = 5 unit-response solves + direct truth + base parts)."""
    st = run.setup(run.ref)
    run.cache._mem.clear()
    t = time.perf_counter()
    st.build(run.draws["design"][0][0], "design", int(run.cfg["ensembles"]["design"]["noise_seed"]))
    one = time.perf_counter() - t
    n_design = len(run.draws["design"][0])
    say(f"one {mname(*run.ref)} design case: {one:.0f} s; design set of {n_design}: ~{one * n_design / 60:.0f} min")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/convergence_stage2.json")
    ap.add_argument("--out", default="results/phase3-convergence")
    ap.add_argument("--work", default=None, help="resumable work dir (default OUT/stage2_work)")
    ap.add_argument("--estimate", action="store_true", help="time one reference-mesh design case and exit")
    a = ap.parse_args()
    cfg = load_config(ROOT / a.config)
    cfg["region"]["manifest"] = str(ROOT / cfg["region"]["manifest"])
    out = ROOT / a.out
    run = Run(cfg, out, Path(a.work) if a.work else out / "stage2_work")
    if a.estimate:
        estimate(run)
        return
    lays = run.config_layouts()
    reps = run.replicates(lays[run.ref])
    run.evaluate(lays, reps)
    say("done")


if __name__ == "__main__":
    main()
