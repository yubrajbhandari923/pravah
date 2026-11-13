"""Gate-06 spot checks on 1-2 E06 scenarios (single core, ~2-3 min each).
Env AUDIT_CASES="veering:0:2,steady_westerly:0:4". Output: audit_spot.out beside this file.
Checks: (1) reproduction of committed CSV sites/values; (2) r_eff mode amplitudes, prefix vs count, time-integrated variant;
(3) greedy tie margins and cand-order sensitivity; (4) practical rank on held-out validation hours (G over hours 96-119);
(5) the noise-free POD archive equals G a (true strengths) and podqr changes when strengths change."""
import os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

root = Path.cwd(); here = Path(__file__).resolve().parent
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
import e02_examples as E
from pravah import benchmarks as B, placement as P
from pravah.identifiability import practical_rank
from pravah.sensitivity import scenario_response_matrix

csv = pd.read_csv(root / "reports/quarto/tables/e06-criteria.csv")
out = []
def pr(*a):
    s = " ".join(map(str, a)); print(s, flush=True); out.append(s)

for case in os.environ.get("AUDIT_CASES", "veering:0:2,steady_westerly:0:4").split(","):
    wind, seed, K = case.split(":"); seed, K = int(seed), int(K)
    t0 = time.time()
    sc = B.generate_scenario(E.make_cfg([wind]), seed, K)
    G = scenario_response_matrix(sc, 120).reshape(sc.n_candidate_sites, 120, K)
    G3, Gv = G[:, :96].copy(), G[:, 96:120]
    pr(f"\n=== {wind} seed={seed} K={K} (sim {time.time() - t0:.0f}s); validation hours {sc.split_hours('validation')[[0, -1]]}")
    ii, jj = np.divmod(np.arange(sc.n_candidate_sites), sc.lattice_n)
    adm = sc.admissible_sites(); small = adm[(ii[adm] % 3 == 1) & (jj[adm] % 3 == 1)]
    rseed = sc.cfg.seeds.streams(seed)["placement"]; sig = float(sc.noise_sigma[0])
    # (5) archive = background + G a ?
    a = sc.truth.strengths_ug_s
    X = P.training_snapshots(sc)
    bg = sc.cfg.sources.background_ug_m3
    pr("archive vs bg + G a: max abs diff", f"{np.abs(X - bg - G3 @ a).max():.3g}", "(field max anomaly", f"{(X - bg).max():.3g})", "| true a / a_ref:", np.round(a / 1e6, 3))
    for cs, cand in (("full", adm), ("small", small)):
        p = P.problem_from_scenario(sc, G3, cand)
        rows = csv[(csv.wind == wind) & (csv.seed == seed) & (csv.K == K) & (csv.cand_set == cs)]
        # (1) reproduction
        mism = 0; nok = 0
        for _, r in rows.iterrows():
            try:
                S = P.design(r.design, p, int(r.k), seed=rseed)
            except ValueError as e:
                mism += int(r.status == "ok"); continue
            nok += 1
            mism += int(r.status != "ok" or r.sites != " ".join(map(str, S)) or abs(P.dopt(p, S) - r.dopt) > 1e-9)
        pr(f"[{cs}] n_cand={len(cand)} r_eff={p.r_eff} r_num={p.r_num}: reproduced {len(rows) - mism}/{len(rows)} rows ({nok} defined)")
        if cs == "full":
            # (2) mode amplitudes
            Xc = X - X.mean(1, keepdims=True)
            U, s, _ = np.linalg.svd(Xc, full_matrices=False)
            amp = (s * np.abs(U[cand])).max(0) / sig / np.sqrt(X.shape[1])
            pr("per-reading mode amplitude (noise sd), modes 1-6:", np.round(amp[:6], 3))
            above = amp >= 1
            prefix = int(np.argmin(above)) if not above.all() else len(above)
            pr("r_eff (count) =", int(above.sum()), "| leading-prefix count =", prefix,
               "| time-integrated variant (amp*sqrt(T) >= 1) =", int((amp * np.sqrt(X.shape[1]) >= 1).sum()))
            # (3) tie margins and cand-order sensitivity, k = 8
            for name, f in (("dopt", P.dopt), ("aopt", P.aopt)):
                S, margins = [], []
                for _ in range(8):
                    v = sorted(((f(p, S + [int(i)]), int(i)) for i in p.cand if p.feasible(int(i), S)), reverse=True)
                    margins.append(v[0][0] - v[1][0]); S.append(P.greedy(p, len(S) + 1, f)[-1])
                prev = P.Problem(xy=p.xy, cand=p.cand[::-1].copy(), min_sep=p.min_sep, a_ref=p.a_ref, sigma=p.sigma, G3=p.G3, snapshots=p.snapshots)
                Sr = P.greedy(prev, 8, f)
                pr(f"{name} k=8: min step margin {min(margins):.3g} (rel {min(m / abs(f(p, S)) for m in margins):.2g});",
                   "reversed cand order gives same set:", sorted(Sr) == sorted(S))
            # (4) held-out practical rank
            res = []
            for name in P.STRATEGIES:
                for k in (2, 4, 8):
                    try:
                        S = P.design(name, p, k, seed=rseed)
                    except ValueError:
                        continue
                    res.append(dict(design=name, k=k, prank_train=practical_rank(G3[S].reshape(-1, K), sig, p.a_ref),
                                    prank_val=practical_rank(Gv[S].reshape(-1, K), sig, p.a_ref),
                                    prank_val_ar1_eff=practical_rank(Gv[S].reshape(-1, K) / 3, sig, p.a_ref)))
            pr("practical rank: train 96 h (design data) vs held-out validation 24 h (and /3 for AR(1) rho=.8 info loss):")
            pr(pd.DataFrame(res).pivot(index="design", columns="k", values=["prank_train", "prank_val", "prank_val_ar1_eff"]).to_string())
            # (5) podqr depends on training-period strengths
            if p.r_num >= 2:
                eq = np.full(K, np.mean(a))
                p2 = P.Problem(xy=p.xy, cand=p.cand, min_sep=p.min_sep, a_ref=p.a_ref, sigma=p.sigma, G3=G3, snapshots=bg + G3 @ eq)
                for k in (2,):
                    pr(f"podqr_num k={k}: true strengths {P.design('podqr_num', p, k)} vs equal strengths {P.design('podqr_num', p2, k)};",
                       "dopt (strength-independent):", P.design('dopt', p, k), P.design('dopt', p2, k))
(here / "audit_spot.out").write_text("\n".join(out) + "\n")
