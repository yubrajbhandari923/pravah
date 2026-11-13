"""Gate-06: greedy D/A/QR vs exhaustive search when the spacing rule BINDS (synthetic 5x5 lattice, 625 m, min_sep 1250 m,
plume-like G3 so neighbours have correlated footprints). Output: audit_binding.out beside this file."""
import itertools
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0, str(Path.cwd() / 'src'))
from pravah import placement as P

here = Path(__file__).resolve().parent
m, out = 5, []
ii, jj = np.divmod(np.arange(m * m), m); xy = np.stack([ii, jj], 1) * 625.0
res = {n: [] for n in ("dopt", "aopt", "podqr_num")}
rel = {}
for seed in range(40):
    rng = np.random.default_rng(seed)
    K, H = 3, 6
    src = rng.uniform(0, 2500, (K, 2)); ang = rng.uniform(0, 2 * np.pi, H)
    G3 = np.zeros((m * m, H, K))
    for h in range(H):
        u = np.array([np.cos(ang[h]), np.sin(ang[h])])
        for k in range(K):
            d = xy - src[k]; al = d @ u; cr = d @ np.array([-u[1], u[0]])
            G3[:, h, k] = np.where(al > 0, np.exp(-cr**2 / (2 * (200 + .3 * al)**2)) / (1 + al / 500), 0)
    p = P.Problem(xy=xy, cand=np.arange(m * m), min_sep=1250.0, a_ref=1.0, sigma=0.05, G3=G3,
                  snapshots=G3 @ rng.uniform(.5, 5, K))  # noise-free 'archive' over H hours
    for k in (3, 4):
        for name, f in (("dopt", P.dopt), ("aopt", P.aopt)):
            g = f(p, P.greedy(p, k, f)); b = P.brute_force(p, k, f)[0]
            res[name].append(b - g); rel.setdefault(name, []).append((b - g) / abs(b))
        if k <= p.r_num:
            g = P.podqr_value(p, P.design("podqr_num", p, k)); b = P.brute_force(p, k, P.podqr_value)[0]
            res["podqr_num"].append(b - g)
_, _, nf = P.brute_force(p, 4, P.dopt)
out.append(f"feasible 4-subsets {nf} of {len(list(itertools.combinations(range(m * m), 4)))} (spacing binds)")
for n, v in res.items():
    v = np.array(v)
    out.append(f"{n}: cases {len(v)}, exact {int((v <= 1e-9).sum())}, median gap {np.median(v):.3g}, max gap {v.max():.3g}" + (f", max rel gap {max(rel[n]):.3g}" if n in rel else ""))
txt = "\n".join(out); print(txt); (here / "audit_binding.out").write_text(txt + "\n")
