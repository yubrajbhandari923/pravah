# Rescued 2026-11-12 from /tmp/audit (gate-04 audit, legacy RUN-0018); was never committed under ResearchOS.
# Run: PRAVAH_ROOT=. RESEARCH_RUN_DIR=<out dir> uv run python scripts/e04_audit_mc.py
"""T-026 audit (read-only on repo code): FIM local-Gaussian ellipse coverage vs empirical repeated-noise coverage.
Reuses the committed E04 pipeline (build G for 'changing4', steady_westerly, informative design from e04-meta.json). Seeded, 2000 draws."""
import json, os, sys
from pathlib import Path
import numpy as np
from scipy.stats import chi2, beta
root = Path(os.environ.get("PRAVAH_ROOT", ".")).resolve()
sys.path.insert(0, str(root / "src")); sys.path.insert(0, str(root / "scripts"))
from pravah import benchmarks as B
from pravah.information import fisher, criteria, confidence_ellipse
from pravah.inverse import solve_nnls
from pravah.observations import bilinear_sample
from pravah.sensitivity import _anomaly_fields
import e02_examples as E
HOURS, REP, NDRAW, SEED = 24, 2, 2000, 20261110
meta = json.loads((root / "reports/quarto/tables/e04-meta.json").read_text())
rep, cells, a0, _, _, _ = E.EXAMPLES["changing4"]
with E.fixed_sources(cells, a0):
    sc = B.generate_scenario(E.make_cfg(["steady_westerly"]), REP, len(cells))
m, dom = sc.cfg.model, tuple(sc.cfg.model.domain_m)
F = np.array([_anomaly_fields(sc, B.source_basis(tuple(m.grid), dom, xy, sc.cfg.sources.footprint_sigma_m) / m.mixing_depth_m, HOURS) for xy in sc.truth.source_xy])
G3 = np.array([np.stack([bilinear_sample(c, sc.site_xy, dom) for c in f], 1) for f in F]).transpose(1, 2, 0)
a_true = np.asarray(sc.truth.strengths_ug_s, float); K = a_true.size
adm = np.asarray(sc.admissible_sites()); Gall = G3[adm].reshape(-1, K)
sig0 = meta["noise"]["info_scale"]
rng = np.random.default_rng(SEED)
metrics = []
def log(n, v): metrics.append({"name": n, "value": float(v), "step": 0})
log("seed", SEED); log("ndraw", NDRAW); log("sigma_info_scale", sig0)
def wilson(k, n):
    return beta.ppf(.025, k, n - k + 1) if k else 0.0, beta.ppf(.975, k + 1, n - k) if k < n else 1.0
def ar1(n_sites, T, rho, sg, size):
    e = rng.standard_normal((size, n_sites, T)); x = np.empty_like(e); x[..., 0] = e[..., 0]
    for t in range(1, T): x[..., t] = rho * x[..., t - 1] + np.sqrt(1 - rho ** 2) * e[..., t]
    return sg * x.reshape(size, -1)
def run(tag, ids, a, mult, noise="white", est="lsq", infl=1.0):
    Gs = G3[ids].reshape(-1, K); sg = sig0 * mult
    M = fisher(Gs, sg)  # assumed model: white noise, sigma = sg
    cr = criteria(M); w = np.linalg.eigvalsh(M); cond = float(w[-1] / max(w[0], 1e-300))
    log(f"{tag}.cond_M", cond); log(f"{tag}.float_rank", cr["rank"])
    if not cr["defined"]:
        return dict(tag=tag, defined=False, cond=cond)
    eps = (ar1(len(ids), HOURS, 0.8, sg, NDRAW) if noise == "ar1" else sg * infl * rng.standard_normal((NDRAW, Gs.shape[0])))
    y = Gs @ a + eps
    if est == "lsq":
        ah = np.linalg.lstsq(Gs, y.T, rcond=None)[0].T
    else:
        ah = np.array([solve_nnls(Gs, yy, noise_cov=sg)["coef"] for yy in y])
    d = ah - a; C = np.linalg.inv(M)
    t4 = np.einsum("ni,ij,nj->n", d, M, d) <= chi2.ppf(.95, K)
    d2 = d[:, :2]; t2 = np.einsum("ni,ij,nj->n", d2, np.linalg.inv(C[:2, :2]), d2) <= chi2.ppf(.95, 2)
    fe = np.sqrt(np.mean((d @ Gall.T) ** 2, 1)); ref = np.sqrt(np.mean((a @ Gall.T) ** 2))
    emp_cov = np.cov(d.T) if True else None
    out = dict(tag=tag, defined=True, cond=cond, cov4=t4.mean(), cov4_ci=wilson(int(t4.sum()), NDRAW), cov2=t2.mean(),
               cov2_ci=wilson(int(t2.sum()), NDRAW), var_ratio_max=float(np.max(np.diag(emp_cov) / np.diag(C))),
               var_ratio_min=float(np.min(np.diag(emp_cov) / np.diag(C))), frac_at_bound=float((ah <= 0).mean()),
               param_rel_rmse=float(np.sqrt((d ** 2).sum(1).mean()) / np.linalg.norm(a)),
               field_rel_rmse=float(np.sqrt((fe ** 2).mean()) / ref))
    for k in ("cov4", "cov2", "var_ratio_max", "var_ratio_min", "frac_at_bound", "param_rel_rmse", "field_rel_rmse"): log(f"{tag}.{k}", out[k])
    return out
inf, red, cov = meta["sites"]["informative"], meta["sites"]["redundant"], meta["sites"]["coverage"]
sd = np.sqrt(np.diag(np.linalg.inv(fisher(G3[inf].reshape(-1, K), sig0))))
log("crb_sd_max_over_min_a", sd.max() / a_true.min()); log("crb_sd_max", sd.max()); log("a_true_min", a_true.min())
ab = a_true.copy(); ab[0] = 0.0            # truth on the nonnegativity boundary
ah1 = a_true.copy(); ah1[0] = sd[0]        # truth one CRB-sd from the boundary
res = []
for mult in (1, 100, 1000):
    for tn, a in (("interior", a_true), ("1sd_from_0", ah1), ("on_boundary", ab)):
        for est in ("lsq", "nnls"):
            res.append(run(f"A.m{mult}.{tn}.{est}", inf, a, mult, est=est))
for tn, kw in (("ar1_rho0.8", dict(noise="ar1")), ("sigma_x0.5", dict(infl=0.5)), ("sigma_x2", dict(infl=2.0))):
    res.append(run(f"B.misspec.{tn}", inf, a_true, 1, **kw))
for dn, ids in (("redundant", red), ("coverage", cov)):
    for mult in (1, 100):
        res.append(run(f"C.{dn}.m{mult}", ids, a_true, mult))
rd = Path(os.environ["RESEARCH_RUN_DIR"])
(rd / "outputs").mkdir(exist_ok=True)
(rd / "outputs/audit_mc.json").write_text(json.dumps(res, indent=1, default=float))
with open(rd / "metrics.jsonl", "a") as f: f.writelines(json.dumps(x) + "\n" for x in metrics)
with open(rd / "artifacts.jsonl", "a") as f: f.write(json.dumps({"path": str(rd / "outputs/audit_mc.json")}) + "\n")
for r in res: print(json.dumps(r, default=float))
