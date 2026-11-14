"""E14 reliability diagnostics and calibration (design: docs/research/reliability-design.md). No I/O.

Inputs are whitened: z (k, T, R) = (y - c_bg)/sigma, A (k, T, K) = a_ref G / sigma. Diagnostics (§4) take only z, A and
the MODEL responses Fr (no truth argument exists); labels and calibration (§5–6) are the only functions that read truth.
"""
import numpy as np
from scipy.stats import norm, rankdata

DIAGS = ("innov_ac1", "nis", "resid_xcorr", "loso", "spread", "drift", "sensor_skill", "sensor_fit")
LEVELS = (0.5, 0.9, 0.95)
W_H = 24


def map_series(A, z, hours, window):
    """MAP (len(hours), K, R) and covariance (len(hours), K, K) of a from the `window` hours ending at each hour (None =
    all past), prior N(0, I). Same algebra as scripts/e09_filter.map_series with prior_mean 0, prior_var 1."""
    K, hours = A.shape[2], np.asarray(hours)
    cP = np.concatenate([np.zeros((1, K, K)), np.cumsum(np.einsum("shk,shl->hkl", A, A), 0)])
    cb = np.concatenate([np.zeros((1, K) + z.shape[2:]), np.cumsum(np.einsum("shk,sh...->hk...", A, z), 0)])
    lo = np.zeros_like(hours) if window is None else np.maximum(hours + 1 - window, 0)
    Nt = np.eye(K) + cP[hours + 1] - cP[lo]
    return np.linalg.solve(Nt, cb[hours + 1] - cb[lo]), np.linalg.inv(Nt)


def _ac1(e):
    """Lag-1 autocorrelation pooled over sites; e (k, n, R) -> (R,)."""
    e = e - e.mean(1, keepdims=True)
    return (e[:, 1:] * e[:, :-1]).sum((0, 1)) / (e ** 2).sum((0, 1))


def block_diagnostics(A, z, t_b, window, Fr_next):
    """Truth-free diagnostics (§4) at block start t_b from hours <= t_b. Fr_next (K, N): model unit responses (a_ref
    scaled) at hour t_b + 1. Returns {name: (R,)}."""
    k, R = z.shape[0], z.shape[2]
    W = np.arange(t_b - W_H, t_b)
    u, C = map_series(A, z, np.arange(t_b - W_H, t_b + 1), window)              # (25, K, R), (25, K, K)
    A1 = A[:, W + 1]                                                            # (k, 24, K)
    nu = z[:, W + 1] - np.einsum("sik,ikr->sir", A1, u[:-1])                    # 1 h innovations (k, 24, R)
    S = np.eye(k) + np.einsum("sik,ikl,uil->isu", A1, C[:-1], A1)
    nis = np.einsum("sir,isu,uir->r", nu, np.linalg.inv(S), nu) / (W_H * k)
    if k > 1:
        cc = np.stack([np.corrcoef(nu[..., r]) for r in range(R)])
        xc = (cc.sum((1, 2)) - k) / (k * (k - 1))
    else:
        xc = np.full(R, np.nan)
    lo = np.zeros(R)
    for s in range(k):
        o = np.arange(k) != s
        us, Cs = map_series(A[o], z[o], [t_b], window)
        As = A[s, W]                                                            # (24, K)
        lo += (((z[s, W] - As @ us[0]) ** 2) / (1 + np.einsum("ik,kl,il->i", As, Cs[0], As))[:, None]).mean(0)
    mu = np.einsum("kr,kn->nr", u[-1], Fr_next)
    sd = np.sqrt(np.maximum(np.einsum("kn,kl,ln->n", Fr_next, C[-1], Fr_next), 0))
    d = u[-1] - u[0]
    fit = z[:, W] - np.einsum("sik,ikr->sir", A[:, W], u[:-1])
    return dict(innov_ac1=_ac1(nu), nis=nis, resid_xcorr=xc, loso=lo / k,
                spread=sd.mean() / np.abs(mu).mean(0),
                drift=np.einsum("kr,kl,lr->r", d, np.linalg.inv(C[-1] + C[0]), d) / len(d),
                sensor_skill=np.sqrt((nu ** 2).mean((0, 1)) / ((z[:, W + 1] - z[:, W]) ** 2).mean((0, 1))),
                sensor_fit=np.sqrt((fit ** 2).mean((0, 1))))


def forecast(A, z, issue, window, Fr):
    """Nominal 1 h field forecast at each issue time: mean (n, N, R), sd (n, N), and the strength MAP/sd."""
    u, C = map_series(A, z, issue, window)
    F1 = Fr[:, np.asarray(issue) + 1]                                            # (K, n, N)
    mu = np.einsum("ikr,kin->inr", u, F1)
    sd = np.sqrt(np.maximum(np.einsum("kin,ikl,lin->in", F1, C, F1), 0))
    return mu, sd, u, np.sqrt(np.einsum("ikk->ik", C))


def block_labels(mu, sd, tf, issue, sd_active, a_hat=None, a_sd=None, a_true=None, pit_bins=10):
    """Truth-based labels and calibration for one block (§5–6), per draw. tf (T, N) truth anomaly."""
    issue = np.asarray(issue)
    x = tf[issue + 1][..., None]
    rm = np.sqrt(((mu - x) ** 2).mean((0, 1)))
    rp = np.sqrt(((tf[issue + 1] - tf[issue]) ** 2).mean())
    act = sd >= sd_active
    out = dict(rmse_model=rm, rmse_persist=np.full(mu.shape[2], rp), n_active=np.full(mu.shape[2], act.sum()))
    sa = np.maximum(sd, 1e-300)[..., None]
    zs = (x - mu) / sa
    for lv in LEVELS:
        out[f"cov{int(lv * 100)}"] = (np.abs(zs) <= norm.ppf((1 + lv) / 2))[act].mean(0) if act.any() else np.full(mu.shape[2], np.nan)
    zz = zs[act]
    out["crps"] = (sa[act] * (zz * (2 * norm.cdf(zz) - 1) + 2 * norm.pdf(zz) - 1 / np.sqrt(np.pi))).mean(0) if act.any() \
        else np.full(mu.shape[2], np.nan)
    pit = norm.cdf(zz)
    for j in range(pit_bins):
        out[f"pit{j}"] = ((pit >= j / pit_bins) & (pit < (j + 1) / pit_bins) | ((j == pit_bins - 1) & (pit == 1))).sum(0)
    if a_hat is not None:
        out["cov95_a"] = (np.abs(a_hat - a_true[:, None]) <= 1.96 * a_sd[..., None]).mean((0, 1))
    out["worse_than_persistence"] = rm > rp
    out["coverage_failure"] = out["cov90"] < 0.70
    return out


# ---------------------------------------------------------------- scoring
def auc(score, label):
    """P(score_pos > score_neg) + 0.5 P(tie) (Mann-Whitney); NaN scores dropped; NaN if a class is empty."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    n1, n0 = y.sum(), (~y).sum()
    if not n1 or not n0:
        return np.nan
    return float((rankdata(s)[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def stratified_auc(score, label, strata):
    """AUC over positive/negative pairs within the same stratum only (pair-count-weighted mean of stratum AUCs)."""
    s, y, g = np.asarray(score, float), np.asarray(label, bool), np.asarray(strata)
    num = den = 0.0
    for v in np.unique(g):
        m = (g == v) & np.isfinite(s)
        w = y[m].sum() * (~y[m]).sum()
        if w:
            num, den = num + w * auc(s[m], y[m]), den + w
    return num / den if den else np.nan


def cluster_auc(score, label, cluster, n_boot=2000, seed=4000, strata=None):
    """Pooled (or within-stratum) AUC, per-cluster AUC (median/min/max) and a cluster-bootstrap 95 % CI."""
    s, y, c = np.asarray(score, float), np.asarray(label, bool), np.asarray(cluster)
    g = np.zeros(len(s), int) if strata is None else np.asarray(strata)
    auc_ = lambda j: stratified_auc(s[j], y[j], g[j])
    ids = np.unique(c)
    per = np.array([auc_(c == i) for i in ids])
    rng = np.random.default_rng(seed)
    idx = {i: np.flatnonzero(c == i) for i in ids}
    bs = [auc_(j) for j in (np.concatenate([idx[i] for i in rng.choice(ids, len(ids))]) for _ in range(n_boot))]
    bs = np.asarray(bs)[np.isfinite(bs)]
    return dict(auc=auc_(np.ones(len(s), bool)), n=int(np.isfinite(s).sum()), n_pos=int(y[np.isfinite(s)].sum()), n_clusters=len(ids),
                auc_cl_median=float(np.nanmedian(per)) if np.isfinite(per).any() else np.nan,
                auc_cl_min=float(np.nanmin(per)) if np.isfinite(per).any() else np.nan,
                auc_cl_max=float(np.nanmax(per)) if np.isfinite(per).any() else np.nan,
                ci_lo=float(np.quantile(bs, .025)) if len(bs) else np.nan,
                ci_hi=float(np.quantile(bs, .975)) if len(bs) else np.nan)


def youden_threshold(score, label):
    """Threshold t maximising TPR - FPR for the rule score >= t (validation data only)."""
    s, y = np.asarray(score, float), np.asarray(label, bool)
    ok = np.isfinite(s)
    s, y = s[ok], y[ok]
    if not y.any() or y.all():
        return np.nan
    ts = np.unique(s)
    j = [(s[y] >= t).mean() - (s[~y] >= t).mean() for t in ts]
    return float(ts[int(np.argmax(j))])
