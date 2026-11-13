"""E05: aggregate e05-duration-{A,B}.csv -> tables/e05-duration.csv, figures/e05-*.png, chapters/05-time.qmd.
Run: `uv run python scripts/e05_report.py`. Every number in the chapter is computed here from the CSVs.
Design follows the gate-05 audit (docs/research/gates/gate-05-duration.md): medians only (no means of errors), fixed readings per
sensor when comparing windows, seed as the unit of replication, 'constant' = exactly constant wind as a null control.
"""
import subprocess, sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah.experiment_config import SourceConfig  # noqa: E402

A_REF = SourceConfig().reference_rate_ug_s
T, F = root / "reports/quarto/tables", root / "reports/quarto/figures"
df = pd.concat([pd.read_csv(T / f"e05-duration-{b}.csv") for b in "AB"], ignore_index=True)
df.to_csv(T / "e05-duration.csv", index=False)
df["rel"] = df.src_rmse / df.true_rms
df["full"] = df.prank_aref1 == df.p
sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
NS2 = float(df.noise_scale[df.noise_scale != 1.0].iloc[0])
WINDS = [w for w in ("constant", "drifting", "changing") if w in set(df.wind)]
M_FIX = 4  # readings per sensor held fixed when comparing windows (W = 1 h cannot hold 4 readings)
A = df[(df.branch == "A") & (df.noise_scale == 1.0)]
rng = np.random.default_rng(4100)


def md(d):
    f = lambda v: format(v, ".3g") if isinstance(v, (float, np.floating)) else str(v)
    return "\n".join(["| " + " | ".join(map(str, d.columns)) + " |", "|" + "---|" * len(d.columns)]
                     + ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)])


def boot_ci(x, n=10000):
    x = np.asarray(x, float)
    m = x[rng.integers(0, len(x), (n, len(x)))].mean(1)
    return np.percentile(m, [2.5, 97.5])


def sign_p(d):
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    return binomtest(pos, pos + neg).pvalue if pos + neg else 1.0


# ---- 1. window effect at fixed m, per wind
fx = A[A.m == M_FIX].groupby(["wind", "K", "N", "W"]).agg(mean_prank=("prank_aref1", "mean"), frac_full=("full", "mean"),
                                                          median_rel_err=("rel", "median")).reset_index()
fx["wind"] = pd.Categorical(fx.wind, WINDS); fx = fx.sort_values(["wind", "K", "N", "W"])

# ---- 2. window gain (W 6 -> 168 h, m = M_FIX) per seed, averaged over K and N; then wind-minus-constant (difference in differences)
g = A[(A.m == M_FIX) & A.W.isin([6, 168])].pivot_table(index=["wind", "seed", "K", "N"], columns="W", values="prank_aref1")
g = (g[168] - g[6]).rename("gain").reset_index()
per_seed = g.groupby(["wind", "seed"]).gain.mean().unstack("wind")
did_rows = []
for w in WINDS:
    x = per_seed[w]
    did_rows.append(dict(contrast=f"window gain under {w}", mean=x.mean(), ci_lo=boot_ci(x)[0], ci_hi=boot_ci(x)[1],
                         seeds_pos=int((x > 0).sum()), seeds_neg=int((x < 0).sum()), sign_p=sign_p(x)))
for w in WINDS[1:]:
    x = per_seed[w] - per_seed["constant"]
    did_rows.append(dict(contrast=f"{w} minus constant (DiD)", mean=x.mean(), ci_lo=boot_ci(x)[0], ci_hi=boot_ci(x)[1],
                         seeds_pos=int((x > 0).sum()), seeds_neg=int((x < 0).sum()), sign_p=sign_p(x)))
did = pd.DataFrame(did_rows)

# ---- 3. matched reading budget B = N m at W = 168 h, paired by seed
mb = A[(A.W == 168) & A.B.isin([8, 16, 32])].groupby(["wind", "K", "B", "N", "m"]).agg(
    mean_prank=("prank_aref1", "mean"), median_rel_err=("rel", "median")).reset_index()
mb = mb[mb.wind == "constant"].drop(columns="wind")
pair_rows = []
for (K, Bt), s in A[(A.W == 168) & A.B.isin([16, 32]) & (A.wind == "constant")].groupby(["K", "B"]):
    piv = s.pivot_table(index="seed", columns="N", values="prank_aref1")
    for a, b in ((8, 4), (8, 2), (4, 2)):
        if a in piv and b in piv:
            d = piv[a] - piv[b]
            pair_rows.append(dict(K=K, B=Bt, comparison=f"N={a} vs N={b}", mean_diff=d.mean(), seeds_pos=int((d > 0).sum()),
                                  seeds_neg=int((d < 0).sum()), sign_p=sign_p(d)))
pairs = pd.DataFrame(pair_rows)

# ---- 4. fewer sensors x longer window vs more sensors snapshot, at EQUAL readings (N=4, m=2, W=168 vs N=8 snapshot)
eq = []
for w in WINDS:
    for K in sorted(A.K.unique()):
        s = A[(A.wind == w) & (A.K == K)]
        a = s[(s.N == 4) & (s.m == 2) & (s.W == 168)].set_index("seed").prank_aref1
        b = s[(s.N == 8) & (s.m == 1) & (s.W == 1)].set_index("seed").prank_aref1
        d = a - b
        eq.append(dict(wind=w, K=K, prank_N4_m2_W168=a.mean(), prank_N8_snapshot=b.mean(), seeds_pos=int((d > 0).sum()),
                       seeds_neg=int((d < 0).sum()), sign_p=sign_p(d)))
eq = pd.DataFrame(eq)

# ---- 5. Branch B: constant-component error relative to Branch A, paired cell by cell (medians)
key = ["wind", "seed", "K", "N", "m", "W", "noise_scale"]
bb = df[df.branch == "B"].set_index(key).src_rmse_const / df[(df.branch == "A") & (df.K == 2)].set_index(key).src_rmse
bt = bb.dropna().groupby(level="noise_scale").median().rename("median ratio B/A").reset_index()
bp = df[df.branch == "B"].groupby(["noise_scale", "wind"]).agg(p=("p", "first"), mean_prank=("prank_at_noise", "mean"),
                                                                 median_rel_err_const=("src_rmse_const", "median")).reset_index()

o1 = pd.concat([pd.read_csv(T / f"e05-oracles-{b}.csv").assign(branch=b) for b in "AB"], ignore_index=True)

# ---- figures (medians / means of ranks only)
fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
for j, K in enumerate((2, 4)):
    for w, ls in zip(WINDS, ("-", ":", "--")):
        for N in (2, 4, 8):
            s = fx[(fx.wind == w) & (fx.K == K) & (fx.N == N)]
            ax[0, j].plot(s.W, s.mean_prank, ls, marker="o", color=f"C{N // 4}", label=f"{w}, N={N}")
            ax[1, j].plot(s.W, s.median_rel_err, ls, marker="o", color=f"C{N // 4}")
    ax[0, j].set(xscale="log", title=f"K={K}, m={M_FIX} readings/sensor, preregistered noise", ylabel="mean practical rank")
    ax[1, j].set(xscale="log", yscale="log", xlabel="window W [h]", ylabel="median source error / rms strength")
    ax[1, j].axhline(1, color="k", lw=.8, ls=":")
ax[0, 0].legend(fontsize=7)
fig.tight_layout(); fig.savefig(F / "e05-rank-vs-window.png", dpi=110); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
for k, w in enumerate(WINDS):
    x = per_seed[w]
    ax.scatter(np.full(len(x), k) + rng.uniform(-.1, .1, len(x)), x, s=14)
    ax.plot([k - .25, k + .25], [x.mean()] * 2, "k")
ax.axhline(0, color="k", lw=.6, ls=":")
ax.set(xticks=range(len(WINDS)), xticklabels=WINDS, ylabel="practical-rank gain, W 6 h -> 168 h (per seed)",
       title=f"Window gain by wind (m={M_FIX}, mean over K, N)")
fig.tight_layout(); fig.savefig(F / "e05-source-rmse.png", dpi=110); plt.close(fig)
(F / "e05-field-rmse.png").unlink(missing_ok=True)  # retired: in-sample field score (gate-05 finding 8)

n_seeds = df.seed.nunique()
null_gain = did.set_index("contrast").loc["window gain under constant"]
qmd = f"""# E05 Temporal observability: sensor count versus observation window

Simulated passive tracer only; nothing here is evidence about ambient PM2.5 or real exposure. Numbers are computed by
`scripts/e05_report.py` from `tables/e05-duration.csv`, written by `scripts/e05_duration.py` (run records in `reports/runs/runs.jsonl`).
Chapter generated at revision `{sha}`. **Everything here is exploratory**: the source scale (amendment A1, a_ref = {A_REF:g} µg s⁻¹,
`state/decisions/2026-11-12-source-scale.md`), the constant-wind control and the analysis contrasts were chosen after seeing earlier
E05 results and the independent audit (`docs/research/gates/gate-05-duration.md`).

## Design (what was actually run)

- 32² grid, 12 h spin-up, perfect model, known source locations and winds, ideal white sensors, {n_seeds} seeds (each seed changes
  sources, strengths, random nested sensor sites and noise draws; noise draws are identical across winds).
- Sensors N ∈ {{2, 4, 8}} (random, nested, never optimised); sources K ∈ {{2, 4}}; window W ∈ {{1, 6, 24, 168}} h ending at the issue
  hour; readings per sensor m ∈ {{1, 2, 4, 8, 16}} spread over W (m = 1 is the reading at the issue hour; m > W is not run).
- Winds: **constant** = exactly 3 m s⁻¹ from 270° (null control); **drifting** = the benchmark's steady westerly, whose direction is
  re-drawn N(270°, 10°) every 48 h; **changing** = veering 270° → 315° during the first 12 h of each 48 h block. Drifting and changing
  share speed factors.
- Unknowns: Branch A p = K constant amplitudes; Branch B p = 3K diurnal coefficients (K = 2) while the true source is constant.
- Noise: preregistered σ (= {df.sigma_prereg.median():.3g} µg m⁻³ median) and a non-preregistered {NS2:g}× level.
- Estimator: unconstrained MAP / ridge least squares with prior sd a_ref per coefficient (no positivity). Practical rank = number of
  singular values s of G/σ with s·a_ref ≥ 1 (a mode a reference-sized change moves by ≥ 1σ; lenient, see audit finding 7).

## Validation checks

{md(o1)}

## Results (Branch A, preregistered σ)

### Window length at fixed readings per sensor (m = {M_FIX})

{md(fx)}

![Practical rank and median source error versus window](../figures/e05-rank-vs-window.png)

### Does elapsed time help, or only new wind directions?

Per seed: gain in practical rank from W = 6 h to 168 h (m = {M_FIX}, averaged over K and N), and the difference between each varying
wind and the constant control. 95 % CI: bootstrap over seeds; sign test over seeds.

{md(did)}

Under the exactly constant wind the mean window gain is {null_gain['mean']:.3g} (CI {null_gain['ci_lo']:.3g} to {null_gain['ci_hi']:.3g}).

![Window gain per seed by wind](../figures/e05-source-rmse.png)

### Fewer sensors over a longer window, at equal readings (N = 4, m = 2, W = 168 h vs N = 8 snapshot)

{md(eq)}

### Matched reading budget at W = 168 h, constant wind

{md(mb)}

Paired by seed:

{md(pairs)}

### Basis branch B (p = 3K, true source constant)

{md(bp)}

Paired median ratio of B's constant-component error to A's error, same cell and noise draws:

{md(bt)}

## Interpretation (kept separate from the observations)

The hypothesis is that changing wind directions raise effective rank for constant sources, while repeated readings under the same
transport only reduce noise. Read the window-gain table first: the constant-wind row isolates elapsed time, and the
difference-in-differences rows isolate wind change. A gain under constant wind would contradict the second half of the
hypothesis; a positive DiD supports the first. The equal-readings and matched-budget tables address the sensor-savings question
without letting reading count stand in for window length. Practical rank ≥ 1 still allows roughly 100 % error per mode, so the median
error columns matter as much as the rank columns.

## Limitations

Random, not optimised, sites; synthetic perfect model; 32² grid; one random-site draw per seed; white noise only (no AR(1) arm);
no held-out forecast or off-sensor field evaluation (the earlier in-sample field score was retired); the veering regime turns only
during the first 12 h of each block; Branch B is K = 2 only. These results motivate, and do not replace, a preregistered E08.
"""
(root / "reports/quarto/chapters/05-time.qmd").write_text(qmd)
print(did.to_string()); print(eq.to_string()); print(pairs.to_string())
