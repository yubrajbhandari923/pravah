"""E02 tiny local benchmark: 3 hand-placed examples, validity checks, figure and tables (T-011).

Run: `uv run python scripts/e02_examples.py`. Uses the T-010 pipeline unchanged
(benchmarks.generate_scenario); only the random source draw is replaced by fixed cells so the geometry is
controlled (documented override: truth_id does not encode it, so each example gets its own replicate seed).
Small settings (32^2 grid, 12 h spin-up, 8-day calendar) are a smoke scale, NOT the preregistered 64^2 / 60-day one.
Metrics go to $RESEARCH_RUN_DIR/metrics.jsonl; every table/figure number is written from computed values.
"""
import copy, csv, hashlib, json, os, sys, tempfile
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(root / "src"))
from pravah import benchmarks as B  # noqa: E402
from pravah.experiment_config import (SCHEMA_VERSION, SourceConfig, ExperimentConfig, LeakError, Split, Winds,  # noqa: E402
                                      DataManifest, check_no_leak, dump_config, load_config, truth_id, _git)

AR = SourceConfig().reference_rate_ug_s  # amendment A1
# name -> (replicate, lattice cells, strengths ug/s (multiples of a_ref / 100, kept from the a_ref = 100 era), train wind regimes, snapshot hour, one-line purpose)
EXAMPLES = {
    "separated2": (0, [(3, 3), (8, 8)], [2 * AR, 1 * AR], None, 36, "two sources ~5.9 km apart, mixed winds"),
    "collinear2": (1, [(3, 6), (6, 6)], [1.5 * AR, 1 * AR], ["steady_westerly"], 36,
                   "two sources on one line along the steady wind: downstream plume overlaps"),
    "changing4": (2, [(2, 2), (2, 9), (8, 3), (9, 9)], [1.2 * AR, 2.5 * AR, .8 * AR, 1.8 * AR], None, 60,
                  "four sources, westerly then veering wind"),
}
LABEL = {"separated2": "A: well-separated 2-source", "collinear2": "B: collinear 2-source (ambiguous)",
         "changing4": "C: 4-source, changing wind"}


def make_cfg(train_winds=None):
    c = ExperimentConfig.default("EXP-003")
    c.model.grid, c.model.spinup_h = (32, 32), 12
    c.design.n_sources = (2, 4)
    c.split = Split(train=(1, 4), validation=(5, 5), test=(6, 8))
    if train_winds:
        c.winds = Winds(train=train_winds, heldout=c.winds.heldout)
    c.validate()
    return c


@contextmanager
def fixed_sources(cells, a):
    orig = B.draw_sources
    B.draw_sources = lambda cfg, rep, n: (list(cells), np.array(a, float))
    try:
        yield
    finally:
        B.draw_sources = orig


def build_examples(names=None):
    out = {}
    for n in names or EXAMPLES:
        rep, cells, a, tw, _, _ = EXAMPLES[n]
        with fixed_sources(cells, a):
            out[n] = B.generate_scenario(make_cfg(tw), rep, len(cells))
    return out


# ---------------------------------------------------------------- checks
def mass_budget(sc):
    """Anomaly-mass budget over hours 0..N-1 of the stored field (spin-up state is not stored).
    outflow is the RESIDUAL emitted - removed - dM; removal uses an hourly trapezoid, so it is approximate."""
    cfg, t = sc.cfg, sc.truth
    m, H = cfg.model, cfg.model.mixing_depth_m
    dx, dy = m.domain_m[0] / m.grid[0], m.domain_m[1] / m.grid[1]
    S = sum(a * B.source_basis(tuple(m.grid), tuple(m.domain_m), c, cfg.sources.footprint_sigma_m)
            for a, c in zip(t.strengths_ug_s, t.source_xy)) / H           # ug m^-3 s^-1
    rate_err = abs(S.sum() * dx * dy * H - t.strengths_ug_s.sum()) / t.strengths_ug_s.sum()
    M = (t.field - cfg.sources.background_ug_m3).sum((1, 2)) * dx * dy * H  # ug, anomaly column mass
    n = len(M)
    emitted = t.strengths_ug_s.sum() * 3600 * (n - 1)
    removed = m.removal_per_s * 3600 * (M[1:] + M[:-1]).sum() / 2
    outflow = emitted - removed - (M[-1] - M[0])
    return dict(emitted_ug=float(emitted), removed_ug=float(removed), stored_ug=float(M[-1]),
                stored_change_ug=float(M[-1] - M[0]), outflow_ug=float(outflow), outflow_frac=float(outflow / emitted),
                source_rate_err=float(rate_err), min_anomaly=float((t.field - cfg.sources.background_ug_m3).min()))


def _design(sc, n, descending):
    adm = sc.admissible_sites()
    shared = adm[len(adm) // 2]
    ids = [int(shared)]
    for s in (adm[::-1] if descending else adm):
        if len(ids) == n:
            break
        d = np.linalg.norm(sc.site_xy[ids] - sc.site_xy[s], axis=1)
        if (d >= sc.cfg.design.min_sensor_separation_m).all():
            ids.append(int(s))
    return ids


def _raises(exc, f):
    try:
        f()
    except exc:
        return True
    except Exception:
        return False
    return False


def leak_checks(sc):
    ids = _design(sc, 4, False)
    ok = lambda f: not _raises(Exception, f)
    v = sc.view("train", ids, "placement")
    adm = sc.admissible_sites()
    ring = [i for i in range(sc.n_candidate_sites) if i not in adm][0]
    close = [int(ids[0]), int(ids[0]) + 1]
    r = [("train placement allowed", ok(lambda: sc.view("train", ids, "placement"))),
         ("validation tuning allowed", ok(lambda: sc.view("validation", ids, "tuning"))),
         ("test evaluation allowed", ok(lambda: sc.view("test", ids, "evaluation"))),
         ("test placement refused", _raises(LeakError, lambda: sc.view("test", ids, "placement"))),
         ("test tuning refused", _raises(LeakError, lambda: sc.view("test", ids, "tuning"))),
         ("validation evaluation refused", _raises(LeakError, lambda: sc.view("validation", ids, "evaluation"))),
         ("held-out wind in fitting refused", _raises(LeakError, lambda: check_no_leak(
             sc.cfg.split, "tuning", [1], sc.cfg.winds.heldout[:1], sc.cfg.winds))),
         ("inadmissible site refused", _raises(ValueError, lambda: sc.view("train", [ring], "placement"))),
         ("too-close sites refused", _raises(ValueError, lambda: sc.view("train", close, "placement"))),
         ("train/val/test hours disjoint", len({*sc.split_hours("train"), *sc.split_hours("validation"),
                                                *sc.split_hours("test")}) == len(sc.truth.winds)),
         ("train winds exclude held-out regimes",
          not set(sc.truth.wind_regimes["train"]) & set(sc.cfg.winds.heldout)),
         ("view arrays read-only", not any(getattr(v, f.name).flags.writeable for f in fields(v)
                                           if isinstance(getattr(v, f.name), np.ndarray))),
         ("view exposes no truth/strength fields", not {f.name for f in fields(v)} & {"field", "strengths_ug_s",
                                                                                      "site_truth", "truth"})]
    return r


def same_truth_across_designs(sc):
    h = lambda: hashlib.sha256(b"".join(a.tobytes() for a in (
        sc.truth.field, sc.truth.site_truth, sc.truth.strengths_ug_s, sc.truth.winds))).hexdigest()
    before, tid = h(), sc.truth_id
    d1, d2 = _design(sc, 4, False), _design(sc, 6, True)
    v1, v2 = sc.view("train", d1, "placement"), sc.view("train", d2, "placement")
    shared = sorted(set(d1) & set(d2))
    r1 = {int(i): v1.readings[k] for k, i in enumerate(v1.site_ids)}
    r2 = {int(i): v2.readings[k] for k, i in enumerate(v2.site_ids)}
    return bool(d1 != d2 and shared and h() == before and truth_id(sc.cfg, sc.replicate, sc.n_sources) == tid
                and all(np.array_equal(r1[i], r2[i]) for i in shared))


def versioning_checks(cfg):
    with tempfile.TemporaryDirectory() as d:
        dump_config(cfg, Path(d) / "c.json")
        rt = load_config(Path(d) / "c.json")
    c2 = copy.deepcopy(cfg)
    c2.model.diffusivity_m2_s *= 2
    return [("config JSON round-trip keeps hash", rt.config_hash() == cfg.config_hash()),
            ("truth_id stable under reload", truth_id(rt, 0, 2) == truth_id(cfg, 0, 2)),
            ("truth_id changes when physics config changes", truth_id(c2, 0, 2) != truth_id(cfg, 0, 2)),
            ("schema_version recorded", cfg.schema_version == SCHEMA_VERSION)]


def splits_table(ex):
    sha = _git("rev-parse", "HEAD") or "unknown"
    rows = []
    for n, sc in ex.items():
        c = sc.cfg
        for s in B.SPLITS:
            lo, hi = getattr(c.split, s)
            rows.append(dict(scenario=n, truth_id=sc.truth_id, schema_version=c.schema_version,
                             config_hash=c.config_hash()[:12], git_sha=sha[:10], replicate=sc.replicate,
                             truth_seed=c.seeds.streams(sc.replicate)["truth"],
                             noise_seed=c.seeds.streams(sc.replicate)["noise"], n_sources=sc.n_sources, split=s,
                             days=f"{lo}-{hi}", n_hours=len(sc.split_hours(s)),
                             wind_regimes="|".join(sc.truth.wind_regimes[s]),
                             fitting_allowed=str(s != "test"), evaluation_allowed=str(s == "test")))
    return rows


# ---------------------------------------------------------------- figure
def figure(ex, path):
    fig, ax = plt.subplots(len(ex), 4, figsize=(19, 4.4 * len(ex)), constrained_layout=True)
    for r, (n, sc) in enumerate(ex.items()):
        cfg, t = sc.cfg, sc.truth
        m = cfg.model
        Lx, Ly = m.domain_m
        ext = (0, Lx / 1e3, 0, Ly / 1e3)
        dx, dy = Lx / m.grid[0], Ly / m.grid[1]
        snap = EXAMPLES[n][4]
        # 1 emissions [ug s^-1 m^-2] = sum a_k phi_k
        E = sum(a * B.source_basis(tuple(m.grid), tuple(m.domain_m), c, cfg.sources.footprint_sigma_m)
                for a, c in zip(t.strengths_ug_s, t.source_xy))
        im = ax[r, 0].imshow(E.T, origin="lower", extent=ext, cmap="magma")
        fig.colorbar(im, ax=ax[r, 0], label="emission density, ug s$^{-1}$ m$^{-2}$")
        for a, c in zip(t.strengths_ug_s, t.source_xy):
            ax[r, 0].annotate(f"{a:.0f} ug/s", c / 1e3, color="w", ha="center", va="bottom", fontsize=8)
        ax[r, 0].set_title(f"{LABEL[n]}\ntrue emissions (hidden)")
        # 2 plume + candidate sensors (noisy white reading) + wind arrow
        bg, sg = cfg.sources.background_ug_m3, sc.noise_sigma[0]
        f = t.field[snap] - bg
        vmax = f.max()
        pm = ax[r, 1].pcolormesh(np.arange(m.grid[0] + 1) * dx / 1e3, np.arange(m.grid[1] + 1) * dy / 1e3, f.T,
                                 cmap="viridis", vmin=0, vmax=vmax)
        reading = t.site_truth[:, snap] + sc._noise["white"][:, snap] - bg
        sd = ax[r, 1].scatter(*(sc.site_xy / 1e3).T, c=reading, cmap="coolwarm", vmin=-3 * sg, vmax=3 * sg, s=22,
                              edgecolors="k", linewidths=.3)
        fig.colorbar(sd, ax=ax[r, 1], label="noisy reading - background, ug m$^{-3}$ (dots)", location="bottom", pad=.12)
        u = t.winds[snap]
        ax[r, 1].quiver([.8], [9.2], [u[0]], [u[1]], color="r", scale=12, width=.01)
        ax[r, 1].plot(*(t.source_xy / 1e3).T, "r*", ms=11)
        fig.colorbar(pm, ax=ax[r, 1], label="true anomaly c - c_bg, ug m$^{-3}$")
        ax[r, 1].set_title(f"plume at hour {snap} (peak {vmax:.1e}); dots: noisy sensors, sigma={sg:.2f}\nred arrow: wind {np.hypot(*u):.1f} m/s")
        # 3 time series at the 4 admissible sites with the highest train-median
        adm = sc.admissible_sites()
        tr = sc.split_hours("train")
        top = adm[np.argsort(-np.median(t.site_truth[adm][:, tr], axis=1))[:4]]
        for k in top:
            l, = ax[r, 2].plot(t.site_truth[k], lw=1.4, label=f"site {k}")
            ax[r, 2].plot(t.site_truth[k] + sc._noise["white"][k], ".", ms=2.5, color=l.get_color(), alpha=.5)
        # 4 wind components
        ax[r, 3].plot(t.winds[:, 0], label="$u_x$"); ax[r, 3].plot(t.winds[:, 1], label="$u_y$")
        for col in (2, 3):
            for s, c in (("validation", "tab:orange"), ("test", "tab:red")):
                h = sc.split_hours(s)
                ax[r, col].axvspan(h[0], h[-1], color=c, alpha=.08)
            ax[r, col].set_xlabel("hour after spin-up (shaded: validation, test)")
        ax[r, 2].set_ylabel("ug m$^{-3}$"); ax[r, 2].set_title("candidate-site series: truth (line), noisy readings (dots)")
        ax[r, 3].set_ylabel("m s$^{-1}$"); ax[r, 3].set_title("wind components"); ax[r, 3].legend(loc="upper right")
        ax[r, 2].legend(loc="upper right", fontsize=7)
        for c in (0, 1):
            ax[r, c].set_xlabel("x, km"); ax[r, c].set_ylabel("y, km")
    fig.savefig(path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main
def main():
    ex = build_examples()
    tab = root / "reports" / "quarto" / "tables"
    figp = root / "reports" / "quarto" / "figures" / "e02-truth-panels.png"
    rows, metrics = [], []

    def rec(sc, check, value, ok):  # ok=None -> informational (not a pass/fail gate)
        rows.append(dict(scenario=sc, check=check, value=value, passed="info" if ok is None else bool(ok)))

    for n, sc in ex.items():
        mb = mass_budget(sc)
        for k, v in mb.items():
            metrics.append((f"{n}.{k}", v))
        rec(n, "source rate integral matches sum(a) (rel err)", f"{mb['source_rate_err']:.2e}", mb["source_rate_err"] < 1e-12)
        rec(n, "anomaly never negative (min, ug m^-3)", f"{mb['min_anomaly']:.2e}", mb["min_anomaly"] >= -1e-9)
        rec(n, "mass-budget outflow fraction in [0,1] (residual)", f"{mb['outflow_frac']:.3f}", 0 <= mb["outflow_frac"] <= 1)
        for case, ok in leak_checks(sc):
            rec(n, case, "ok" if ok else "FAILED", ok)
        pk = float((sc.truth.field - sc.cfg.sources.background_ug_m3).max())
        sig = float(sc.noise_sigma[0])
        rec(n, "peak plume anomaly over background (ug m^-3)", f"{pk:.2e}", None)
        rec(n, "instrument noise sigma (ug m^-3)", f"{sig:.3f}", None)
        rec(n, "SNR = peak anomaly / noise sigma (NEGATIVE RESULT if << 1)", f"{pk / sig:.2e}", None)
        metrics += [(f"{n}.peak_anomaly_ug_m3", pk), (f"{n}.noise_sigma_ug_m3", sig), (f"{n}.snr_peak", pk / sig)]
        st = same_truth_across_designs(sc)
        rec(n, "identical truth and shared-site noise across two designs", str(st), st)
        metrics += [(f"{n}.same_truth", float(st)), (f"{n}.peak_ug_m3", float(sc.truth.field.max()))]
    for case, ok in versioning_checks(ex["separated2"].cfg):
        rec("all", case, "ok" if ok else "FAILED", ok)
    ids = {sc.truth_id for sc in ex.values()}
    rec("all", "three distinct truth_ids", str(len(ids)), len(ids) == 3)
    metrics += [("checks_total", float(len(rows))), ("checks_failed", float(sum(r["passed"] is False for r in rows)))]

    for name, data in (("e02-splits.csv", splits_table(ex)), ("e02-checks.csv", rows)):
        with open(tab / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(data[0]))
            w.writeheader(); w.writerows(data)
    figure(ex, figp)

    rd = os.environ.get("RESEARCH_RUN_DIR")
    outs = [tab / "e02-splits.csv", tab / "e02-checks.csv", figp]
    if rd:
        od = Path(rd) / "outputs"; od.mkdir(parents=True, exist_ok=True)
        for n, sc in ex.items():
            np.savez_compressed(od / f"{n}.npz", field=sc.truth.field.astype(np.float32), winds=sc.truth.winds,
                                site_truth=sc.truth.site_truth, source_xy=sc.truth.source_xy,
                                strengths_ug_s=sc.truth.strengths_ug_s)
            outs.append(od / f"{n}.npz")
        man = DataManifest.build(outs, Path("/"))  # entries are filesystem paths relative to /; run dirs live outside the repo
        (od / "manifest.json").write_text(json.dumps({"schema_version": SCHEMA_VERSION, "entries": man.entries}, indent=1))
        outs.append(od / "manifest.json")
        with open(Path(rd) / "metrics.jsonl", "a") as f:
            for k, v in metrics:
                f.write(json.dumps({"name": k, "value": v, "step": 0}) + "\n")
        with open(Path(rd) / "artifacts.jsonl", "a") as f:
            for p in outs:
                f.write(json.dumps({"path": str(p.resolve())}) + "\n")
    bad = [r for r in rows if r["passed"] is False]
    print(f"{len(rows)} checks, {len(bad)} failed")
    for r in bad:
        print("FAILED", r)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
