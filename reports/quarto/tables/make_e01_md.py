"""Render tables/e01-verification.csv into markdown fragments included by chapters/01-forward.qmd."""
import pandas as pd, pathlib
here = pathlib.Path(__file__).parent
d = pd.read_csv(here / "e01-verification.csv")
g = lambda grp: d[d.group == grp].dropna(axis=1, how="all")
def md(df):
    f = lambda v: f"{v:.3g}" if isinstance(v, float) else str(v)
    rows = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * len(df.columns)]
    return "\n".join(rows + ["| " + " | ".join(f(v) for v in r) + " |" for r in df.itertuples(index=False)]) + "\n"
o = g("order").assign(floor=0.8)
o["passes"] = o.observed_order >= 0.8
fin = o[o.case.str.endswith("32_64")]
out = {
 "grid": md(g("grid")[["case", "n", "dt", "rel_l2", "peak_ratio", "min_over_max", "mass_rel_err"]]),
 "order": md(o[["case", "observed_order", "floor", "passes"]])
   + f"\nFinest-pair rule (order ≥ p − 0.2 = 0.8): {int(fin.passes.sum())} of {len(fin)} oracles pass.\n",
 "dt": md(g("dt")[["case", "dt", "courant", "rel_l2", "peak_ratio"]]),
 "cfl": md(g("cfl_peclet")[["case", "U", "courant", "peclet", "rel_l2", "peak_ratio", "min_over_max"]]),
 "mass": md(g("mass")),
}
for k, v in out.items():
    (here / f"e01-{k}.md").write_text(v)
