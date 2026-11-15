"""Check that the Quarto report and the evidence matrix point at things that exist.

Checks (exit 1 on any failure):
1. every image / include / .qmd link in reports/quarto/**/*.qmd resolves to a file;
2. every backticked `tables/...` or `figures/...` path (globs and {a,b} allowed) in the .qmd files matches a file;
3. every citation key (@key, [@key]) in the .qmd files is defined in references.bib;
4. every `source_path` in reports/evidence-matrix.csv exists (';'-separated, globs allowed, relative to repo root);
5. every `run_id` in the matrix is a record in reports/runs/*.jsonl (keys run / run_id, or job:<scheduler job>).

Usage: uv run python scripts/validate_report.py [--self-test]
"""
import csv
import glob
import json
import re
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LINK = re.compile(r"\]\(([^)\s]+?\.(?:png|jpg|jpeg|svg|pdf|qmd|md|csv))(?:#[^)]*)?\)")
INCLUDE = re.compile(r"\{\{<\s*include\s+([^\s>]+)\s*>\}\}")
TICK = re.compile(r"`((?:tables|figures)/[^`\s]+)`")
CITE = re.compile(r"(?<![\w.`/])@([A-Za-z][\w:-]*\d{4}[a-z]?)\b")  # project keys are name+year
BIBKEY = re.compile(r"^@\w+\{([^,\s]+),", re.M)


def expand(pattern):
    """Expand one level of {a,b} braces (shell style), then the caller globs each result."""
    m = re.search(r"\{([^{}]*)\}", pattern)
    if not m:
        return [pattern]
    return [p for alt in m.group(1).split(",") for p in expand(pattern[:m.start()] + alt + pattern[m.end():])]


def exists(base, rel):
    return any(glob.glob(str(base / p)) for p in expand(rel))


def check_qmd(qroot):
    errs = []
    keys = set(BIBKEY.findall((qroot / "references.bib").read_text()))
    for f in sorted(qroot.rglob("*.qmd")):
        if "_build" in f.parts:
            continue
        text = f.read_text()
        rel = f.relative_to(qroot)
        for target in LINK.findall(text) + INCLUDE.findall(text):
            if target.startswith(("http://", "https://")):
                continue
            if not (f.parent / target).exists():
                errs.append(f"{rel}: missing link target {target}")
        for t in TICK.findall(text):
            if not exists(qroot, t.rstrip(".,;:")):
                errs.append(f"{rel}: backticked path matches nothing: {t}")
        for k in CITE.findall(text):
            if k not in keys:
                errs.append(f"{rel}: citation @{k} not in references.bib")
    return errs


def run_ids(runs_dir):
    ids = set()
    for f in runs_dir.glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if line.strip():
                d = json.loads(line)
                ids.update(str(d[k]) for k in ("run", "run_id") if d.get(k))
                if d.get("scheduler_job"):
                    ids.add(f"job:{d['scheduler_job']}")
    return ids


def check_matrix(matrix, root, ids):
    errs = []
    with open(matrix, newline="") as fh:
        rows = list(csv.DictReader(fh))
    seen = set()
    for r in rows:
        cid = r["claim_id"]
        if cid in seen:
            errs.append(f"matrix: duplicate claim_id {cid}")
        seen.add(cid)
        for p in filter(None, (s.strip() for s in r["source_path"].split(";"))):
            if not exists(root, p):
                errs.append(f"matrix {cid}: source_path missing: {p}")
        for rid in filter(None, (s.strip() for s in r["run_id"].split(";"))):
            if rid not in ids:
                errs.append(f"matrix {cid}: run_id not in reports/runs/*.jsonl: {rid}")
    return errs, len(rows)


def self_test():
    """Tiny fixture: one good and one broken reference of each kind must give exactly the expected errors."""
    with tempfile.TemporaryDirectory() as d:
        root = Path(d)
        q = root / "reports/quarto"
        (q / "figures").mkdir(parents=True)
        (q / "tables").mkdir()
        (q / "figures/ok.png").write_bytes(b"")
        (q / "tables/t-a.csv").write_text("x\n")
        (q / "references.bib").write_text("@article{good2020,\n title={x}}\n")
        (q / "c.qmd").write_text(
            "![a](figures/ok.png) ![b](figures/gone.png) `tables/t-{a,b}.csv` `tables/none-*` "
            "[@good2020] @bad2021 mail me@x.org\n")
        (root / "reports/runs").mkdir(parents=True)
        (root / "reports/runs/runs.jsonl").write_text('{"run": "r1"}\n{"scheduler_job": 7}\n')
        m = root / "m.csv"
        m.write_text("claim_id,source_path,run_id\nC1,reports/quarto/tables/t-*.csv,r1;job:7\n"
                     "C2,reports/quarto/tables/zzz.csv,r9\n")
        e1 = check_qmd(q)
        e2, n = check_matrix(m, root, run_ids(root / "reports/runs"))
        assert len(e1) == 3 and any("gone.png" in e for e in e1) and any("none-*" in e for e in e1) \
            and any("@bad2021" in e for e in e1), e1
        assert n == 2 and len(e2) == 2 and "zzz.csv" in e2[0] and "r9" in e2[1], e2
    print("self-test ok")


def main():
    if "--self-test" in sys.argv:
        self_test()
        return 0
    q = ROOT / "reports/quarto"
    errs = check_qmd(q)
    m_errs, n = check_matrix(ROOT / "reports/evidence-matrix.csv", ROOT, run_ids(ROOT / "reports/runs"))
    errs += m_errs
    for e in errs:
        print("FAIL", e)
    print(f"{len(list(q.rglob('*.qmd')))} qmd files, {n} matrix rows, {len(errs)} problems")
    return 1 if errs else 0


if __name__ == "__main__":
    sys.exit(main())
