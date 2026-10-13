"""Dependency direction (docs/architecture.md): the core never imports the data layer, the study drivers or Research OS."""
import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "pravah"
CORE = ["config", "sources", "forward", "inverse", "observations", "metrics", "winds", "indoor", "attribution", "design", "io", "runner"]
ALIASES = ["experiments", "transboundary", "windunc", "belauri", "geodesign", "geodesign_robust"]


def imported(path: Path) -> set[str]:
    out = set()
    for n in ast.walk(ast.parse(path.read_text())):
        if isinstance(n, ast.ImportFrom):
            base = n.module or ""
            out.add(("." * n.level) + base)
            out.update(("." * n.level) + base + "." + a.name for a in n.names)
        elif isinstance(n, ast.Import):
            out.update(a.name for a in n.names)
    return out


def _bad(names: set[str], forbidden: tuple[str, ...]) -> list[str]:
    return sorted(n for n in names if any(n.strip(".").split(".")[0] == f for f in forbidden))


def test_core_does_not_import_geo_studies_or_research():
    for m in CORE:
        assert not _bad(imported(SRC / f"{m}.py"), ("geo", "studies", "research")), m


def test_only_cli_and_aliases_reach_into_studies():
    for p in SRC.glob("*.py"):
        if p.stem in ("cli", *ALIASES):
            continue
        assert not _bad(imported(p), ("studies",) + tuple(ALIASES)), p.name


def test_old_study_paths_are_the_same_module_objects():
    import importlib
    for m in ALIASES[:3]:
        assert importlib.import_module(f"pravah.{m}") is importlib.import_module(f"pravah.studies.toy.{m}")
    for m in ALIASES[3:]:
        assert importlib.import_module(f"pravah.{m}") is importlib.import_module(f"pravah.studies.belauri.{m}")
