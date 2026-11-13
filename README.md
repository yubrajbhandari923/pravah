# PRAVAH

Sparse-sensor air quality reconstruction under uncertain meteorology and emissions.
Currently a synthetic 2-D depth-averaged passive-tracer testbed (FiPy); no real PM2.5 or exposure claims.

**Report:** https://yubrajbhandari923.github.io/pravah/ (Quarto book, rendered from `reports/quarto/`).

Start with `state/STATUS.md`, then `settings.yaml` and the relevant part of `plans/`.

| Path | What |
|---|---|
| `AGENTS.md` | How agents work here (generic operating philosophy) |
| `settings.yaml` | Project paths, environment, HPC defaults |
| `plans/` | Scientific intent: plan-01 (E00–E03), plan-02 (E04–E11), plan-03 (E12–E20) |
| `state/` | Current status, per-experiment journals, decisions, history |
| `src/pravah/` | Forward solver, benchmark, observations, inverse, sensitivity, information, temporal basis, placement |
| `scripts/` | Experiment drivers `eNN_*.py` (write to `reports/quarto/tables` and `figures`) |
| `tests/` | `uv run pytest -q` |
| `configs/` | `benchmark-v1.yaml` (draft) |
| `docs/research/` | Preregistered protocol (draft), derivations, gate audits, literature notes |
| `reports/quarto/` | Quarto book, result tables and figures |
| `reports/runs/` | Run manifests |

```bash
uv sync
uv run pytest -q
uv run python scripts/e04_footprints.py
```

Migrated from a ResearchOS layout on 2026-11-12; the old Git history lives under `refs/legacy/`.
