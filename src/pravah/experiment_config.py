"""Typed experiment configuration and provenance writer (stdlib only; runs on HPC nodes too).

Mirrors docs/research/preregistered-protocol.md. Changing a default here is a scientific
amendment, not a refactor. Truth and inference stay separate: this module holds no truth data.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

SCHEMA_VERSION = 1
UNITS = {"length": "m", "time": "s", "concentration": "ug m^-3",
         "emission_rate": "ug s^-1", "areal_flux": "ug m^-2 s^-1"}


class ConfigError(ValueError):
    """Invalid or inconsistent configuration/provenance input."""


class LeakError(ConfigError):
    """A held-out (test / held-out wind) quantity was used for tuning, placement or selection."""


def _req(cond, msg):
    if not cond:
        raise ConfigError(msg)


@dataclass
class ModelConfig:
    domain_m: tuple = (10000.0, 10000.0)
    grid: tuple = (64, 64)
    dt_s: int = 300  # solver step; observations are hourly (3600 // dt_s steps). vanleer needs Courant <= 1 (A2)
    advection_scheme: str = "upwind"  # amendment A2: 64^2 runs use "vanleer" with dt_s = 30 (state/decisions/2026-11-12-advection-scheme.md)
    diffusivity_m2_s: float = 50.0
    removal_per_s: float = 1 / (12 * 3600)
    mixing_depth_m: float = 500.0
    spinup_h: int = 48
    units: dict = field(default_factory=lambda: dict(UNITS))

    def validate(self):
        _req(all(x > 0 for x in self.domain_m), "domain must be positive")
        _req(len(self.grid) == 2 and all(isinstance(n, int) and n > 1 for n in self.grid), "bad grid")
        _req(self.dt_s > 0 and 3600 % self.dt_s == 0, "dt_s must divide one hour")
        _req(self.advection_scheme in ("upwind", "vanleer"), "advection_scheme must be upwind or vanleer")
        _req(self.diffusivity_m2_s > 0, "diffusivity must be > 0")
        _req(self.removal_per_s >= 0, "removal must be >= 0")
        _req(self.mixing_depth_m > 0, "mixing depth must be > 0")
        _req(self.spinup_h >= 0, "spinup must be >= 0")


@dataclass
class SourceConfig:
    footprint_sigma_m: float = 400.0
    reference_rate_ug_s: float = 1e6  # amendment A1 (state/decisions/2026-11-12-source-scale.md); was 100
    strength_loguniform: tuple = (0.5, 5.0)
    lattice: tuple = (12, 12)
    min_separation_m: float = 1500.0
    background_ug_m3: float = 5.0

    def validate(self):
        lo, hi = self.strength_loguniform
        _req(0 < lo < hi, "strength range must satisfy 0 < lo < hi")
        _req(self.footprint_sigma_m > 0 and self.reference_rate_ug_s > 0, "sigma/a_ref must be > 0")
        _req(self.background_ug_m3 >= 0, "background must be >= 0")


@dataclass
class Design:
    n_sources: tuple = (2, 4, 8)
    sensor_budgets: tuple = (2, 4, 8, 16, 32)
    windows: tuple = ("snapshot", "6h", "24h", "7d")
    noise: tuple = ("white", "ar1")
    min_sensor_separation_m: float = 1250.0

    def validate(self):
        _req(self.n_sources and all(n > 0 for n in self.n_sources), "n_sources must be positive")
        _req(self.sensor_budgets and all(n > 0 for n in self.sensor_budgets), "budgets must be positive")


@dataclass
class Seeds:
    n_replicates: int = 20
    truth_base: int = 1000
    noise_base: int = 2000
    placement_base: int = 3000
    bootstrap: int = 4000

    def streams(self, replicate: int) -> dict:
        _req(0 <= replicate < self.n_replicates, f"replicate {replicate} outside 0..{self.n_replicates - 1}")
        return {"truth": self.truth_base + replicate, "noise": self.noise_base + replicate,
                "placement": self.placement_base + replicate}

    def validate(self):
        _req(self.n_replicates > 0, "n_replicates must be > 0")
        bases = [self.truth_base, self.noise_base, self.placement_base]
        _req(all(b2 - b1 >= self.n_replicates for b1, b2 in zip(sorted(bases), sorted(bases)[1:])),
             "seed streams overlap")


@dataclass
class Split:
    """Inclusive 1-based day ranges after spin-up."""
    train: tuple = (1, 30)
    validation: tuple = (31, 42)
    test: tuple = (43, 60)

    def validate(self):
        segs = (self.train, self.validation, self.test)
        _req(all(len(s) == 2 and 1 <= s[0] <= s[1] for s in segs), "bad segment")
        _req(self.train[0] == 1, "train must start at day 1")
        for a, b in zip(segs, segs[1:]):
            _req(b[0] == a[1] + 1, "segments must be contiguous, ordered and disjoint")

    def segment_of(self, day: int) -> str:
        for name in ("train", "validation", "test"):
            lo, hi = getattr(self, name)
            if lo <= day <= hi:
                return name
        raise ConfigError(f"day {day} outside the calendar")


@dataclass
class Winds:
    train: list = field(default_factory=lambda: ["steady_westerly", "veering", "weak_variable"])
    heldout: list = field(default_factory=lambda: ["steady_southerly", "stagnation_then_front", "diurnal_rotation"])

    def validate(self):
        _req(not set(self.train) & set(self.heldout), "held-out wind regimes must be disjoint from training")


@dataclass
class MetricDef:
    name: str
    unit: str
    definition: str
    primary: bool = False


def _default_metrics():
    return [
        MetricDef("forecast_rmse_1h_preassim", "ug m^-3",
                  "RMSE vs noise-free truth of the 1 h field forecast issued before assimilating the new "
                  "observation; hourly issue times over test days 43-60 minus 6 h warm-up (426 issues)", True),
        MetricDef("source_rmse_rel", "1", "sqrt(mean_k (a_hat-a)^2)/a_ref; undefined estimate counts as failure"),
        MetricDef("analysis_rmse", "ug m^-3", "RMSE of the post-assimilation analysis field vs truth, test period"),
        MetricDef("interval_coverage", "1", "coverage of nominal 50/80/95% intervals at held-out cells; plus CRPS"),
        MetricDef("popweighted_rmse", "ug m^-3", "weighted field RMSE with a synthetic population density"),
    ]


@dataclass
class ResourceProfile:
    name: str
    backend: str  # "local" | "slurm"
    cpus: int
    mem_gb: float
    walltime_h: float
    gpus: int = 0
    # HPC jobs return results only if written under $RESEARCH_RUN_DIR and registered in artifacts.jsonl
    registers_via: str = "RESEARCH_RUN_DIR"

    def validate(self):
        _req(self.backend in ("local", "slurm"), "backend must be local or slurm")
        _req(self.cpus >= 1 and self.mem_gb > 0 and self.walltime_h > 0 and self.gpus >= 0,
             f"profile {self.name}: cpus/mem/walltime must be positive")

    def reservation_cpu_hours(self) -> float:
        return self.cpus * self.walltime_h


def _default_resources():
    # Protocol: local-only until E01 verifies; the hpc profile is a declared ceiling, not a request.
    return {"local": ResourceProfile("local", "local", cpus=4, mem_gb=8, walltime_h=2),
            "hpc": ResourceProfile("hpc", "slurm", cpus=8, mem_gb=16, walltime_h=6)}


_NESTED = {"model": ModelConfig, "sources": SourceConfig, "design": Design, "seeds": Seeds,
           "split": Split, "winds": Winds}


@dataclass
class ExperimentConfig:
    experiment_id: str
    model: ModelConfig = field(default_factory=ModelConfig)
    sources: SourceConfig = field(default_factory=SourceConfig)
    design: Design = field(default_factory=Design)
    seeds: Seeds = field(default_factory=Seeds)
    split: Split = field(default_factory=Split)
    winds: Winds = field(default_factory=Winds)
    metrics: list = field(default_factory=_default_metrics)
    resources: dict = field(default_factory=_default_resources)
    schema_version: int = SCHEMA_VERSION

    @classmethod
    def default(cls, experiment_id: str) -> "ExperimentConfig":
        c = cls(experiment_id)
        c.validate()
        return c

    def validate(self):
        _req(self.experiment_id, "experiment_id required")
        _req(self.schema_version == SCHEMA_VERSION, "unsupported schema_version")
        for part in (self.model, self.sources, self.design, self.seeds, self.split, self.winds):
            part.validate()
        _req(sum(m.primary for m in self.metrics) == 1, "exactly one primary metric is required")
        _req(all(m.name and m.unit and m.definition for m in self.metrics), "metrics need name, unit, definition")
        _req(len({m.name for m in self.metrics}) == len(self.metrics), "duplicate metric names")
        _req(self.resources, "at least one resource profile")
        for p in self.resources.values():
            p.validate()

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, indent=2)

    def config_hash(self) -> str:
        return hashlib.sha256(json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d)
        unknown = set(d) - {f.name for f in fields(cls)}
        _req(not unknown, f"unknown config keys: {sorted(unknown)}")
        _req("experiment_id" in d, "experiment_id required")
        kw = {k: _build(_NESTED[k], d.pop(k)) for k in list(d) if k in _NESTED}
        if "metrics" in d:
            kw["metrics"] = [_build(MetricDef, m) for m in d.pop("metrics")]
        if "resources" in d:
            kw["resources"] = {k: _build(ResourceProfile, v) for k, v in d.pop("resources").items()}
        c = cls(**d, **kw)
        c.validate()
        return c


def _build(klass, d: dict):
    """Build a flat dataclass from a dict; reject unknown keys; JSON lists -> tuples where the default is a tuple."""
    unknown = set(d) - {f.name for f in fields(klass)}
    _req(not unknown, f"unknown keys for {klass.__name__}: {sorted(unknown)}")
    try:
        proto = klass()
    except TypeError:  # required fields (MetricDef, ResourceProfile)
        proto = None
    kw = {}
    for k, v in d.items():
        if isinstance(v, list) and proto is not None and isinstance(getattr(proto, k), tuple):
            v = tuple(v)
        kw[k] = v
    return klass(**kw)


def dump_config(cfg: ExperimentConfig, path) -> None:
    Path(path).write_text(cfg.to_json() + "\n")


def load_config(path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(json.loads(Path(path).read_text()))


def truth_id(cfg: ExperimentConfig, replicate: int, n_sources: int) -> str:
    """Deterministic ID of a simulated truth: depends only on truth-side config, the seed and K_s."""
    _req(n_sources in cfg.design.n_sources, f"n_sources {n_sources} not in design {cfg.design.n_sources}")
    key = {"model": asdict(cfg.model), "sources": asdict(cfg.sources), "winds": asdict(cfg.winds),
           "split": asdict(cfg.split), "truth_seed": cfg.seeds.streams(replicate)["truth"], "K": n_sources}
    return "TRUTH-" + hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


_FITTING = ("tuning", "placement", "model_selection")


def check_no_leak(split: Split, purpose: str, days, wind_regimes=(), winds: Winds | None = None) -> None:
    """Raise LeakError if test days feed fitting purposes, if evaluation touches non-test days,
    or if held-out wind regimes feed fitting. `winds` defaults to the protocol's held-out set."""
    _req(purpose in _FITTING + ("evaluation",), f"unknown purpose {purpose!r}")
    segs = {split.segment_of(d) for d in days}
    if purpose in _FITTING and "test" in segs:
        raise LeakError(f"test-period days used for {purpose}")
    if purpose == "evaluation" and segs != {"test"}:
        raise LeakError("headline evaluation must use test days only")
    heldout = set((winds or Winds()).heldout)
    if purpose in _FITTING and heldout & set(wind_regimes):
        raise LeakError(f"held-out wind regime used for {purpose}: {sorted(heldout & set(wind_regimes))}")


@dataclass
class DataManifest:
    entries: list

    @staticmethod
    def _sha(p: Path) -> str:
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    @classmethod
    def build(cls, paths, root) -> "DataManifest":
        root = Path(root)
        out = []
        for p in map(Path, paths):
            _req(p.is_file(), f"manifest file missing: {p}")
            out.append({"path": str(p.resolve().relative_to(root.resolve())), "sha256": cls._sha(p),
                        "bytes": p.stat().st_size})
        return cls(out)

    def verify(self, root) -> list:
        """Relative paths that are missing or whose hash changed."""
        bad = []
        for e in self.entries:
            p = Path(root) / e["path"]
            if not p.is_file() or self._sha(p) != e["sha256"]:
                bad.append(e["path"])
        return bad


def _git(*args) -> str:
    try:
        return subprocess.run(["git", *args], capture_output=True, text=True, timeout=10, check=True).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _append(path: Path, rec: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")


def register_artifact(run_dir, path) -> None:
    """Register an existing file so it comes home from the cluster (artifacts.jsonl, absolute path)."""
    p = Path(path)
    _req(p.is_file(), f"cannot register missing artifact: {p}")
    _append(Path(run_dir) / "artifacts.jsonl", {"path": str(p.resolve())})


def log_metric(run_dir, name: str, value: float, step: int = 0) -> None:
    _req(isinstance(value, (int, float)) and math.isfinite(value), f"metric {name} must be finite")
    _append(Path(run_dir) / "metrics.jsonl", {"name": name, "value": value, "step": step})


def write_provenance(run_dir, cfg: ExperimentConfig, command, replicate: int, n_sources: int,
                     run_id: str, manifest: DataManifest) -> dict:
    """Write outputs/provenance.json under run_dir and register it. Returns the record."""
    cfg.validate()
    run_dir = Path(run_dir)
    prov = {
        "run_id": run_id, "experiment_id": cfg.experiment_id, "command": list(command),
        "git_sha": _git("rev-parse", "HEAD") or "unknown",
        "git_dirty": bool(_git("status", "--porcelain")),
        "seeds": cfg.seeds.streams(replicate), "replicate": replicate, "n_sources": n_sources,
        "truth_id": truth_id(cfg, replicate, n_sources),
        "config_hash": cfg.config_hash(), "config": cfg.to_dict(), "units": cfg.model.units,
        "metrics": [asdict(m) for m in cfg.metrics],
        "manifest": asdict(manifest),
        "machine": {"node": platform.node(), "platform": platform.platform(), "cpus": os.cpu_count()},
        "python": sys.version.split()[0],
    }
    out = run_dir / "outputs" / "provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(prov, sort_keys=True, indent=2))
    register_artifact(run_dir, out)
    return json.loads(out.read_text())
