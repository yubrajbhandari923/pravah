"""Test-only forward solvers for tests/test_solver_routing.py (imported as ``tests.helpers_solver``, also through
``--solver tests.helpers_solver:make_recording`` on the CLI).  INTERFACE TESTS, NOT PHYSICS BENCHMARKS.

They delegate to the original FiPy ``simulate`` function object captured at import time, so a test can patch
``pravah.forward.simulate`` (and thereby FIPY) to raise and still detect any hidden FiPy call."""
from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from pravah.config import FieldResult, Scenario, SolverOptions, to_dict
from pravah.forward import BackendInfo
from pravah.forward import simulate as _REAL_SIMULATE

CALLS: list[tuple[str, str]] = []  # (role, scenario fingerprint) of every call of every instance, in order


def fingerprint(scenario: Scenario) -> str:
    return hashlib.sha256(json.dumps(to_dict(scenario), sort_keys=True).encode()).hexdigest()[:12]


class RecordingForward:
    """Delegates to the real FiPy solver, records (role, scenario fingerprint, q).  ``offset`` adds a constant
    concentration (a DistinguishableForward: any hidden FiPy call would lack it); ``linear`` is the declared capability."""

    def __init__(self, role: str = "rec", offset: float = 0.0, linear: bool = True) -> None:
        self.role, self.offset, self.linear = role, float(offset), bool(linear)
        self.calls: list[tuple[str, str, tuple[float, ...]]] = []

    @property
    def info(self) -> BackendInfo:
        return BackendInfo(f"recording-{self.role}", "test", {"role": self.role, "offset": self.offset},
                           {"linear": self.linear, "budget": True, "boundary_bases": False})

    def __call__(self, scenario: Scenario, q: Any, solver: SolverOptions = SolverOptions()) -> FieldResult:
        fp = fingerprint(scenario)
        self.calls.append((self.role, fp, tuple(float(v) for v in np.asarray(q))))
        CALLS.append((self.role, fp))
        res = _REAL_SIMULATE(scenario, q, solver)
        if self.offset:
            res.concentration.values = res.concentration.values + self.offset
        return res


def make_recording(**kwargs: Any) -> RecordingForward:
    """Factory for ``get_forward('tests.helpers_solver:make_recording', kwargs)``."""
    return RecordingForward(**kwargs)
