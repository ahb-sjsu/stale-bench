"""Scenario definitions and seeded random streams.

A scenario fixes the state process, how old each record is, how late labels
arrive, the budget, and how many runs to score. Public runs use seeds derived
from the scenario name. Hidden runs mix in a secret key held by the organizers,
so no one can regenerate the hidden sequences.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

import numpy as np

from . import generators

STREAMS = ("state", "age", "label", "draw", "receive")


def seed_for(key: str, name: str, run: int, stream: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{key}|{name}|{run}|{stream}".encode()).digest()
    return np.random.default_rng(np.random.SeedSequence(int.from_bytes(digest[:16], "little")))


def _sample_delay(spec: dict, T: int, rng: np.random.Generator, minimum: int) -> np.ndarray:
    kind = spec["kind"]
    if kind == "fixed":
        out = np.full(T, spec["value"], dtype=np.int64)
    elif kind == "uniform":
        out = rng.integers(spec["low"], spec["high"] + 1, size=T)
    elif kind == "geometric":
        # base + Geometric(p) - 1, so the minimum is ``base``
        out = spec["base"] + rng.geometric(spec["p"], size=T) - 1
    else:
        raise ValueError(f"unknown delay kind {kind!r}")
    return np.maximum(out, minimum).astype(np.int64)


def data_root(hidden: bool) -> Path:
    if hidden:
        root = os.environ.get("STALE_PRIVATE_DATA")
        if not root:
            raise RuntimeError("hidden trace scenarios need STALE_PRIVATE_DATA")
        return Path(root)
    return Path(str(resources.files("stale_bench") / "data"))


@dataclass
class Scenario:
    name: str
    set: str
    generator: dict
    age: dict
    label_delay: dict
    budget: float
    burst: float
    dt: float
    T: int = 0
    runs: int = 1
    notes: str = ""

    def trace_files(self, hidden: bool) -> list[Path]:
        root = data_root(hidden) / self.generator["dir"]
        return sorted(root.glob("*.csv"))

    def n_runs(self, hidden: bool) -> int:
        if self.generator["kind"] == "trace":
            return len(self.trace_files(hidden))
        return self.runs

    def states(self, run: int, key: str, hidden: bool) -> np.ndarray:
        g = dict(self.generator)
        kind = g.pop("kind")
        if kind == "trace":
            return generators.load_trace(self.trace_files(hidden)[run])
        rng = seed_for(key, self.name, run, "state")
        return generators.GENERATORS[kind](self.T, rng, **g)

    def ages(self, T: int, rng: np.random.Generator) -> np.ndarray:
        a = _sample_delay(self.age, T, rng, minimum=0)
        return np.minimum(a, np.arange(T))

    def label_delays(self, T: int, rng: np.random.Generator) -> np.ndarray:
        return _sample_delay(self.label_delay, T, rng, minimum=1)

    def public_meta(self) -> dict:
        """What a policy is told. The state process parameters are withheld."""
        return {
            "scenario": self.name,
            "set": self.set,
            "budget": self.budget,
            "burst": self.burst,
            "dt": self.dt,
        }


def load_scenarios(path: str | Path | None = None) -> list[Scenario]:
    if path is None:
        text = (resources.files("stale_bench") / "scenarios" / "dev.json").read_text()
    else:
        text = Path(path).read_text()
    return [Scenario(**s) for s in json.loads(text)["scenarios"]]
