"""Ground-truth state sequences for one binary schedulability unit.

Every generator returns a ``uint8`` array of length ``T`` with 1 = available.
The state is exogenous. Nothing a policy does changes it, which is what makes
replaying one sequence against many policies fair.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

BURN_IN = 1000


def markov_sym(T: int, rng: np.random.Generator, p: float) -> np.ndarray:
    """Symmetric two-state chain, flip probability ``p`` per epoch."""
    flips = rng.random(T) < p
    x0 = int(rng.random() < 0.5)
    return ((x0 + np.cumsum(flips)) % 2).astype(np.uint8)


def _alternating(T: int, rng: np.random.Generator, draw_on, draw_off, pi_on: float) -> np.ndarray:
    """Alternate on/off runs with the given holding-time samplers, after burn-in."""
    total = T + BURN_IN
    out = np.empty(total, dtype=np.uint8)
    state = int(rng.random() < pi_on)
    i = 0
    while i < total:
        n = int(draw_on() if state else draw_off())
        n = max(n, 1)
        out[i : i + n] = state
        i += n
        state ^= 1
    return out[BURN_IN:]


def markov_asym(T: int, rng: np.random.Generator, p_on: float, p_off: float) -> np.ndarray:
    """Two-state chain with P(0->1) = ``p_on`` and P(1->0) = ``p_off``."""
    return _alternating(
        T,
        rng,
        lambda: rng.geometric(p_off),
        lambda: rng.geometric(p_on),
        p_on / (p_on + p_off),
    )


def semi_markov(
    T: int, rng: np.random.Generator, mean_on: float, mean_off: float, sigma: float
) -> np.ndarray:
    """Lognormal holding times with the given means (epochs) and log-scale ``sigma``."""

    def lognormal(mean: float):
        mu = np.log(mean) - sigma**2 / 2
        return lambda: np.ceil(rng.lognormal(mu, sigma))

    return _alternating(T, rng, lognormal(mean_on), lognormal(mean_off), mean_on / (mean_on + mean_off))


def regime(
    T: int, rng: np.random.Generator, p_slow: float, p_fast: float, p_switch: float
) -> np.ndarray:
    """Symmetric flips whose rate switches between a slow and a fast hidden regime."""
    total = T + BURN_IN
    z = markov_sym(total, rng, p_switch)
    p = np.where(z == 1, p_fast, p_slow)
    flips = rng.random(total) < p
    x0 = int(rng.random() < 0.5)
    return ((x0 + np.cumsum(flips)) % 2).astype(np.uint8)[BURN_IN:]


def load_trace(path: Path) -> np.ndarray:
    """Read a two-column ``epoch,state`` CSV."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return np.array([int(r["state"]) for r in rows], dtype=np.uint8)


GENERATORS = {
    "markov_sym": markov_sym,
    "markov_asym": markov_asym,
    "semi_markov": semi_markov,
    "regime": regime,
}
