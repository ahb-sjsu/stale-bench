"""Replay one scenario run against one policy.

The harness, not the policy, draws each randomized action from a seeded
stream, so every decision can be recomputed from the audit log. In
subprocess mode the policy lives in a separate process and receives one
epoch at a time, so it cannot look ahead in the trace.
"""

from __future__ import annotations

import gzip
import importlib
import importlib.util
import json
import math
import multiprocessing as mp
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from .scenarios import Scenario, seed_for
from .types import Label, Ledger, Record

CALL_TIMEOUT_S = 5.0


@dataclass
class EpisodeStats:
    scenario: str
    run: int
    T: int
    offers: int = 0
    useful: int = 0
    unavailable: int = 0
    available_epochs: int = 0
    prefix_violations: int = 0
    first_violation_epoch: int = -1
    invalid_actions: int = 0
    unresolved_at_end: int = 0
    error: str = ""


def load_policy(spec: str):
    """``path/to/file.py:Class`` or ``package.module:Class``."""
    target, _, cls = spec.rpartition(":")
    if not target or not cls:
        raise ValueError(f"policy spec must look like file.py:Class, got {spec!r}")
    if target.endswith(".py"):
        mspec = importlib.util.spec_from_file_location(Path(target).stem, target)
        module = importlib.util.module_from_spec(mspec)
        mspec.loader.exec_module(module)
    else:
        module = importlib.import_module(target)
    return getattr(module, cls)


def _child(conn, spec: str) -> None:
    cls = load_policy(spec)
    policy = None
    while True:
        op, payload = conn.recv()
        if op == "init":
            policy = cls(**payload)
            conn.send(None)
        elif op == "act":
            try:
                conn.send(float(policy.act(*payload)))
            except Exception as e:  # report, do not crash the harness
                conn.send(f"error: {type(e).__name__}: {e}")
        else:
            return


class SubprocessPolicy:
    """Runs the entrant's policy in its own process."""

    def __init__(self, spec: str):
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_child, args=(child, spec), daemon=True)
        self._proc.start()

    def init(self, **kwargs) -> None:
        self._call("init", kwargs)

    def act(self, record, labels, ledger):
        return self._call("act", (record, labels, ledger))

    def _call(self, op, payload):
        self._conn.send((op, payload))
        if not self._conn.poll(CALL_TIMEOUT_S):
            self.close()
            raise TimeoutError(f"policy did not answer {op} within {CALL_TIMEOUT_S}s")
        out = self._conn.recv()
        if isinstance(out, str):
            raise RuntimeError(out)
        return out

    def close(self) -> None:
        if self._proc.is_alive():
            try:
                self._conn.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
            self._proc.join(timeout=2)
            if self._proc.is_alive():
                self._proc.kill()


def make_policy(spec: str, sc: Scenario, isolate: bool):
    kwargs = {"budget": sc.budget, "meta": sc.public_meta()}
    if isolate:
        p = SubprocessPolicy(spec)
        p.init(**kwargs)
        return p
    return load_policy(spec)(**kwargs)


def run_episode(
    spec: str,
    sc: Scenario,
    run: int,
    key: str = "public",
    hidden: bool = False,
    isolate: bool = False,
    log_dir: Path | None = None,
) -> EpisodeStats:
    x = sc.states(run, key, hidden)
    T = len(x)
    ages = sc.ages(T, seed_for(key, sc.name, run, "age"))
    delays = sc.label_delays(T, seed_for(key, sc.name, run, "label"))
    draws = seed_for(key, sc.name, run, "draw").random(T)
    recv = seed_for(key, sc.name, run, "receive").random(T)

    st = EpisodeStats(scenario=sc.name, run=run, T=T, available_epochs=int(x.sum()))
    pending: dict[int, list[Label]] = defaultdict(list)
    resolved_unavail = 0
    unresolved = 0
    last_offer = 0
    log = None
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        log = gzip.open(log_dir / f"{sc.name}__{run}.jsonl.gz", "wt")
        header = {"scenario": sc.name, "run": run, "T": T, "budget": sc.budget, "burst": sc.burst}
        log.write(json.dumps(header) + "\n")

    policy = make_policy(spec, sc, isolate)
    try:
        for t in range(T):
            arrived = pending.pop(t, [])
            for lab in arrived:
                unresolved -= 1
                resolved_unavail += lab.state == 0
            a_t = int(ages[t])
            t_source = (t - a_t) * sc.dt
            t_decide = t * sc.dt
            rec = Record(
                epoch=t,
                unit="u0",
                sampled_state=int(x[t - a_t]),
                age=a_t,
                t_source=t_source,
                t_receive=t_source + float(recv[t]) * (t_decide - t_source),
                t_decide=t_decide,
            )
            allowance = sc.budget * (t + 1) + sc.burst
            charged = resolved_unavail + unresolved
            ledger = Ledger(charged, allowance, resolved_unavail, unresolved, last_offer)

            raw = policy.act(rec, arrived, ledger)
            try:
                q = float(raw)
            except (TypeError, ValueError):
                q = math.nan
            if not 0.0 <= q <= 1.0:  # also catches NaN
                st.invalid_actions += 1
                q = 0.0
            offer = bool(draws[t] < q)
            last_offer = int(offer)

            if offer:
                st.offers += 1
                if x[t]:
                    st.useful += 1
                else:
                    st.unavailable += 1
                unresolved += 1
                L = int(delays[t])
                if t + L < T:
                    pending[t + L].append(Label(offer_epoch=t, state=int(x[t]), arrive_epoch=t + L))
                if resolved_unavail + unresolved > allowance:
                    st.prefix_violations += 1
                    if st.first_violation_epoch < 0:
                        st.first_violation_epoch = t
            if log is not None:
                row = [t, int(x[t]), rec.sampled_state, a_t, int(delays[t]), q, float(draws[t]), int(offer)]
                log.write(json.dumps(row) + "\n")
    except Exception as e:
        st.error = f"{type(e).__name__}: {e}"
    finally:
        if isolate:
            policy.close()
        if log is not None:
            log.close()
    st.unresolved_at_end = unresolved
    return st


def run_scenario(spec: str, sc: Scenario, **kw) -> list[dict]:
    hidden = kw.get("hidden", False)
    return [asdict(run_episode(spec, sc, r, **kw)) for r in range(sc.n_runs(hidden))]
