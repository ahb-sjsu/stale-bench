"""Recompute every episode count from the audit logs alone.

Each log line is ``[t, state, sampled_state, age, label_delay, q, draw, offer]``.
The auditor checks that each record matches the logged state ``age`` epochs
earlier, that each offer equals ``draw < q``, and then replays the label
schedule and the prefix ledger. It compares its counts with a results file
and reports every mismatch.
"""

from __future__ import annotations

import gzip
import json
from collections import defaultdict
from pathlib import Path

FIELDS = ("offers", "useful", "unavailable", "available_epochs", "prefix_violations", "unresolved_at_end")


def audit_log(path: Path) -> tuple[dict, dict, list[str]]:
    problems: list[str] = []
    with gzip.open(path, "rt") as f:
        header = json.loads(f.readline())
        rows = [json.loads(line) for line in f]
    T, budget, burst = header["T"], header["budget"], header["burst"]
    if len(rows) != T:
        problems.append(f"expected {T} epochs, found {len(rows)}")
    states = [r[1] for r in rows]
    c = dict.fromkeys(FIELDS, 0)
    c["available_epochs"] = sum(states)
    arrivals: dict[int, list[int]] = defaultdict(list)
    resolved_unavail = unresolved = 0
    for t, x, b, age, L, q, u, offer in rows:
        for s in arrivals.pop(t, []):
            unresolved -= 1
            resolved_unavail += states[s] == 0
        if not 0 <= age <= t:
            problems.append(f"epoch {t}: age {age} out of range")
        elif b != states[t - age]:
            problems.append(f"epoch {t}: record bit {b} != state at epoch {t - age}")
        if not 0.0 <= q <= 1.0:
            problems.append(f"epoch {t}: q={q} out of range")
        if L < 1:
            problems.append(f"epoch {t}: label delay {L} < 1")
        if int(u < q) != offer:
            problems.append(f"epoch {t}: offer {offer} != (draw < q)")
        if offer:
            c["offers"] += 1
            c["useful" if x else "unavailable"] += 1
            unresolved += 1
            if t + L < T:
                arrivals[t + L].append(t)
            if resolved_unavail + unresolved > budget * (t + 1) + burst:
                c["prefix_violations"] += 1
    c["unresolved_at_end"] = unresolved
    return header, c, problems


def audit_dir(log_dir: Path, results: dict) -> list[str]:
    """Compare every log in ``log_dir`` with the episodes in ``results``."""
    expected = {}
    for name, sc in results["scenarios"].items():
        for ep in sc["episodes"]:
            expected[(name, ep["run"])] = ep
    problems = []
    seen = set()
    for path in sorted(Path(log_dir).glob("*.jsonl.gz")):
        header, counts, probs = audit_log(path)
        key = (header["scenario"], header["run"])
        seen.add(key)
        problems += [f"{path.name}: {p}" for p in probs[:10]]
        ep = expected.get(key)
        if ep is None:
            problems.append(f"{path.name}: no matching episode in results")
            continue
        for k in FIELDS:
            if counts[k] != ep[k]:
                problems.append(f"{path.name}: {k} log={counts[k]} results={ep[k]}")
    for key in expected.keys() - seen:
        problems.append(f"missing log for {key[0]} run {key[1]}")
    return problems
