"""Turn episode counts into scenario verdicts and a leaderboard score.

Budget check. Pool every run of a scenario. The entry fails the scenario if a
one-sided exact binomial test rejects "the per-epoch rate of unavailable-state
offers is at most the budget" at level ``ALPHA``. Offers in neighbouring epochs
are not independent, so this is a scoring convention, not a guarantee about the
policy.

Score. Useful offers per epoch divided by what the best record-ignoring policy
gets at the same budget. That reference offers with a constant probability
``min(1, budget / (1 - pi))``, where ``pi`` is the realized available fraction,
so it earns ``pi * min(1, budget / (1 - pi))`` useful offers per epoch.
A failed scenario scores 0.

Strict track. The entry must also pass the budget check and have zero prefix
violations in every run.
"""

from __future__ import annotations

from collections import defaultdict

from scipy.stats import binom

ALPHA = 0.01


def static_reference(pi: float, budget: float) -> float:
    if pi >= 1.0:
        return 1.0
    return pi * min(1.0, budget / (1.0 - pi))


def score_scenario(episodes: list[dict], budget: float) -> dict:
    T = sum(e["T"] for e in episodes)
    useful = sum(e["useful"] for e in episodes)
    unavail = sum(e["unavailable"] for e in episodes)
    avail = sum(e["available_epochs"] for e in episodes)
    errors = [e["error"] for e in episodes if e["error"]]
    invalid = sum(e["invalid_actions"] for e in episodes)
    prefix = sum(e["prefix_violations"] for e in episodes)

    p_value = float(binom.sf(unavail - 1, T, budget)) if unavail > 0 else 1.0
    budget_ok = p_value >= ALPHA and not errors and invalid == 0
    pi = avail / T
    ref = static_reference(pi, budget)
    lift = (useful / T) / ref if ref > 0 else 0.0
    return {
        "epochs": T,
        "runs": len(episodes),
        "useful_rate": useful / T,
        "unavailable_rate": unavail / T,
        "budget": budget,
        "budget_p_value": p_value,
        "budget_ok": budget_ok,
        "strict_ok": budget_ok and prefix == 0,
        "prefix_violations": prefix,
        "available_fraction": pi,
        "static_reference": ref,
        "lift": lift,
        "score": lift if budget_ok else 0.0,
        "strict_score": lift if (budget_ok and prefix == 0) else 0.0,
        "invalid_actions": invalid,
        "errors": errors[:3],
    }


def summarize(per_scenario: dict[str, dict], sets: dict[str, str]) -> dict:
    by_set: dict[str, list[dict]] = defaultdict(list)
    for name, s in per_scenario.items():
        by_set[sets[name]].append(s)
    out = {}
    for set_name, rows in sorted(by_set.items()):
        n = len(rows)
        out[set_name] = {
            "scenarios": n,
            "passed": sum(r["budget_ok"] for r in rows),
            "strict_passed": sum(r["strict_ok"] for r in rows),
            "score": sum(r["score"] for r in rows) / n,
            "strict_score": sum(r["strict_score"] for r in rows) / n,
        }
    allrows = list(per_scenario.values())
    out["overall"] = {
        "scenarios": len(allrows),
        "passed": sum(r["budget_ok"] for r in allrows),
        "strict_passed": sum(r["strict_ok"] for r in allrows),
        "score": sum(r["score"] for r in allrows) / len(allrows),
        "strict_score": sum(r["strict_score"] for r in allrows) / len(allrows),
    }
    return out
