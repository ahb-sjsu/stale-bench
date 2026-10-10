"""Command line: ``stale-bench run | audit | leaderboard``."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from . import __version__
from .audit import audit_dir
from .runner import run_scenario
from .scenarios import load_scenarios
from .scoring import score_scenario, summarize

BASELINES = {
    "static": "stale_bench.baselines.static:StaticBudget",
    "trust-record": "stale_bench.baselines.trust_record:TrustRecord",
    "aimd": "stale_bench.baselines.aimd:AIMD",
    "allocation": "stale_bench.baselines.allocation:LearnedAllocation",
    "lagrangian": "stale_bench.baselines.lagrangian:LagrangianReinforce",
    "guarded-allocation": "stale_bench.baselines.allocation:GuardedAllocation",
}


def cmd_run(a: argparse.Namespace) -> int:
    spec = BASELINES.get(a.policy, a.policy)
    scenarios = load_scenarios(a.scenarios)
    if a.only:
        wanted = set(a.only.split(","))
        scenarios = [s for s in scenarios if s.name in wanted or s.set in wanted]
    key = "public"
    if a.hidden:
        key = os.environ.get("STALE_HIDDEN_KEY", "")
        if not key:
            print("--hidden needs STALE_HIDDEN_KEY", file=sys.stderr)
            return 2
    out = {"version": __version__, "policy": a.policy, "hidden": a.hidden, "scenarios": {}}
    for sc in scenarios:
        eps = run_scenario(
            spec,
            sc,
            key=key,
            hidden=a.hidden,
            isolate=a.isolate,
            log_dir=Path(a.log_dir) if a.log_dir else None,
        )
        verdict = score_scenario(eps, sc.budget)
        out["scenarios"][sc.name] = {"set": sc.set, "verdict": verdict, "episodes": eps}
        flag = "ok " if verdict["budget_ok"] else "FAIL"
        strict = "strict-ok" if verdict["strict_ok"] else "strict-fail"
        print(
            f"{flag} {sc.name:<28} lift {verdict['lift']:.3f}  "
            f"unavail {verdict['unavailable_rate']:.4f}/{sc.budget}  {strict}",
            file=sys.stderr,
        )
    out["summary"] = summarize(
        {n: s["verdict"] for n, s in out["scenarios"].items()},
        {n: s["set"] for n, s in out["scenarios"].items()},
    )
    text = json.dumps(out, indent=1)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(text)
    for set_name, s in out["summary"].items():
        print(
            f"{set_name:<8} score {s['score']:.3f} ({s['passed']}/{s['scenarios']} pass)  "
            f"strict {s['strict_score']:.3f} ({s['strict_passed']}/{s['scenarios']})",
            file=sys.stderr,
        )
    return 0


def cmd_audit(a: argparse.Namespace) -> int:
    results = json.loads(Path(a.results).read_text())
    problems = audit_dir(Path(a.log_dir), results)
    for p in problems:
        print(p)
    print(f"{len(problems)} problem(s)", file=sys.stderr)
    return 1 if problems else 0


def cmd_leaderboard(a: argparse.Namespace) -> int:
    rows = []
    for path in a.results:
        r = json.loads(Path(path).read_text())
        rows.append((r["policy"], r["summary"]))
    sets = sorted({k for _, s in rows for k in s if k != "overall"}) + ["overall"]
    print("| policy | " + " | ".join(sets) + " | strict overall |")
    print("|---" * (len(sets) + 2) + "|")
    for policy, s in sorted(rows, key=lambda r: -r[1]["overall"]["score"]):
        cells = [
            f"{s[k]['score']:.3f} ({s[k]['passed']}/{s[k]['scenarios']})" if k in s else "" for k in sets
        ]
        print(f"| {policy} | " + " | ".join(cells) + f" | {s['overall']['strict_score']:.3f} |")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="stale-bench")
    p.add_argument("--version", action="version", version=__version__)
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="score one policy")
    r.add_argument("policy", help="baseline name or file.py:Class")
    r.add_argument("--scenarios", help="scenario JSON (default: built-in dev set)")
    r.add_argument("--only", help="comma-separated scenario or set names")
    r.add_argument("--hidden", action="store_true", help="organizers only: use the secret key")
    r.add_argument("--isolate", action="store_true", help="run the policy in a separate process")
    r.add_argument("--log-dir", help="write audit logs here")
    r.add_argument("--out", help="write results JSON here")
    r.set_defaults(fn=cmd_run)

    au = sub.add_parser("audit", help="recompute results from audit logs")
    au.add_argument("results")
    au.add_argument("log_dir")
    au.set_defaults(fn=cmd_audit)

    lb = sub.add_parser("leaderboard", help="markdown table from results files")
    lb.add_argument("results", nargs="+")
    lb.set_defaults(fn=cmd_leaderboard)

    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
