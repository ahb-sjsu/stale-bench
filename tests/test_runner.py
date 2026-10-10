import json
from pathlib import Path

import pytest
from conftest import short

from stale_bench.audit import audit_dir
from stale_bench.cli import BASELINES
from stale_bench.runner import run_episode, run_scenario
from stale_bench.scoring import score_scenario, static_reference

HERE = Path(__file__).parent


@pytest.mark.parametrize("name", sorted(BASELINES))
def test_audit_reproduces_every_baseline(tmp_path, scenarios, name):
    results = {"scenarios": {}}
    for sc in scenarios.values():
        sc = short(sc, T=1500, runs=1)
        eps = run_scenario(BASELINES[name], sc, log_dir=tmp_path)
        assert all(not e["error"] for e in eps), eps
        results["scenarios"][sc.name] = {"episodes": eps}
    assert audit_dir(tmp_path, results) == []


def test_audit_catches_tampering(tmp_path, scenarios):
    import gzip

    sc = short(scenarios["s1-mid-age2"], T=500, runs=1)
    eps = run_scenario(BASELINES["static"], sc, log_dir=tmp_path)
    path = next(tmp_path.glob("*.jsonl.gz"))
    lines = gzip.open(path, "rt").read().splitlines()
    row = json.loads(lines[100])
    row[7] = 1 - row[7]  # flip one offer
    lines[100] = json.dumps(row)
    with gzip.open(path, "wt") as f:
        f.write("\n".join(lines) + "\n")
    problems = audit_dir(tmp_path, {"scenarios": {sc.name: {"episodes": eps}}})
    assert any("draw < q" in p for p in problems)


def test_policy_sees_consistent_records_and_labels(scenarios):
    sc = short(scenarios["s2-late-labels"], T=2000, runs=1)
    st = run_episode(f"{HERE / 'spy_policy.py'}:Spy", sc, 0)
    assert st.error == "", st.error


def test_isolated_matches_in_process(scenarios):
    sc = short(scenarios["s1-mid-age-jitter"], T=400, runs=1)
    a = run_scenario(BASELINES["allocation"], sc)
    b = run_scenario(BASELINES["allocation"], sc, isolate=True)
    assert a == b


def test_hidden_key_changes_states(scenarios):
    sc = short(scenarios["s1-mid-age2"], T=1000, runs=1)
    assert (sc.states(0, "public", False) != sc.states(0, "secret", False)).any()
    assert (sc.states(0, "public", False) == sc.states(0, "public", False)).all()


def test_guarded_allocation_never_breaks_a_prefix(scenarios):
    for sc in scenarios.values():
        for e in run_scenario(BASELINES["guarded-allocation"], short(sc)):
            assert e["prefix_violations"] == 0, (sc.name, e)


def test_trust_record_fails_when_records_are_stale(scenarios):
    sc = short(scenarios["s1-fast-age2"], T=5000)
    v = score_scenario(run_scenario(BASELINES["trust-record"], sc), sc.budget)
    assert not v["budget_ok"]


def test_static_passes_s1(scenarios):
    for sc in scenarios.values():
        if sc.set == "S1":
            sc = short(sc, T=5000)
            v = score_scenario(run_scenario(BASELINES["static"], sc), sc.budget)
            assert v["budget_ok"], (sc.name, v)


def test_invalid_actions_fail_the_scenario(scenarios):
    sc = short(scenarios["s1-mid-age2"], T=300, runs=1)
    eps = run_scenario(f"{HERE / 'spy_policy.py'}:Broken", sc)
    v = score_scenario(eps, sc.budget)
    assert eps[0]["invalid_actions"] == 300 and not v["budget_ok"] and v["score"] == 0


def test_static_reference():
    assert static_reference(0.5, 0.05) == pytest.approx(0.05)
    assert static_reference(0.25, 0.05) == pytest.approx(0.25 * 0.05 / 0.75)
    assert static_reference(0.9, 0.5) == pytest.approx(0.9)
