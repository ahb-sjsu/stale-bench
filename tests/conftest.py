from dataclasses import replace

import pytest

from stale_bench.scenarios import load_scenarios


@pytest.fixture(scope="session")
def scenarios():
    return {s.name: s for s in load_scenarios()}


def short(sc, T=3000, runs=2):
    return sc if sc.generator["kind"] == "trace" else replace(sc, T=T, runs=runs)
