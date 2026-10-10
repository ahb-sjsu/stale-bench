import numpy as np

from stale_bench import generators as g


def rng(s=0):
    return np.random.default_rng(s)


def test_markov_sym_flip_rate():
    x = g.markov_sym(200_000, rng(), p=0.05)
    assert x.dtype == np.uint8 and set(np.unique(x)) <= {0, 1}
    assert abs(np.mean(x[1:] != x[:-1]) - 0.05) < 0.003
    assert abs(x.mean() - 0.5) < 0.05


def test_markov_asym_stationary_fraction():
    x = g.markov_asym(200_000, rng(1), p_on=0.01, p_off=0.03)
    assert len(x) == 200_000
    assert abs(x.mean() - 0.25) < 0.03


def test_semi_markov_run_lengths():
    x = g.semi_markov(400_000, rng(2), mean_on=40, mean_off=20, sigma=1.0)
    assert len(x) == 400_000
    # ceil adds about half an epoch to each mean, so the fraction stays near 2/3
    assert abs(x.mean() - 2 / 3) < 0.04


def test_regime_is_binary_and_sized():
    x = g.regime(50_000, rng(3), p_slow=0.005, p_fast=0.3, p_switch=0.002)
    assert len(x) == 50_000 and set(np.unique(x)) <= {0, 1}
