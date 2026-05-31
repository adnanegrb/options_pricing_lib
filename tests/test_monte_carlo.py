import pytest
import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.models.black_scholes import BlackScholes


def make(option_type="call"):
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type=option_type)


def test_call_close_to_bs():
    p = make("call")
    mc_price = MonteCarlo(p, n_paths=200_000).price()
    bs_price = BlackScholes(p).price()
    assert abs(mc_price - bs_price) < 0.15


def test_put_close_to_bs():
    p = make("put")
    mc_price = MonteCarlo(p, n_paths=200_000).price()
    bs_price = BlackScholes(p).price()
    assert abs(mc_price - bs_price) < 0.15


def test_antithetic_reduces_variance():
    p = make("call")
    prices_no_av = [MonteCarlo(p, n_paths=10_000, seed=i).price(antithetic=False) for i in range(10)]
    prices_av = [MonteCarlo(p, n_paths=10_000, seed=i).price(antithetic=True) for i in range(10)]
    assert np.std(prices_av) < np.std(prices_no_av)


def test_price_with_ci_contains_bs():
    p = make("call")
    result = MonteCarlo(p, n_paths=200_000).price_with_ci()
    bs_price = BlackScholes(p).price()
    assert result["lower"] < bs_price < result["upper"]


def test_asian_lower_than_vanilla():
    p = make("call")
    asian = MonteCarlo(p).price_asian()
    vanilla = MonteCarlo(p).price()
    assert asian < vanilla


def test_barrier_knock_out_lower_than_vanilla():
    p = make("call")
    barrier_price = MonteCarlo(p).price_barrier(barrier=120, barrier_type="knock_out", direction="up")
    vanilla = MonteCarlo(p).price()
    assert barrier_price < vanilla


def test_paths_shape():
    p = make("call")
    mc = MonteCarlo(p, n_paths=1000, n_steps=50)
    paths = mc.simulate_paths()
    assert paths.shape == (1000, 50)


def test_positive_price():
    for opt in ["call", "put"]:
        p = make(opt)
        assert MonteCarlo(p).price() > 0
