import pytest
import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.calibration.implied_vol import ImpliedVolatility


def make(sigma=0.2):
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=sigma)


def test_iv_recovers_input_vol():
    true_sigma = 0.2
    p = make(true_sigma)
    market_price = BlackScholes(p).price()
    iv = ImpliedVolatility(p).compute(market_price)
    assert abs(iv - true_sigma) < 1e-6


def test_iv_various_strikes():
    for K in [80, 90, 100, 110, 120]:
        p = OptionParams(S=100, K=K, T=1.0, r=0.05, sigma=0.25)
        market_price = BlackScholes(p).price()
        iv = ImpliedVolatility(p).compute(market_price)
        assert abs(iv - 0.25) < 1e-5


def test_iv_various_maturities():
    for T in [0.25, 0.5, 1.0, 2.0]:
        p = OptionParams(S=100, K=100, T=T, r=0.05, sigma=0.3)
        market_price = BlackScholes(p).price()
        iv = ImpliedVolatility(p).compute(market_price)
        assert abs(iv - 0.3) < 1e-5


def test_iv_below_intrinsic_raises():
    p = make()
    with pytest.raises(ValueError):
        ImpliedVolatility(p).compute(0.0)


def test_newton_raphson_recovers_vol():
    p = make(0.35)
    market_price = BlackScholes(p).price()
    iv = ImpliedVolatility(p).newton_raphson(market_price, sigma0=0.2)
    assert abs(iv - 0.35) < 1e-5


def test_smile_returns_array():
    strikes = np.array([90, 95, 100, 105, 110], dtype=float)
    p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    prices = np.array([BlackScholes(OptionParams(S=100, K=K, T=1.0, r=0.05, sigma=0.2)).price() for K in strikes])
    ivs = ImpliedVolatility(p).smile(strikes, prices)
    assert len(ivs) == len(strikes)
    assert all(abs(iv - 0.2) < 1e-5 for iv in ivs)
