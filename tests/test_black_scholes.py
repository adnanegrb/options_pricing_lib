import pytest
import numpy as np
from options_pricing.models.base import OptionParams, OptionType
from options_pricing.models.black_scholes import BlackScholes


def make(option_type="call", S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0):
    return OptionParams(S=S, K=K, T=T, r=r, sigma=sigma, option_type=option_type, q=q)


def test_call_price_atm():
    bs = BlackScholes(make("call"))
    assert 8 < bs.price() < 12


def test_put_price_atm():
    bs = BlackScholes(make("put"))
    assert 5 < bs.price() < 10


def test_put_call_parity():
    call = BlackScholes(make("call")).price()
    put = BlackScholes(make("put")).price()
    p = make("call")
    lhs = call - put
    rhs = p.S * np.exp(-p.q * p.T) - p.K * np.exp(-p.r * p.T)
    assert abs(lhs - rhs) < 1e-8


def test_call_delta_range():
    bs = BlackScholes(make("call"))
    assert 0 < bs.delta() < 1


def test_put_delta_range():
    bs = BlackScholes(make("put"))
    assert -1 < bs.delta() < 0


def test_gamma_positive():
    bs = BlackScholes(make("call"))
    assert bs.gamma() > 0


def test_vega_positive():
    bs = BlackScholes(make("call"))
    assert bs.vega() > 0


def test_theta_negative():
    bs = BlackScholes(make("call"))
    assert bs.theta() < 0


def test_intrinsic_itm_call():
    bs = BlackScholes(make("call", S=120, K=100))
    assert bs.price() > 20


def test_intrinsic_otm_call():
    bs = BlackScholes(make("call", S=80, K=100))
    assert 0 < bs.price() < 20


def test_deep_itm_call_delta():
    bs = BlackScholes(make("call", S=200, K=100))
    assert bs.delta() > 0.99


def test_deep_otm_call_delta():
    bs = BlackScholes(make("call", S=50, K=100))
    assert bs.delta() < 0.01


def test_all_greeks_keys():
    bs = BlackScholes(make("call"))
    g = bs.all_greeks()
    for key in ["price", "delta", "gamma", "theta", "vega", "rho", "vanna", "volga"]:
        assert key in g


def test_call_put_delta_symmetry_no_dividends():
    call_delta = BlackScholes(make("call")).delta()
    put_delta = BlackScholes(make("put")).delta()
    assert abs(call_delta - put_delta - 1) < 1e-6


def test_short_expiry_deep_itm():
    bs = BlackScholes(make("call", S=150, K=100, T=0.01))
    assert bs.price() > 49


def test_high_vol_raises_price():
    low = BlackScholes(make("call", sigma=0.1)).price()
    high = BlackScholes(make("call", sigma=0.5)).price()
    assert high > low
