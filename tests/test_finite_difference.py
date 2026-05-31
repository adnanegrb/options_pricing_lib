import pytest
import numpy as np
from options_pricing.models.base import OptionParams, ExerciseType
from options_pricing.models.finite_difference import FiniteDifference
from options_pricing.models.black_scholes import BlackScholes


def make(option_type="call", exercise="european"):
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type=option_type, exercise=exercise)


def test_cn_call_close_to_bs():
    p = make("call")
    bs = BlackScholes(p).price()
    fd = FiniteDifference(p).crank_nicolson()
    assert abs(fd - bs) < 0.05


def test_cn_put_close_to_bs():
    p = make("put")
    bs = BlackScholes(p).price()
    fd = FiniteDifference(p).crank_nicolson()
    assert abs(fd - bs) < 0.05


def test_american_put_geq_european_put():
    eu = FiniteDifference(make("put", "european")).crank_nicolson()
    am = FiniteDifference(make("put", "american")).crank_nicolson()
    assert am >= eu - 1e-6


def test_american_call_no_dividend_equals_european():
    eu = FiniteDifference(make("call", "european")).crank_nicolson()
    am = FiniteDifference(make("call", "american")).crank_nicolson()
    assert abs(am - eu) < 0.1


def test_price_method_default():
    p = make("call")
    fd = FiniteDifference(p)
    assert abs(fd.price() - BlackScholes(p).price()) < 0.05


def test_itm_call_price_reasonable():
    p = OptionParams(S=110, K=100, T=1.0, r=0.05, sigma=0.2)
    bs = BlackScholes(p).price()
    fd = FiniteDifference(p).crank_nicolson()
    assert abs(fd - bs) < 0.1
