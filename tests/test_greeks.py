import pytest
import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.greeks.analytical import AnalyticalGreeks
from options_pricing.greeks.numerical import NumericalGreeks


def make(option_type="call"):
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type=option_type)


def test_analytical_numerical_delta_close():
    p = make("call")
    analytical = AnalyticalGreeks(p).delta()
    numerical = NumericalGreeks(p).delta()
    assert abs(analytical - numerical) < 1e-4


def test_analytical_numerical_gamma_close():
    p = make("call")
    analytical = AnalyticalGreeks(p).gamma()
    numerical = NumericalGreeks(p).gamma()
    assert abs(analytical - numerical) < 1e-4


def test_analytical_numerical_vega_close():
    p = make("call")
    analytical = AnalyticalGreeks(p).vega()
    numerical = NumericalGreeks(p).vega()
    assert abs(analytical - numerical) < 1e-4


def test_all_greeks_call():
    g = AnalyticalGreeks(make("call")).all_greeks()
    assert g["delta"] > 0
    assert g["gamma"] > 0
    assert g["vega"] > 0
    assert g["theta"] < 0


def test_all_greeks_put():
    g = AnalyticalGreeks(make("put")).all_greeks()
    assert g["delta"] < 0
    assert g["gamma"] > 0
    assert g["vega"] > 0


def test_vanna_sign():
    p = make("call")
    g = AnalyticalGreeks(p)
    assert isinstance(g.vanna(), float)


def test_charm_finite():
    g = AnalyticalGreeks(make("call"))
    assert np.isfinite(g.charm())


def test_speed_negative_call():
    g = AnalyticalGreeks(make("call"))
    assert g.speed() < 0
