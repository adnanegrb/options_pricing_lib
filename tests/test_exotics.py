import pytest
import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.exotics.asian import AsianOption
from options_pricing.exotics.barrier import BarrierOption
from options_pricing.exotics.lookback import LookbackOption
from options_pricing.exotics.digital import DigitalOption


def make(option_type="call"):
    return OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type=option_type)


def test_asian_arith_positive():
    assert AsianOption(make()).price_mc("arithmetic") > 0


def test_asian_geom_positive():
    assert AsianOption(make()).price_mc("geometric") > 0


def test_asian_geom_closed_form_vs_mc():
    opt = AsianOption(make())
    closed = opt.price_geometric_closed_form()
    mc = opt.price_mc("geometric")
    assert abs(closed - mc) < 0.3


def test_asian_arith_geq_geom():
    opt = AsianOption(make())
    assert opt.price_mc("arithmetic") >= opt.price_mc("geometric") - 0.5


def test_barrier_knock_out_positive():
    p = make()
    b = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up")
    assert b.price("mc") >= 0


def test_barrier_knock_in_positive():
    p = make()
    b = BarrierOption(p, barrier=120, barrier_type="knock_in", direction="up")
    assert b.price("mc") >= 0


def test_barrier_ko_plus_ki_equals_vanilla():
    from options_pricing.models.black_scholes import BlackScholes
    p = make()
    ko = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up").price_mc(n_paths=200_000)
    ki = BarrierOption(p, barrier=120, barrier_type="knock_in", direction="up").price_mc(n_paths=200_000)
    vanilla = BlackScholes(p).price()
    assert abs(ko + ki - vanilla) < 0.5


def test_lookback_fixed_positive():
    assert LookbackOption(make()).price_fixed_strike_mc() > 0


def test_lookback_floating_positive():
    assert LookbackOption(make()).price_floating_strike_mc() > 0


def test_digital_cash_or_nothing_range():
    d = DigitalOption(make(), cash=1.0)
    price = d.price_cash_or_nothing()
    assert 0 < price < 1


def test_digital_mc_close_to_analytical():
    d = DigitalOption(make(), cash=1.0)
    analytical = d.price_cash_or_nothing()
    mc = d.price_mc(n_paths=200_000)
    assert abs(analytical - mc) < 0.05
