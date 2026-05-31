import numpy as np
from dataclasses import replace
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


def bump(params, **kwargs):
    return BlackScholes(replace(params, **kwargs)).price()


class NumericalGreeks:
    def __init__(self, params: OptionParams, dS: float = None, dsigma: float = 0.001, dr: float = 0.0001, dT: float = 1/365):
        self.p = params
        self.dS = dS if dS is not None else params.S * 0.001
        self.dsigma = dsigma
        self.dr = dr
        self.dT = dT

    def delta(self):
        p, dS = self.p, self.dS
        return (bump(p, S=p.S + dS) - bump(p, S=p.S - dS)) / (2 * dS)

    def gamma(self):
        p, dS = self.p, self.dS
        V0 = BlackScholes(p).price()
        return (bump(p, S=p.S + dS) - 2 * V0 + bump(p, S=p.S - dS)) / dS**2

    def vega(self):
        p, ds = self.p, self.dsigma
        return (bump(p, sigma=p.sigma + ds) - bump(p, sigma=p.sigma - ds)) / (2 * ds * 100)

    def theta(self):
        p, dT = self.p, self.dT
        if p.T <= dT:
            return 0.0
        return (bump(p, T=p.T - dT) - BlackScholes(p).price()) / dT / 365

    def rho(self):
        p, dr = self.p, self.dr
        return (bump(p, r=p.r + dr) - bump(p, r=p.r - dr)) / (2 * dr * 100)

    def vanna(self):
        p, dS, ds = self.p, self.dS, self.dsigma
        vega_up = (bump(p, S=p.S + dS, sigma=p.sigma + ds) - bump(p, S=p.S + dS, sigma=p.sigma - ds)) / (2 * ds)
        vega_dn = (bump(p, S=p.S - dS, sigma=p.sigma + ds) - bump(p, S=p.S - dS, sigma=p.sigma - ds)) / (2 * ds)
        return (vega_up - vega_dn) / (2 * dS)

    def volga(self):
        p, ds = self.p, self.dsigma
        V0 = BlackScholes(p).price()
        return (bump(p, sigma=p.sigma + ds) - 2 * V0 + bump(p, sigma=p.sigma - ds)) / ds**2

    def all_greeks(self):
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega":  self.vega(),
            "rho":   self.rho(),
            "vanna": self.vanna(),
            "volga": self.volga(),
        }
