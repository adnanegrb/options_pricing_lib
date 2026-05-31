import numpy as np
from scipy.stats import norm
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class AnalyticalGreeks:
    def __init__(self, params: OptionParams):
        self.p = params
        self.bs = BlackScholes(params)

    def delta(self):
        return self.bs.delta()

    def gamma(self):
        return self.bs.gamma()

    def theta(self):
        return self.bs.theta()

    def vega(self):
        return self.bs.vega()

    def rho(self):
        return self.bs.rho()

    def vanna(self):
        return self.bs.vanna()

    def volga(self):
        return self.bs.volga()

    def charm(self):
        p = self.p
        d1, d2 = self.bs.d1(), self.bs.d2()
        part1 = p.q * np.exp(-p.q * p.T) * norm.cdf(p.phi * d1)
        part2 = np.exp(-p.q * p.T) * norm.pdf(d1) * (
            2 * (p.r - p.q) * p.T - d2 * p.sigma * np.sqrt(p.T)
        ) / (2 * p.T * p.sigma * np.sqrt(p.T))
        return p.phi * (part1 - part2)

    def speed(self):
        p = self.p
        d1 = self.bs.d1()
        gamma = self.gamma()
        return -gamma / p.S * (d1 / (p.sigma * np.sqrt(p.T)) + 1)

    def color(self):
        p = self.p
        d1, d2 = self.bs.d1(), self.bs.d2()
        gamma = self.gamma()
        return -gamma / (2 * p.T) * (
            2 * p.q * p.T + 1 + d1 * (2 * (p.r - p.q) * p.T - d2 * p.sigma * np.sqrt(p.T))
            / (p.sigma * np.sqrt(p.T))
        )

    def all_greeks(self):
        return {
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega":  self.vega(),
            "rho":   self.rho(),
            "vanna": self.vanna(),
            "volga": self.volga(),
            "charm": self.charm(),
            "speed": self.speed(),
            "color": self.color(),
        }
