import numpy as np
from scipy.stats import norm
from .base import OptionParams


class BlackScholes:
    def __init__(self, params: OptionParams):
        self.p = params

    def d1(self):
        p = self.p
        return (np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))

    def d2(self):
        return self.d1() - self.p.sigma * np.sqrt(self.p.T)

    def price(self):
        p = self.p
        d1, d2 = self.d1(), self.d2()
        discount = np.exp(-p.r * p.T)
        div_discount = np.exp(-p.q * p.T)

        if p.is_call:
            return p.S * div_discount * norm.cdf(d1) - p.K * discount * norm.cdf(d2)
        else:
            return p.K * discount * norm.cdf(-d2) - p.S * div_discount * norm.cdf(-d1)

    def delta(self):
        p = self.p
        d1 = self.d1()
        div_discount = np.exp(-p.q * p.T)
        if p.is_call:
            return div_discount * norm.cdf(d1)
        else:
            return div_discount * (norm.cdf(d1) - 1)

    def gamma(self):
        p = self.p
        d1 = self.d1()
        return np.exp(-p.q * p.T) * norm.pdf(d1) / (p.S * p.sigma * np.sqrt(p.T))

    def theta(self):
        p = self.p
        d1, d2 = self.d1(), self.d2()
        term1 = -np.exp(-p.q * p.T) * p.S * norm.pdf(d1) * p.sigma / (2 * np.sqrt(p.T))
        if p.is_call:
            return (term1 - p.r * p.K * np.exp(-p.r * p.T) * norm.cdf(d2)
                    + p.q * p.S * np.exp(-p.q * p.T) * norm.cdf(d1)) / 365
        else:
            return (term1 + p.r * p.K * np.exp(-p.r * p.T) * norm.cdf(-d2)
                    - p.q * p.S * np.exp(-p.q * p.T) * norm.cdf(-d1)) / 365

    def vega(self):
        p = self.p
        d1 = self.d1()
        return p.S * np.exp(-p.q * p.T) * norm.pdf(d1) * np.sqrt(p.T) / 100

    def rho(self):
        p = self.p
        d2 = self.d2()
        if p.is_call:
            return p.K * p.T * np.exp(-p.r * p.T) * norm.cdf(d2) / 100
        else:
            return -p.K * p.T * np.exp(-p.r * p.T) * norm.cdf(-d2) / 100

    def vanna(self):
        p = self.p
        d1, d2 = self.d1(), self.d2()
        return -np.exp(-p.q * p.T) * norm.pdf(d1) * d2 / p.sigma

    def volga(self):
        p = self.p
        d1, d2 = self.d1(), self.d2()
        return p.S * np.exp(-p.q * p.T) * norm.pdf(d1) * np.sqrt(p.T) * d1 * d2 / p.sigma

    def all_greeks(self):
        return {
            "price": self.price(),
            "delta": self.delta(),
            "gamma": self.gamma(),
            "theta": self.theta(),
            "vega":  self.vega(),
            "rho":   self.rho(),
            "vanna": self.vanna(),
            "volga": self.volga(),
        }
