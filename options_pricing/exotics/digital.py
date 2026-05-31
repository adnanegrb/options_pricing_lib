import numpy as np
from scipy.stats import norm
from options_pricing.models.base import OptionParams
from options_pricing.models.monte_carlo import MonteCarlo


class DigitalOption:
    def __init__(self, params: OptionParams, cash: float = 1.0):
        self.p = params
        self.cash = cash

    def d2(self):
        p = self.p
        return (np.log(p.S / p.K) + (p.r - p.q - 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))

    def price_cash_or_nothing(self):
        p = self.p
        d2 = self.d2()
        if p.is_call:
            return self.cash * np.exp(-p.r * p.T) * norm.cdf(d2)
        else:
            return self.cash * np.exp(-p.r * p.T) * norm.cdf(-d2)

    def price_asset_or_nothing(self):
        p = self.p
        d1 = (np.log(p.S / p.K) + (p.r - p.q + 0.5 * p.sigma**2) * p.T) / (p.sigma * np.sqrt(p.T))
        if p.is_call:
            return p.S * np.exp(-p.q * p.T) * norm.cdf(d1)
        else:
            return p.S * np.exp(-p.q * p.T) * norm.cdf(-d1)

    def price_mc(self, n_paths: int = 100_000, seed: int = 42):
        p = self.p
        mc = MonteCarlo(p, n_paths, seed=seed)
        paths = mc.simulate_paths()
        S_T = paths[:, -1]

        if p.is_call:
            payoffs = np.where(S_T > p.K, self.cash, 0)
        else:
            payoffs = np.where(S_T < p.K, self.cash, 0)

        return float(np.exp(-p.r * p.T) * np.mean(payoffs))

    def delta(self):
        p = self.p
        d2 = self.d2()
        return self.cash * np.exp(-p.r * p.T) * norm.pdf(d2) * p.phi / (p.S * p.sigma * np.sqrt(p.T))
