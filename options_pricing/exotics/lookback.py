import numpy as np
from scipy.stats import norm
from options_pricing.models.base import OptionParams
from options_pricing.models.monte_carlo import MonteCarlo


class LookbackOption:
    def __init__(self, params: OptionParams):
        self.p = params

    def price_fixed_strike_mc(self, n_paths: int = 100_000, n_steps: int = 252, seed: int = 42):
        p = self.p
        mc = MonteCarlo(p, n_paths, n_steps, seed)
        paths = mc.simulate_paths()

        if p.is_call:
            payoffs = np.maximum(np.max(paths, axis=1) - p.K, 0)
        else:
            payoffs = np.maximum(p.K - np.min(paths, axis=1), 0)

        return float(np.exp(-p.r * p.T) * np.mean(payoffs))

    def price_floating_strike_mc(self, n_paths: int = 100_000, n_steps: int = 252, seed: int = 42):
        p = self.p
        mc = MonteCarlo(p, n_paths, n_steps, seed)
        paths = mc.simulate_paths()
        S_T = paths[:, -1]

        if p.is_call:
            payoffs = S_T - np.min(paths, axis=1)
        else:
            payoffs = np.max(paths, axis=1) - S_T

        return float(np.exp(-p.r * p.T) * np.mean(payoffs))

    def price_floating_strike_analytical(self):
        p = self.p
        S, r, q, sigma, T = p.S, p.r, p.q, p.sigma, p.T

        a1 = (np.log(S / S) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        a2 = a1 - sigma * np.sqrt(T)
        a3 = (np.log(S / S) + (-r + q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))

        if p.is_call:
            price = (S * np.exp(-q * T) * norm.cdf(a1)
                     - S * np.exp(-r * T) * norm.cdf(a2)
                     + S * np.exp(-r * T) * sigma**2 / (2 * (r - q))
                     * (norm.cdf(-a1) - np.exp((r - q) * T) * norm.cdf(-a3)))
        else:
            a1 = -a1
            a2 = -a2
            price = (S * np.exp(-r * T) * norm.cdf(a2)
                     - S * np.exp(-q * T) * norm.cdf(a1)
                     + S * np.exp(-r * T) * sigma**2 / (2 * (r - q))
                     * (norm.cdf(a1) - np.exp((r - q) * T) * norm.cdf(a3)))

        return float(price)
