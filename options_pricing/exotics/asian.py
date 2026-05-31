import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.models.monte_carlo import MonteCarlo


class AsianOption:
    def __init__(self, params: OptionParams, n_paths: int = 100_000, n_steps: int = 252, seed: int = 42):
        self.p = params
        self.mc = MonteCarlo(params, n_paths, n_steps, seed)

    def price_mc(self, averaging: str = "arithmetic"):
        return self.mc.price_asian(averaging)

    def price_geometric_closed_form(self):
        p = self.p
        n = self.mc.n_steps
        sigma_adj = p.sigma / np.sqrt(3)
        mu_adj = 0.5 * (p.r - p.q - p.sigma**2 / 6)

        from scipy.stats import norm
        d1 = (np.log(p.S / p.K) + (mu_adj + 0.5 * sigma_adj**2) * p.T) / (sigma_adj * np.sqrt(p.T))
        d2 = d1 - sigma_adj * np.sqrt(p.T)

        if p.is_call:
            price = np.exp(-p.r * p.T) * (
                p.S * np.exp(mu_adj * p.T) * norm.cdf(d1) - p.K * norm.cdf(d2)
            )
        else:
            price = np.exp(-p.r * p.T) * (
                p.K * norm.cdf(-d2) - p.S * np.exp(mu_adj * p.T) * norm.cdf(-d1)
            )

        return float(price)

    def price_arithmetic_control_variate(self):
        p = self.p
        paths = self.mc.simulate_paths()

        arith_avg = np.mean(paths, axis=1)
        geom_avg = np.exp(np.mean(np.log(paths), axis=1))

        arith_payoff = np.maximum(p.phi * (arith_avg - p.K), 0)
        geom_payoff = np.maximum(p.phi * (geom_avg - p.K), 0)

        geom_price = self.price_geometric_closed_form()

        beta = np.cov(arith_payoff, geom_payoff)[0, 1] / np.var(geom_payoff)
        controlled = arith_payoff - beta * (geom_payoff - geom_price * np.exp(p.r * p.T))

        return float(np.exp(-p.r * p.T) * np.mean(controlled))
