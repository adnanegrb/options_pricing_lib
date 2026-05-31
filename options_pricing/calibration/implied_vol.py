import numpy as np
from scipy.optimize import brentq, minimize
from dataclasses import replace
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class ImpliedVolatility:
    def __init__(self, params: OptionParams):
        self.p = params

    def compute(self, market_price: float, tol: float = 1e-8, max_iter: int = 500):
        p = self.p

        intrinsic = p.intrinsic_value()
        upper_bound = p.S * np.exp(-p.q * p.T)

        if market_price <= intrinsic:
            raise ValueError(f"Market price {market_price:.4f} is below intrinsic value {intrinsic:.4f}")
        if market_price >= upper_bound:
            raise ValueError(f"Market price {market_price:.4f} is above upper bound {upper_bound:.4f}")

        def objective(sigma):
            return BlackScholes(replace(p, sigma=sigma)).price() - market_price

        try:
            iv = brentq(objective, 1e-6, 10.0, xtol=tol, maxiter=max_iter)
        except ValueError:
            raise ValueError("Could not find implied vol. Check that market price is valid.")

        return float(iv)

    def newton_raphson(self, market_price: float, sigma0: float = 0.2, tol: float = 1e-8, max_iter: int = 100):
        p = self.p
        sigma = sigma0

        for _ in range(max_iter):
            bs = BlackScholes(replace(p, sigma=sigma))
            price = bs.price()
            vega = bs.vega() * 100

            if abs(vega) < 1e-12:
                break

            diff = price - market_price
            if abs(diff) < tol:
                return float(sigma)

            sigma -= diff / vega

            if sigma <= 0:
                sigma = 1e-6

        return float(sigma)

    def smile(self, strikes: np.ndarray, market_prices: np.ndarray):
        ivs = []
        for K, price in zip(strikes, market_prices):
            try:
                params = replace(self.p, K=K)
                iv = ImpliedVolatility(params).compute(price)
                ivs.append(iv)
            except Exception:
                ivs.append(np.nan)
        return np.array(ivs)
