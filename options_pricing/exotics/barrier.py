import numpy as np
from scipy.stats import norm
from options_pricing.models.base import OptionParams
from options_pricing.models.monte_carlo import MonteCarlo


class BarrierOption:
    def __init__(self, params: OptionParams, barrier: float, barrier_type: str = "knock_out", direction: str = "up"):
        self.p = params
        self.barrier = barrier
        self.barrier_type = barrier_type
        self.direction = direction

    def price_mc(self, n_paths: int = 100_000, n_steps: int = 252, seed: int = 42):
        mc = MonteCarlo(self.p, n_paths, n_steps, seed)
        return mc.price_barrier(self.barrier, self.barrier_type, self.direction)

    def price_analytical(self):
        p = self.p
        S, K, H = p.S, p.K, self.barrier
        r, q, sigma, T = p.r, p.q, p.sigma, p.T
        phi = p.phi
        eta = 1 if self.direction == "down" else -1

        mu = (r - q - 0.5 * sigma**2) / sigma**2
        lam = np.sqrt(mu**2 + 2 * r / sigma**2)

        def x1():
            return np.log(S / K) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

        def x2():
            return np.log(S / H) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

        def y1():
            return np.log(H**2 / (S * K)) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

        def y2():
            return np.log(H / S) / (sigma * np.sqrt(T)) + (1 + mu) * sigma * np.sqrt(T)

        A = phi * S * np.exp(-q * T) * norm.cdf(phi * x1()) - phi * K * np.exp(-r * T) * norm.cdf(phi * x1() - phi * sigma * np.sqrt(T))
        B = phi * S * np.exp(-q * T) * norm.cdf(phi * x2()) - phi * K * np.exp(-r * T) * norm.cdf(phi * x2() - phi * sigma * np.sqrt(T))
        C = phi * S * np.exp(-q * T) * (H / S)**(2 * (mu + 1)) * norm.cdf(eta * y1()) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * norm.cdf(eta * y1() - eta * sigma * np.sqrt(T))
        D = phi * S * np.exp(-q * T) * (H / S)**(2 * (mu + 1)) * norm.cdf(eta * y2()) - phi * K * np.exp(-r * T) * (H / S)**(2 * mu) * norm.cdf(eta * y2() - eta * sigma * np.sqrt(T))

        if self.direction == "down" and p.is_call:
            if K >= H:
                return A - C
            else:
                return B - D
        elif self.direction == "up" and p.is_call:
            if K >= H:
                return 0.0
            else:
                return A - B + C - D
        elif self.direction == "down" and not p.is_call:
            if K >= H:
                return A - B + C - D
            else:
                return 0.0
        else:
            if K >= H:
                return B - D
            else:
                return A - C

    def price(self, method: str = "analytical"):
        if method == "mc":
            return self.price_mc()
        return self.price_analytical()
