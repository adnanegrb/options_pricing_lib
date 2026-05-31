import numpy as np
from .base import OptionParams


class MonteCarlo:
    def __init__(self, params: OptionParams, n_paths: int = 100_000, n_steps: int = 252, seed: int = 42):
        self.p = params
        self.n_paths = n_paths
        self.n_steps = n_steps
        self.seed = seed

    def simulate_paths(self, antithetic: bool = True):
        p = self.p
        rng = np.random.default_rng(self.seed)
        dt = p.T / self.n_steps
        drift = (p.r - p.q - 0.5 * p.sigma**2) * dt
        vol = p.sigma * np.sqrt(dt)

        half = self.n_paths // 2
        Z = rng.standard_normal((half, self.n_steps))
        if antithetic:
            Z = np.concatenate([Z, -Z], axis=0)

        log_returns = drift + vol * Z
        log_paths = np.log(p.S) + np.cumsum(log_returns, axis=1)
        paths = np.exp(log_paths)
        return paths

    def price(self, antithetic: bool = True, control_variate: bool = False):
        p = self.p
        paths = self.simulate_paths(antithetic)
        S_T = paths[:, -1]
        payoffs = np.maximum(p.phi * (S_T - p.K), 0)

        if control_variate:
            cv_mean = p.S * np.exp((p.r - p.q) * p.T)
            beta = np.cov(payoffs, S_T)[0, 1] / np.var(S_T)
            payoffs = payoffs - beta * (S_T - cv_mean)

        discounted = np.exp(-p.r * p.T) * payoffs
        return float(np.mean(discounted))

    def price_with_ci(self, confidence: float = 0.95, antithetic: bool = True):
        p = self.p
        paths = self.simulate_paths(antithetic)
        S_T = paths[:, -1]
        payoffs = np.exp(-p.r * p.T) * np.maximum(p.phi * (S_T - p.K), 0)

        mean = float(np.mean(payoffs))
        std = float(np.std(payoffs, ddof=1))
        n = len(payoffs)

        from scipy.stats import norm
        z = norm.ppf((1 + confidence) / 2)
        margin = z * std / np.sqrt(n)

        return {
            "price": mean,
            "lower": mean - margin,
            "upper": mean + margin,
            "std_error": std / np.sqrt(n),
        }

    def price_asian(self, averaging: str = "arithmetic"):
        p = self.p
        paths = self.simulate_paths()

        if averaging == "arithmetic":
            avg = np.mean(paths, axis=1)
        else:
            avg = np.exp(np.mean(np.log(paths), axis=1))

        payoffs = np.maximum(p.phi * (avg - p.K), 0)
        return float(np.exp(-p.r * p.T) * np.mean(payoffs))

    def price_barrier(self, barrier: float, barrier_type: str = "knock_out", direction: str = "up"):
        p = self.p
        paths = self.simulate_paths()

        if direction == "up":
            crossed = np.any(paths >= barrier, axis=1)
        else:
            crossed = np.any(paths <= barrier, axis=1)

        S_T = paths[:, -1]
        vanilla_payoff = np.maximum(p.phi * (S_T - p.K), 0)

        if barrier_type == "knock_out":
            payoffs = np.where(crossed, 0, vanilla_payoff)
        else:
            payoffs = np.where(crossed, vanilla_payoff, 0)

        return float(np.exp(-p.r * p.T) * np.mean(payoffs))
