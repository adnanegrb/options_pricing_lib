import numpy as np
from .base import OptionParams, ExerciseType


class BinomialTree:
    def __init__(self, params: OptionParams, n_steps: int = 500):
        self.p = params
        self.n_steps = n_steps

    def price(self):
        p = self.p
        n = self.n_steps
        dt = p.T / n

        u = np.exp(p.sigma * np.sqrt(dt))
        d = 1 / u
        q = (np.exp((p.r - p.q) * dt) - d) / (u - d)
        discount = np.exp(-p.r * dt)

        S_T = p.S * u ** np.arange(n, -1, -1) * d ** np.arange(0, n + 1)
        V = np.maximum(p.phi * (S_T - p.K), 0)

        for step in range(n - 1, -1, -1):
            S = p.S * u ** np.arange(step, -1, -1) * d ** np.arange(0, step + 1)
            V = discount * (q * V[:-1] + (1 - q) * V[1:])

            if p.exercise == ExerciseType.AMERICAN:
                V = np.maximum(V, np.maximum(p.phi * (S - p.K), 0))

        return float(V[0])

    def delta(self):
        p = self.p
        dt = p.T / self.n_steps
        u = np.exp(p.sigma * np.sqrt(dt))
        d = 1 / u

        S_u = p.S * u
        S_d = p.S * d

        from copy import deepcopy
        from .base import OptionParams

        p_up = OptionParams(S_u, p.K, p.T, p.r, p.sigma, p.option_type, p.exercise, p.q)
        p_dn = OptionParams(S_d, p.K, p.T, p.r, p.sigma, p.option_type, p.exercise, p.q)

        V_up = BinomialTree(p_up, self.n_steps).price()
        V_dn = BinomialTree(p_dn, self.n_steps).price()

        return (V_up - V_dn) / (S_u - S_d)
