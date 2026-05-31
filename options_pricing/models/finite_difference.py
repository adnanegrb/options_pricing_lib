import numpy as np
from .base import OptionParams, ExerciseType


class FiniteDifference:
    def __init__(self, params: OptionParams, n_S: int = 200, n_T: int = 500):
        self.p = params
        self.n_S = n_S
        self.n_T = n_T

    def crank_nicolson(self):
        p = self.p
        S_max = 4 * max(p.S, p.K)
        dS = S_max / self.n_S
        dt = p.T / self.n_T

        S = np.linspace(0, S_max, self.n_S + 1)
        V = np.maximum(p.phi * (S - p.K), 0).astype(float)

        n = self.n_S - 1
        idx = np.arange(1, self.n_S)
        j = idx.astype(float)

        a = 0.25 * dt * (p.sigma**2 * j**2 - (p.r - p.q) * j)
        b = -0.5 * dt * (p.sigma**2 * j**2 + p.r)
        c = 0.25 * dt * (p.sigma**2 * j**2 + (p.r - p.q) * j)

        M_impl = np.diag(1 - b) - np.diag(a[1:], -1) - np.diag(c[:-1], 1)
        M_expl = np.diag(1 + b) + np.diag(a[1:], -1) + np.diag(c[:-1], 1)

        for step in range(self.n_T):
            rhs = M_expl @ V[1:-1]

            rhs[0]  += a[0]  * (V[0] + V[0])
            rhs[-1] += c[-1] * (V[-1] + V[-1])

            V[1:-1] = np.linalg.solve(M_impl, rhs)

            if p.exercise == ExerciseType.AMERICAN:
                V[1:-1] = np.maximum(V[1:-1], np.maximum(p.phi * (S[1:-1] - p.K), 0))

            if p.is_call:
                V[0] = 0.0
                V[-1] = S_max - p.K * np.exp(-p.r * (step + 1) * dt)
            else:
                V[0] = p.K * np.exp(-p.r * (step + 1) * dt)
                V[-1] = 0.0

        return float(np.interp(p.S, S, V))

    def price(self, method: str = "crank_nicolson"):
        return self.crank_nicolson()
