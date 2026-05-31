import numpy as np
from scipy.integrate import quad
from dataclasses import dataclass
from scipy.optimize import minimize


@dataclass
class HestonParams:
    S: float
    K: float
    T: float
    r: float
    v0: float
    kappa: float
    theta: float
    xi: float
    rho: float


class HestonModel:
    def __init__(self, params: HestonParams):
        self.p = params

    def characteristic_function(self, phi, j):
        p = self.p
        if j == 1:
            u, b = 0.5, p.kappa - p.rho * p.xi
        else:
            u, b = -0.5, p.kappa

        a = p.kappa * p.theta
        x = np.log(p.S)

        d = np.sqrt((p.rho * p.xi * phi * 1j - b)**2 - p.xi**2 * (2 * u * phi * 1j - phi**2))
        g = (b - p.rho * p.xi * phi * 1j + d) / (b - p.rho * p.xi * phi * 1j - d)

        C = (p.r * phi * 1j * p.T
             + a / p.xi**2 * ((b - p.rho * p.xi * phi * 1j + d) * p.T
             - 2 * np.log((1 - g * np.exp(d * p.T)) / (1 - g))))

        D = (b - p.rho * p.xi * phi * 1j + d) / p.xi**2 * ((1 - np.exp(d * p.T)) / (1 - g * np.exp(d * p.T)))

        return np.exp(C + D * p.v0 + 1j * phi * x)

    def integrand(self, phi, j, K):
        cf = self.characteristic_function(phi, j)
        return np.real(np.exp(-1j * phi * np.log(K)) * cf / (1j * phi))

    def price(self, option_type: str = "call"):
        p = self.p
        P1, _ = quad(lambda phi: self.integrand(phi, 1, p.K), 0, 500, limit=500)
        P2, _ = quad(lambda phi: self.integrand(phi, 2, p.K), 0, 500, limit=500)

        P1 = 0.5 + P1 / np.pi
        P2 = 0.5 + P2 / np.pi

        call = p.S * P1 - p.K * np.exp(-p.r * p.T) * P2

        if option_type == "call":
            return float(call)
        else:
            return float(call - p.S + p.K * np.exp(-p.r * p.T))

    def simulate_paths(self, n_paths: int = 10_000, n_steps: int = 252, seed: int = 42):
        p = self.p
        rng = np.random.default_rng(seed)
        dt = p.T / n_steps

        S = np.full(n_paths, p.S)
        v = np.full(n_paths, p.v0)

        for _ in range(n_steps):
            Z1 = rng.standard_normal(n_paths)
            Z2 = p.rho * Z1 + np.sqrt(1 - p.rho**2) * rng.standard_normal(n_paths)

            v_pos = np.maximum(v, 0)
            v = (v + p.kappa * (p.theta - v_pos) * dt + p.xi * np.sqrt(v_pos * dt) * Z2)
            S = S * np.exp((p.r - 0.5 * v_pos) * dt + np.sqrt(v_pos * dt) * Z1)

        return S

    @staticmethod
    def calibrate(market_prices: np.ndarray, strikes: np.ndarray, S: float, T: float, r: float):
        def objective(x):
            kappa, theta, xi, rho, v0 = x
            if xi <= 0 or theta <= 0 or v0 <= 0 or abs(rho) >= 1:
                return 1e10
            if 2 * kappa * theta < xi**2:
                return 1e10

            total = 0
            for K, mkt in zip(strikes, market_prices):
                params = HestonParams(S, K, T, r, v0, kappa, theta, xi, rho)
                model_price = HestonModel(params).price("call")
                total += (model_price - mkt)**2
            return total

        x0 = [2.0, 0.04, 0.3, -0.7, 0.04]
        bounds = [(0.01, 15), (0.001, 1), (0.01, 2), (-0.99, 0.99), (0.001, 1)]
        result = minimize(objective, x0, bounds=bounds, method="L-BFGS-B")

        kappa, theta, xi, rho, v0 = result.x
        return HestonParams(S, strikes[0], T, r, v0, kappa, theta, xi, rho)
