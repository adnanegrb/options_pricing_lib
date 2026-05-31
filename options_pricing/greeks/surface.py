import numpy as np
from dataclasses import replace
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class GreeksSurface:
    def __init__(self, params: OptionParams):
        self.p = params

    def over_spot(self, S_range: np.ndarray, greek: str = "delta"):
        results = []
        for S in S_range:
            bs = BlackScholes(replace(self.p, S=S))
            results.append(getattr(bs, greek)())
        return np.array(results)

    def over_vol(self, sigma_range: np.ndarray, greek: str = "vega"):
        results = []
        for sigma in sigma_range:
            bs = BlackScholes(replace(self.p, sigma=sigma))
            results.append(getattr(bs, greek)())
        return np.array(results)

    def over_time(self, T_range: np.ndarray, greek: str = "theta"):
        results = []
        for T in T_range:
            if T <= 0:
                results.append(np.nan)
                continue
            bs = BlackScholes(replace(self.p, T=T))
            results.append(getattr(bs, greek)())
        return np.array(results)

    def vol_surface(self, S_range: np.ndarray, sigma_range: np.ndarray, greek: str = "delta"):
        surface = np.zeros((len(sigma_range), len(S_range)))
        for i, sigma in enumerate(sigma_range):
            for j, S in enumerate(S_range):
                bs = BlackScholes(replace(self.p, S=S, sigma=sigma))
                surface[i, j] = getattr(bs, greek)()
        return surface
