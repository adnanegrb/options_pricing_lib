import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import replace
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes


class Visualizer:
    def __init__(self, params: OptionParams):
        self.p = params

    def payoff_diagram(self, ax=None):
        p = self.p
        S_range = np.linspace(0.5 * p.K, 1.5 * p.K, 300)
        payoffs = np.maximum(p.phi * (S_range - p.K), 0)
        prices = np.array([BlackScholes(replace(p, S=S)).price() for S in S_range])

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))

        ax.plot(S_range, payoffs, "k--", label="Payoff at expiry", linewidth=1.5)
        ax.plot(S_range, prices, "royalblue", label="Option price today", linewidth=2)
        ax.axvline(p.K, color="gray", linestyle=":", alpha=0.6, label=f"Strike K={p.K}")
        ax.axvline(p.S, color="tomato", linestyle=":", alpha=0.6, label=f"Spot S={p.S}")
        ax.fill_between(S_range, payoffs, prices, alpha=0.08, color="royalblue")
        ax.set_xlabel("Spot price S")
        ax.set_ylabel("Option value")
        ax.set_title(f"Payoff diagram — {p.option_type.value.upper()}")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return ax

    def greeks_dashboard(self, S_range=None):
        p = self.p
        if S_range is None:
            S_range = np.linspace(0.6 * p.K, 1.4 * p.K, 200)

        greeks = {name: [] for name in ["price", "delta", "gamma", "vega", "theta"]}
        for S in S_range:
            bs = BlackScholes(replace(p, S=S))
            g = bs.all_greeks()
            for name in greeks:
                greeks[name].append(g[name])

        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig)

        axes = [
            fig.add_subplot(gs[0, :2]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 0]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]

        labels = ["Price", "Delta", "Gamma", "Vega (×100)", "Theta (daily)"]
        colors = ["royalblue", "seagreen", "darkorange", "mediumpurple", "crimson"]
        keys = ["price", "delta", "gamma", "vega", "theta"]

        for ax, label, color, key in zip(axes, labels, colors, keys):
            ax.plot(S_range, greeks[key], color=color, linewidth=2)
            ax.axvline(p.K, color="gray", linestyle=":", alpha=0.5)
            ax.axvline(p.S, color="red", linestyle=":", alpha=0.5)
            ax.set_title(label)
            ax.set_xlabel("S")
            ax.grid(True, alpha=0.3)

        fig.suptitle(f"Greeks dashboard — {p.option_type.value.upper()} | K={p.K}, T={p.T:.2f}, σ={p.sigma:.0%}", fontsize=13)
        plt.tight_layout()
        return fig

    def vol_surface_plot(self, strikes: np.ndarray, maturities: np.ndarray, ivs: np.ndarray):
        from mpl_toolkits.mplot3d import Axes3D

        K_grid, T_grid = np.meshgrid(strikes, maturities)

        fig = plt.figure(figsize=(12, 7))
        ax = fig.add_subplot(111, projection="3d")
        surf = ax.plot_surface(K_grid, T_grid, ivs * 100, cmap="RdYlGn", edgecolor="none", alpha=0.9)
        ax.set_xlabel("Strike K")
        ax.set_ylabel("Maturity T (years)")
        ax.set_zlabel("Implied Vol (%)")
        ax.set_title("Implied Volatility Surface")
        fig.colorbar(surf, ax=ax, shrink=0.5, label="IV %")
        return fig

    def price_convergence(self, n_paths_list: list):
        from options_pricing.models.monte_carlo import MonteCarlo
        from options_pricing.models.black_scholes import BlackScholes

        bs_price = BlackScholes(self.p).price()
        mc_prices = []

        for n in n_paths_list:
            mc = MonteCarlo(self.p, n_paths=n)
            mc_prices.append(mc.price())

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogx(n_paths_list, mc_prices, "o-", color="royalblue", label="Monte Carlo")
        ax.axhline(bs_price, color="tomato", linestyle="--", label=f"Black-Scholes: {bs_price:.4f}")
        ax.fill_between(n_paths_list,
                        [bs_price * 0.99] * len(n_paths_list),
                        [bs_price * 1.01] * len(n_paths_list),
                        alpha=0.1, color="tomato")
        ax.set_xlabel("Number of paths")
        ax.set_ylabel("Price")
        ax.set_title("Monte Carlo convergence to Black-Scholes")
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
