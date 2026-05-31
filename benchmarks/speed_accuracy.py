import time
import numpy as np
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.models.finite_difference import FiniteDifference
from options_pricing.models.binomial_tree import BinomialTree


params = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)


def benchmark(name, fn, n=5):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = fn()
        times.append(time.perf_counter() - t0)
    avg = np.mean(times) * 1000
    print(f"{name:<35} price={result:.6f}  avg={avg:.2f}ms")
    return result, avg


print("=" * 70)
print("Options Pricing Library — Speed Benchmark")
print("=" * 70)

bs_price, _ = benchmark("Black-Scholes (analytical)", lambda: BlackScholes(params).price())

benchmark("Monte Carlo (10k paths)",   lambda: MonteCarlo(params, n_paths=10_000).price())
benchmark("Monte Carlo (100k paths)",  lambda: MonteCarlo(params, n_paths=100_000).price())
benchmark("Monte Carlo (500k paths)",  lambda: MonteCarlo(params, n_paths=500_000).price())

benchmark("Finite Difference (CN, 100x100)",   lambda: FiniteDifference(params, n_S=100, n_T=100).crank_nicolson())
benchmark("Finite Difference (CN, 200x200)",   lambda: FiniteDifference(params, n_S=200, n_T=200).crank_nicolson())

benchmark("Binomial Tree (100 steps)",  lambda: BinomialTree(params, n_steps=100).price())
benchmark("Binomial Tree (500 steps)",  lambda: BinomialTree(params, n_steps=500).price())
benchmark("Binomial Tree (1000 steps)", lambda: BinomialTree(params, n_steps=1000).price())

print("=" * 70)
print(f"Reference Black-Scholes price: {bs_price:.6f}")
print("=" * 70)

print("\nAccuracy vs Black-Scholes:")
print("-" * 50)
configs = [
    ("MC 10k",   lambda: MonteCarlo(params, n_paths=10_000).price()),
    ("MC 100k",  lambda: MonteCarlo(params, n_paths=100_000).price()),
    ("MC 500k",  lambda: MonteCarlo(params, n_paths=500_000).price()),
    ("FD CN 200", lambda: FiniteDifference(params).crank_nicolson()),
    ("BT 500",   lambda: BinomialTree(params, n_steps=500).price()),
]

for name, fn in configs:
    price = fn()
    error = abs(price - bs_price)
    rel_error = error / bs_price * 100
    print(f"{name:<15} price={price:.6f}  abs_error={error:.6f}  rel_error={rel_error:.4f}%")
