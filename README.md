# Options Pricing Library

A professional Python library for pricing vanilla and exotic options. Built for clarity and correctness — every model is cross-validated against the others, and every Greek is tested both analytically and numerically.

## What's inside

**Pricing models**
- Black-Scholes (closed-form, with dividends)
- Monte Carlo (antithetic variates, control variates, confidence intervals)
- Finite Difference — Crank-Nicolson (stable, second-order accurate)
- Binomial Tree — CRR (European and American)

**Greeks engine**
- Analytical: Δ, Γ, Θ, Vega, ρ, Vanna, Volga, Charm, Speed, Color
- Numerical: bump-and-reprice for all Greeks, cross-validated against analytical

**Exotic options**
- Asian (arithmetic and geometric, MC + closed-form, control variate)
- Barrier (knock-in / knock-out, up / down, analytical + MC)
- Lookback (fixed strike, floating strike)
- Digital (cash-or-nothing, asset-or-nothing)

**Calibration**
- Implied volatility (Brent + Newton-Raphson)
- Volatility smile fitting
- Heston stochastic volatility model (characteristic function, MC simulation, calibration)

## Quick start

```bash
pip install -r requirements.txt
```

```python
from options_pricing.models.base import OptionParams
from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.greeks.analytical import AnalyticalGreeks

p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")

print(BlackScholes(p).price())
print(MonteCarlo(p, n_paths=100_000).price())
print(AnalyticalGreeks(p).all_greeks())
```

## Pricing a barrier option

```python
from options_pricing.models.base import OptionParams
from options_pricing.exotics.barrier import BarrierOption

p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
barrier = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up")

print(barrier.price("analytical"))
print(barrier.price("mc"))
```

## Implied volatility

```python
from options_pricing.models.base import OptionParams
from options_pricing.calibration.implied_vol import ImpliedVolatility

p = OptionParams(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
iv = ImpliedVolatility(p).compute(market_price=5.50)
print(f"Implied vol: {iv:.2%}")
```

## Running tests

```bash
pytest tests/ -v
```

All 55 tests pass. Tests cover put-call parity, Greeks sign/magnitude checks, parity between knock-in and knock-out barriers, Monte Carlo convergence, and implied vol recovery.

## Project structure

```
options_pricing/
├── models/
│   ├── base.py              OptionParams dataclass
│   ├── black_scholes.py     Closed-form pricing + full Greeks
│   ├── monte_carlo.py       GBM simulation with variance reduction
│   ├── finite_difference.py Crank-Nicolson PDE solver
│   └── binomial_tree.py     CRR binomial model
├── greeks/
│   ├── analytical.py        Closed-form Greeks (10 Greeks)
│   ├── numerical.py         Bump-and-reprice numerical Greeks
│   └── surface.py           Greeks across spot, vol, and time
├── exotics/
│   ├── asian.py             Asian options (arith, geom, control variate)
│   ├── barrier.py           Barrier options (analytical + MC)
│   ├── lookback.py          Lookback options (fixed and floating)
│   └── digital.py           Digital options (cash-or-nothing, asset-or-nothing)
├── calibration/
│   ├── implied_vol.py       IV via Brent / Newton-Raphson
│   └── heston.py            Heston model pricing and calibration
└── utils/
    ├── visualization.py     Payoff diagrams, Greeks dashboards, vol surface
    └── market_data.py       Live data via yfinance
```

## Math

**Black-Scholes** (with continuous dividend yield q):

$$C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)$$

$$d_1 = \frac{\ln(S/K) + (r - q + \sigma^2/2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

**Monte Carlo** (GBM with antithetic variates):

$$S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \quad Z \sim \mathcal{N}(0,1)$$

**Crank-Nicolson** (second-order in time and space):

$$\frac{V^{n+1}_j - V^n_j}{\Delta t} = \frac{1}{2}\left[\mathcal{L}V^{n+1}_j + \mathcal{L}V^n_j\right]$$

where $\mathcal{L}$ is the Black-Scholes differential operator.

## License

MIT
