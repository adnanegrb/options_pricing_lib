# Options Pricing Library

![Language](https://img.shields.io/badge/Language-Python-3776AB) ![Topic](https://img.shields.io/badge/Topic-Derivatives%20Pricing-8A2BE2) ![Models](https://img.shields.io/badge/Models-Black--Scholes%20%7C%20Monte%20Carlo%20%7C%20PDE%20%7C%20Binomial-D2691E) ![Greeks](https://img.shields.io/badge/Greeks-10%20Analytical-2E8B57) ![Tests](https://img.shields.io/badge/Tests-55%20Passing-4C1)

This library prices vanilla and exotic options using four independent numerical methods, so every price and every Greek can be checked against a method that does not share the same assumptions or the same source of numerical error. Correctness is treated as the actual deliverable here, not a side effect of the pricing being fast or the code being clean.

## What it does

Four pricing models are implemented from the ground up: Black-Scholes in closed form with continuous dividends, Monte Carlo simulation with antithetic and control variates for variance reduction, a Crank-Nicolson finite difference solver for the Black-Scholes PDE, and a Cox-Ross-Rubinstein binomial tree covering both European and American exercise. Every model is meant to agree with the others on a vanilla option to within numerical tolerance, and the test suite checks exactly that rather than assuming it.

Ten Greeks are computed analytically, delta, gamma, theta, vega, rho, vanna, volga, charm, speed, and color, and every one of them is cross-validated against a numerical bump-and-reprice calculation, so a sign error or a closed-form typo in one of the less common higher-order Greeks does not go unnoticed.

Four families of exotic options are priced on top of the same underlying framework: Asian options, arithmetic and geometric averaging, with a closed-form solution for the geometric case and a control variate for the arithmetic one; barrier options, knock-in and knock-out, up and down, priced both analytically and by simulation; lookback options with fixed and floating strikes; and digital options, both cash-or-nothing and asset-or-nothing.

Implied volatility is recovered from a market price using Brent's method with a Newton-Raphson refinement step, and the same machinery supports fitting a full volatility smile. A Heston stochastic volatility model is implemented as well, priced through its characteristic function and by direct simulation, with a calibration routine to fit it to market quotes.

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

Pricing a barrier option looks the same, just with the extra parameters a barrier needs:

```python
from options_pricing.models.base import OptionParams
from options_pricing.exotics.barrier import BarrierOption

p = OptionParams(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
barrier = BarrierOption(p, barrier=120, barrier_type="knock_out", direction="up")

print(barrier.price("analytical"))
print(barrier.price("mc"))
```

Recovering implied volatility from a market price:

```python
from options_pricing.models.base import OptionParams
from options_pricing.calibration.implied_vol import ImpliedVolatility

p = OptionParams(S=100, K=105, T=0.5, r=0.05, sigma=0.2)
iv = ImpliedVolatility(p).compute(market_price=5.50)
print(f"Implied vol: {iv:.2%}")
```

## Validating the library

```bash
pytest tests/ -v
```

All 55 tests currently pass. They check put-call parity across all four pricing models, the sign and magnitude of every Greek against known option-price intuition, agreement between knock-in and knock-out barrier prices with a vanilla option, Monte Carlo convergence as the number of paths grows, and that implied volatility recovery returns the volatility a price was originally generated from.

## Project structure

```
options_pricing/
├── models/
│   ├── base.py                 OptionParams dataclass shared by every model
│   ├── black_scholes.py        Closed-form pricing and full Greeks
│   ├── monte_carlo.py          GBM simulation with variance reduction
│   ├── finite_difference.py    Crank-Nicolson PDE solver
│   └── binomial_tree.py        Cox-Ross-Rubinstein binomial model
├── greeks/
│   ├── analytical.py           Closed-form Greeks, ten in total
│   ├── numerical.py            Bump-and-reprice numerical Greeks
│   └── surface.py              Greeks across spot, volatility, and time
├── exotics/
│   ├── asian.py                Arithmetic and geometric Asian options
│   ├── barrier.py               Barrier options, analytical and Monte Carlo
│   ├── lookback.py              Fixed and floating strike lookback options
│   └── digital.py               Cash-or-nothing and asset-or-nothing digitals
├── calibration/
│   ├── implied_vol.py          Implied volatility via Brent and Newton-Raphson
│   └── heston.py                Heston model pricing and calibration
└── utils/
    ├── visualization.py        Payoff diagrams, Greeks dashboards, vol surface
    └── market_data.py           Live market data via yfinance
```

## Mathematical formulation

Every formula below is implemented exactly as stated in the module referenced alongside it, so the code can be checked against the mathematics directly rather than taken on faith.

**Black-Scholes.** With a continuous dividend yield $q$, the price of a European call is

$$
C = S e^{-qT} N(d_1) - K e^{-rT} N(d_2)
$$

$$
d_1 = \frac{\ln(S/K) + \left(r - q + \tfrac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
$$

implemented in `models/black_scholes.py`, with the put price following from put-call parity.

**Monte Carlo simulation.** Terminal prices are simulated under the risk-neutral measure using geometric Brownian motion,

$$
S_T = S_0 \exp\left[\left(r - q - \frac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \qquad Z \sim \mathcal{N}(0,1)
$$

with antithetic variates, pairing each $Z$ with $-Z$, and a control variate against the closed-form Black-Scholes price used to reduce the variance of the resulting estimator, in `models/monte_carlo.py`.

**Crank-Nicolson finite difference.** The Black-Scholes PDE is discretized with the Crank-Nicolson scheme, averaging the implicit and explicit finite-difference operators to achieve second-order accuracy in both time and space,

$$
\frac{V_j^{n+1} - V_j^n}{\Delta t} = \frac{1}{2}\left[ \mathcal{L}V_j^{n+1} + \mathcal{L}V_j^n \right]
$$

where $\mathcal{L}$ is the Black-Scholes differential operator

$$
\mathcal{L}V = \frac{1}{2}\sigma^2 S^2 \frac{\partial^2 V}{\partial S^2} + (r-q) S \frac{\partial V}{\partial S} - rV
$$

implemented in `models/finite_difference.py`.

**Binomial tree.** At each node, the underlying moves up by a factor $u$ or down by $d$, with the risk-neutral probability

$$
u = e^{\sigma\sqrt{\Delta t}}, \qquad d = \frac{1}{u}, \qquad p = \frac{e^{(r-q)\Delta t} - d}{u - d}
$$

and the option value at each node is computed by backward induction from the terminal payoff, with early exercise checked at every step for American options, in `models/binomial_tree.py`.

**Analytical Greeks.** Each Greek is the corresponding partial derivative of the option price. For a call, in terms of the standard normal density $\phi$ and CDF $N$,

$$
\Delta = e^{-qT}N(d_1), \qquad \Gamma = \frac{e^{-qT}\phi(d_1)}{S\sigma\sqrt{T}}, \qquad \text{Vega} = S e^{-qT}\phi(d_1)\sqrt{T}
$$

$$
\Theta = -\frac{S e^{-qT}\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}N(d_2) + qSe^{-qT}N(d_1)
$$

with the second-order Greeks, vanna, volga, charm, speed, and color, obtained by differentiating these expressions once more, in `greeks/analytical.py`. The numerical counterpart in `greeks/numerical.py` recomputes each Greek by finite difference on the price itself,

$$
\frac{\partial V}{\partial S} \approx \frac{V(S+h) - V(S-h)}{2h}
$$

and the test suite requires the analytical and numerical values to agree to within a small tolerance.

**Asian options.** For a geometric average, the payoff depends on $G_T = \left(\prod_{i=1}^n S_{t_i}\right)^{1/n}$, which is itself lognormal under GBM, giving a closed-form Black-Scholes-type price with an adjusted volatility and drift. The arithmetic average $A_T = \frac{1}{n}\sum_{i=1}^n S_{t_i}$ has no closed form, and is priced by Monte Carlo using the geometric-average price as a control variate, implemented in `exotics/asian.py`.

**Barrier options.** For a continuously monitored barrier $H$, the analytical price uses the reflection principle for Brownian motion to express the barrier price in terms of vanilla Black-Scholes prices evaluated at both the strike and the barrier level, in `exotics/barrier.py`, with the Monte Carlo implementation checking the barrier condition at each simulated time step and the two methods validated against each other in the test suite.

**Implied volatility.** Given a market price $C_{\text{mkt}}$, implied volatility solves

$$
C_{\text{BS}}(\sigma) = C_{\text{mkt}}
$$

for $\sigma$, using Brent's method to bracket a root and a Newton-Raphson step using vega as the derivative to refine it, in `calibration/implied_vol.py`.

**Heston model.** The stochastic volatility process is

$$
dS_t = (r-q)S_t\,dt + \sqrt{v_t}\,S_t\,dW_t^S, \qquad dv_t = \kappa(\theta - v_t)\,dt + \xi\sqrt{v_t}\,dW_t^v, \qquad d\langle W^S, W^v\rangle_t = \rho\,dt
$$

priced through its characteristic function using the Heston semi-closed-form and by direct Euler simulation of the variance process, with the calibration routine fitting $(\kappa, \theta, \xi, \rho, v_0)$ to a set of market quotes by minimizing pricing error, in `calibration/heston.py`.

## License

MIT
