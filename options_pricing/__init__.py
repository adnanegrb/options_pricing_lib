"""
Options Pricing Library
=======================
A professional-grade library for pricing vanilla and exotic options using:
- Black-Scholes (closed-form)
- Monte Carlo simulation (with variance reduction)
- Finite Difference Methods (Explicit, Implicit, Crank-Nicolson)
- Binomial Trees (CRR)

Includes full Greeks engine (analytical + numerical), exotic options,
implied volatility calibration, and visualization tools.

Author: [Your Name]
License: MIT
"""

__version__ = "1.0.0"
__author__ = "Your Name"

from options_pricing.models.black_scholes import BlackScholes
from options_pricing.models.monte_carlo import MonteCarlo
from options_pricing.models.finite_difference import FiniteDifference
from options_pricing.models.binomial_tree import BinomialTree
from options_pricing.greeks.analytical import AnalyticalGreeks
from options_pricing.greeks.numerical import NumericalGreeks

__all__ = [
    "BlackScholes",
    "MonteCarlo",
    "FiniteDifference",
    "BinomialTree",
    "AnalyticalGreeks",
    "NumericalGreeks",
]
