# Coffee Option Pricing Analysis


Project Overview

This project implements a professional-grade quantitative finance framework for pricing and analyzing commodity options, using coffee futures as a practical case study. It combines classical option pricing theory with advanced computational methods to provide accurate valuations, comprehensive risk metrics, and stress-tested scenario analysis for traders, risk managers, and quantitative analysts.

Core Components

The Cost of Carry Model establishes the theoretical forward price of coffee by accounting for storage costs, interest rates, and time to maturity. This forward price serves as the foundation for all subsequent pricing calculations and reflects the true economic cost of holding the commodity.

The Black-Scholes Model extends this framework to calculate fair option prices for both calls and puts. Beyond valuations, it computes the Greeks—Delta (price sensitivity), Gamma (delta acceleration), Vega (volatility sensitivity), Theta (time decay), and Rho (interest rate sensitivity)—providing traders with actionable metrics to manage position risk and determine optimal hedging strategies.

The Monte Carlo Simulation validates theoretical prices through computational methods, generating 100,000+ realistic price paths using geometric Brownian motion. It calculates probability distributions of future spot prices, determines the likelihood of options expiring in-the-money, and provides confidence intervals around price estimates, adding empirical rigor to the analysis.

The Scenario Analysis evaluates how market shocks impact option valuations across six realistic scenarios: supply disruptions (drought), supply surplus (bumper crop), demand surges, demand weakness, geopolitical crises, and base case conditions. Each scenario adjusts spot prices and volatility to show traders how their positions would perform under stress.

The Visualization Engine synthesizes all outputs into publication-ready charts: Monte Carlo price paths with confidence bands, final price distributions, Greeks sensitivity analysis, and side-by-side model comparisons, enabling stakeholders to understand assumptions, validate results, and make informed decisions.
