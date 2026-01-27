# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A toolkit for calculating the probability that a limit order will fill within a specified time horizon (30-60 days). Implements three progressively sophisticated models to account for fat tails and volatility clustering in asset returns.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run portfolio analysis (reads from open_orders.csv, outputs to order_analysis.csv)
python analyze_orders.py

# Run with custom input/output files
python analyze_orders.py my_orders.csv my_results.csv

# Run fill probability calculator example
python fill_probability.py

# Test data fetcher
python data_fetcher.py

# Run unit tests
python -m unittest test_fill_probability -v
```

## Architecture

### Core Modules

- **fill_probability.py**: Core probability calculations
  - `FillProbabilityCalculator`: Main class with three models (GBM closed-form, Student's t Monte Carlo, GJR-GARCH Monte Carlo)
  - `VolatilityEstimator`: Multiple volatility estimation methods (close-to-close, Yang-Zhang OHLC, EWMA, GJR-GARCH)
  - `calculate_p70_limit()`: Inverse function to find limit price for target probability

- **data_fetcher.py**: Multi-source data fetching
  - `DataFetcher`: Unified interface for Alpaca, Tiingo, and yfinance
  - `ImpliedVolatilityFetcher`: Scrapes IV data from Option Strategist
  - `fetch_portfolio_data()`: Convenience function for batch fetching

- **analyze_orders.py**: Portfolio analysis script
  - `CURRENT_ORDERS`: Dictionary of open limit orders to analyze
  - `ASSET_CLASSES`: Categorization affecting model parameters (equities use nu=4, bonds use nu=6 for Student's t)
  - `analyze_portfolio_orders()`: Full analysis pipeline with recommendations

### Key Design Patterns

- Models return `FillProbabilityResult` dataclass with probability, confidence interval, and model details
- Volatility estimators return annualized values (multiply daily by sqrt(252))
- GARCH model expects returns in percentage terms (multiply decimal by 100)
- Data sources are tried in priority order: Alpaca > Tiingo > yfinance

### Environment Variables

```bash
ALPACA_API_KEY      # Alpaca API key (free paper account)
ALPACA_SECRET_KEY   # Alpaca secret key
TIINGO_API_KEY      # Tiingo API key ($10/month)
```

## Model Notes

- GBM closed-form underestimates fill probability by 30-100% due to ignoring fat tails
- Student's t with nu=4 captures fat tails but not volatility clustering
- GJR-GARCH is recommended - captures both effects via time-varying volatility with leverage effect
- GARCH requires 100+ days of historical data for reliable parameter estimation
