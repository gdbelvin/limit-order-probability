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

# Run MCP server tests
python -m unittest test_mcp_server -v

# Run all tests
python -m unittest discover -v
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

## MCP Server

The project includes an MCP server (`mcp_server.py`) that exposes the calculator as tools for Claude and other MCP clients.

### Running the Server

```bash
# Run standalone
python mcp_server.py

# Test with MCP inspector
npx @anthropics/mcp-inspector python mcp_server.py
```

### Claude Desktop Configuration

Add to `~/Library/Application Support/Claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "limit-order-probability": {
      "command": "/path/to/limit-order-probability/venv/bin/python",
      "args": ["/path/to/limit-order-probability/mcp_server.py"]
    }
  }
}
```

Use the full path to the virtualenv Python to ensure dependencies are available.

### Available Tools

1. **calculate_fill_probability** - Calculate fill probability for a limit order
   - Parameters: `symbol`, `limit_price`, `horizon_days` (default 60), `model` ("garch"/"student_t"/"gbm"/"all")
   - Returns: fill probability, confidence interval, model details, recommendation

2. **find_limit_for_probability** - Find limit price for a target fill probability (inverse function)
   - Parameters: `symbol`, `direction` ("buy"/"sell"), `target_probability` (default 0.70), `horizon_days`
   - Returns: recommended limit price, distance from current price

3. **get_volatility_estimates** - Get volatility estimates using multiple methods
   - Parameters: `symbol`, `lookback_days` (default 252)
   - Returns: close-to-close, Yang-Zhang, EWMA, GARCH volatility estimates

4. **analyze_order** - Comprehensive order analysis with all models
   - Parameters: `symbol`, `limit_price`, `horizon_days`
   - Returns: all model results, volatility data, status (GOOD/MARGINAL/TOO_AGGRESSIVE), recommendations

### Design Notes

- 5-minute cache for market data to avoid excessive API calls
- GARCH model used by default; falls back to Student's t if insufficient data (<100 days)
- All tools fetch market data automatically via DataFetcher (yfinance by default)
