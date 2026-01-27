# Limit Order Fill Probability Calculator

A sophisticated toolkit for calculating the probability that a limit order will fill within a specified time horizon (30-60 days). Built for advanced retail investors who want rigorous but practical approaches to limit order pricing.

## Why This Exists

Simple volatility-based limit order calculations (like the basic P70 formula) assume returns follow a normal distribution. In reality:

- **Stock returns have fat tails**: Extreme moves happen 10-100x more often than normal distributions predict
- **Volatility clusters**: High volatility periods tend to persist, making fills more likely in turbulent markets
- **Asset classes differ**: Bonds mean-revert differently than equities

This toolkit implements **three progressively sophisticated models**:

1. **GBM Closed-Form**: Fast baseline (but underestimates fill probabilities by 30-100%)
2. **Student's t Monte Carlo**: Captures fat tails
3. **GJR-GARCH Monte Carlo**: Captures both fat tails AND volatility clustering (recommended)

## Installation

```bash
# Clone or download the repository
cd limit_order_probability

# Install dependencies
pip install -r requirements.txt

# For the recommended data source (free with Alpaca paper account):
pip install alpaca-py

# Or for the best value paid option ($10/month):
pip install tiingo
```

## Quick Start

### Basic Usage

```python
from fill_probability import FillProbabilityCalculator, FillDirection

# Create calculator
calc = FillProbabilityCalculator(
    current_price=633.0,     # Current VOO price
    limit_price=614.82,      # Your limit order
    horizon_days=60,         # 60 trading days
    volatility=0.1275        # 12.75% annualized volatility
)

# Get fill probability using different models
gbm_result = calc.gbm_closed_form()
print(f"GBM (baseline): {gbm_result.probability*100:.1f}%")

t_result = calc.student_t_monte_carlo(nu=4.0)
print(f"Student's t: {t_result.probability*100:.1f}%")
```

### Full Portfolio Analysis

```python
from analyze_orders import analyze_portfolio_orders

# Analyze all orders (fetches live data)
results = analyze_portfolio_orders(
    horizon_days=60,
    preferred_source='yfinance'  # or 'alpaca', 'tiingo'
)

# Export to CSV
from analyze_orders import export_to_csv
export_to_csv(results, 'my_analysis.csv')
```

### Calculate P70 Limit Price

```python
from fill_probability import calculate_p70_limit, FillDirection

# What limit price gives 70% fill probability?
p70_price = calculate_p70_limit(
    current_price=633.0,
    volatility=0.1275,
    horizon_days=60,
    direction=FillDirection.BUY
)
print(f"P70 limit: ${p70_price:.2f}")
```

## Volatility Estimation

The toolkit includes multiple volatility estimators:

```python
from fill_probability import VolatilityEstimator
import pandas as pd

# Assuming you have OHLC data in a DataFrame
# with columns: open, high, low, close

# Standard close-to-close (baseline)
vol_cc = VolatilityEstimator.close_to_close(returns, window=60)

# Yang-Zhang (8x more efficient than close-to-close)
vol_yz = VolatilityEstimator.yang_zhang(ohlc_df, window=60)

# EWMA (captures recent volatility changes)
vol_ewma = VolatilityEstimator.ewma(returns, lambda_param=0.97)

# GJR-GARCH (best for forecasting)
vol_garch, details = VolatilityEstimator.fit_garch(returns_pct, horizon=30)
```

## Data Sources

### Alpaca (Recommended - Free)

Sign up for a free paper trading account at [alpaca.markets](https://alpaca.markets):

```bash
export ALPACA_API_KEY="your_key"
export ALPACA_SECRET_KEY="your_secret"
```

```python
from data_fetcher import DataFetcher

fetcher = DataFetcher(preferred_source='alpaca')
df = fetcher.get_ohlc('VOO', days=252)
```

### Tiingo (Best Value - $10/month)

Sign up at [tiingo.com](https://www.tiingo.com):

```bash
export TIINGO_API_KEY="your_key"
```

### yfinance (Fallback - Free)

No API key needed, but less reliable:

```python
fetcher = DataFetcher(preferred_source='yfinance')
```

## Model Details

### GBM Closed-Form

Uses the reflection principle for Brownian motion:

```
P(τ_B ≤ T) = Φ(d₁) + (B/S₀)^(2μ/σ²) × Φ(d₂)
```

**Pros**: Fast, closed-form
**Cons**: Underestimates tail events by 30-100%

### Student's t Monte Carlo

Replaces normal innovations with Student's t-distribution (ν ≈ 4 for equities).

**Pros**: Captures fat tails (excess kurtosis)
**Cons**: Doesn't capture volatility clustering

### GJR-GARCH Monte Carlo

Models time-varying volatility with asymmetric response to negative returns:

```
σ²_t = ω + (α + γ·I_{t-1})ε²_{t-1} + βσ²_{t-1}
```

Where I_{t-1} = 1 for negative returns (leverage effect).

**Pros**: Captures both fat tails AND volatility clustering
**Cons**: Requires sufficient historical data (100+ days)

## File Structure

```
limit_order_probability/
├── fill_probability.py   # Core probability calculations
├── data_fetcher.py       # Multi-source data fetching
├── analyze_orders.py     # Portfolio analysis script
├── requirements.txt      # Dependencies
└── README.md            # This file
```

## Typical Results Comparison

For a buy limit order 3% below current price with 60-day horizon:

| Model | Fill Probability |
|-------|-----------------|
| GBM (baseline) | 65% |
| Student's t (ν=4) | 72% |
| GJR-GARCH | 75% |

The difference grows larger for orders further from current price.

## Limitations

- **Not a crystal ball**: These are probability estimates, not guarantees
- **Assumes liquidity**: ETF limit orders are assumed to fill if price touches limit
- **Historical data dependency**: GARCH model needs 100+ days of data
- **No intraday dynamics**: Uses daily data only

## References

- First-passage time problems: [Wikipedia](https://en.wikipedia.org/wiki/First-hitting-time_model)
- GARCH models: Bollerslev (1986), Glosten, Jagannathan & Runkle (1993)
- Fat tails in returns: Mandelbrot, Cont (2001)
- Yang-Zhang volatility: Yang & Zhang (2000)

## License

MIT License - Use freely for personal investment decisions.
