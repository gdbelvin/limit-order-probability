# Limit Order Fill Probability Calculator

An MCP server and Python toolkit for calculating the probability that a limit order will fill within a specified time horizon. Uses GJR-GARCH models to account for fat tails and volatility clustering in asset returns.

## Example Prompts

Once configured as an MCP server, ask Claude things like:

- *"What's the probability my VOO limit order at $600 will fill in 60 days?"*
- *"What limit price should I set for AAPL if I want a 70% chance of filling?"*
- *"Analyze my limit order for MSFT at $400"*
- *"What's the current volatility for SPY using different methods?"*

## Using with Claude Desktop (MCP Server)

### Setup

1. **Clone and install:**

   ```bash
   git clone https://github.com/gdbelvin/limit-order-probability.git
   cd limit-order-probability
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Add to your Claude Desktop config** (`~/Library/Application Support/Claude/claude_desktop_config.json` on macOS, `%APPDATA%\Claude\claude_desktop_config.json` on Windows):

   ```json
   {
     "mcpServers": {
       "limit-order-probability": {
         "command": "/full/path/to/limit-order-probability/venv/bin/python",
         "args": ["/full/path/to/limit-order-probability/mcp_server.py"]
       }
     }
   }
   ```

   **Important:** Use the full path to the virtualenv Python (`venv/bin/python`) to ensure
   all dependencies are available.

3. **Restart Claude Desktop** to load the new server.

### Available Tools

| Tool | Description |
|------|-------------|
| `calculate_fill_probability` | Calculate fill probability for a limit order |
| `find_limit_for_probability` | Find limit price for a target probability (inverse) |
| `get_volatility_estimates` | Get volatility using multiple estimation methods |
| `analyze_order` | Comprehensive analysis with all models and recommendations |

### Testing the Server

```bash
# Test with MCP inspector
npx @anthropics/mcp-inspector python mcp_server.py

# Run unit tests
python -m unittest test_mcp_server -v
```

## Why This Exists

Simple volatility-based limit order calculations assume returns follow a normal distribution. In reality:

- **Stock returns have fat tails**: Extreme moves happen 10-100x more often than normal distributions predict
- **Volatility clusters**: High volatility periods tend to persist, making fills more likely in turbulent markets
- **Asset classes differ**: Bonds mean-revert differently than equities

This toolkit implements three progressively sophisticated models:

| Model | Fill Probability | Notes |
|-------|-----------------|-------|
| GBM (baseline) | 65% | Underestimates by 30-100% |
| Student's t (v=4) | 72% | Captures fat tails |
| GJR-GARCH | 75% | Captures fat tails + volatility clustering (recommended) |

*Example: buy limit order 3% below current price, 60-day horizon.*

## Python Library Usage

### Basic Usage

```python
from fill_probability import FillProbabilityCalculator, FillDirection

calc = FillProbabilityCalculator(
    current_price=633.0,     # Current VOO price
    limit_price=614.82,      # Your limit order
    horizon_days=60,         # 60 trading days
    volatility=0.1275        # 12.75% annualized volatility
)

gbm_result = calc.gbm_closed_form()
print(f"GBM (baseline): {gbm_result.probability*100:.1f}%")

t_result = calc.student_t_monte_carlo(nu=4.0)
print(f"Student's t: {t_result.probability*100:.1f}%")
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

### Full Portfolio Analysis

```python
from analyze_orders import analyze_orders_from_csv

# Reads open_orders.csv, outputs order_analysis.csv
results = analyze_orders_from_csv('open_orders.csv')
```

### Volatility Estimation

```python
from fill_probability import VolatilityEstimator

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

| Source | Cost | Setup |
|--------|------|-------|
| [Alpaca](https://alpaca.markets) | Free (paper account) | `export ALPACA_API_KEY=... ALPACA_SECRET_KEY=...` |
| [Tiingo](https://www.tiingo.com) | $10/month | `export TIINGO_API_KEY=...` |
| yfinance | Free | No setup needed (fallback) |

## Model Details

### GBM Closed-Form

Uses the reflection principle for Brownian motion:
`P(τ_B ≤ T) = Φ(d₁) + (B/S₀)^(2μ/σ²) × Φ(d₂)`

Fast but underestimates tail events by 30-100%.

### Student's t Monte Carlo

Replaces normal innovations with Student's t-distribution (v ≈ 4 for equities). Captures fat tails but not volatility clustering.

### GJR-GARCH Monte Carlo

Models time-varying volatility with asymmetric response to negative returns:
`σ²_t = ω + (α + γ·I_{t-1})ε²_{t-1} + βσ²_{t-1}`

Where I_{t-1} = 1 for negative returns (leverage effect). Captures both fat tails and volatility clustering. Requires 100+ days of historical data.

## Limitations

- **Not a crystal ball**: These are probability estimates, not guarantees
- **Assumes liquidity**: ETF limit orders are assumed to fill if price touches limit
- **Historical data dependency**: GARCH model needs 100+ days of data
- **No intraday dynamics**: Uses daily data only

## References

### First-Passage Time & Barrier Options

The probability of a price path hitting a limit order is a *first-passage time* problem. For GBM, this has a closed-form solution using the reflection principle.

- **Harrison, J.M.** (1985). *Brownian Motion and Stochastic Flow Systems*. Wiley. — The definitive mathematical treatment of first-passage times for Brownian motion.
- **Karatzas, I. & Shreve, S.E.** (1991). *Brownian Motion and Stochastic Calculus*. Springer. — Graduate-level text covering hitting times (Chapter 2.6).
- [First-hitting-time model](https://en.wikipedia.org/wiki/First-hitting-time_model) — Wikipedia overview with key formulas.
- **Merton, R.C.** (1973). "Theory of Rational Option Pricing." *Bell Journal of Economics*, 4(1), 141-183. — Introduces barrier option pricing, closely related to limit order fills.

### Fat Tails in Asset Returns

Stock returns have "fat tails" — extreme moves occur far more often than normal distributions predict. The Student's t distribution captures this with its degrees-of-freedom parameter (ν).

- **Mandelbrot, B.** (1963). "The Variation of Certain Speculative Prices." *Journal of Business*, 36(4), 394-419. — The seminal paper showing stock returns aren't normal.
- **Cont, R.** (2001). "Empirical Properties of Asset Returns: Stylized Facts and Statistical Issues." *Quantitative Finance*, 1(2), 223-236. — Comprehensive review of return distribution properties. [PDF](https://www.quantresearch.org/Cont_stylized_facts.pdf)
- **Blattberg, R.C. & Gonedes, N.J.** (1974). "A Comparison of the Stable and Student Distributions as Statistical Models for Stock Prices." *Journal of Business*, 47(2), 244-280. — Early evidence for Student's t with ν ≈ 4-5.
- **Bollerslev, T., Todorov, V., & Li, S.Z.** (2013). "Jump Tails, Extreme Dependencies, and the Distribution of Stock Returns." *Journal of Econometrics*, 172(2), 307-324.

### GARCH Models & Volatility Clustering

Volatility clusters — high-volatility periods tend to persist. GARCH models capture this with autoregressive conditional heteroskedasticity.

- **Engle, R.F.** (1982). "Autoregressive Conditional Heteroscedasticity with Estimates of the Variance of United Kingdom Inflation." *Econometrica*, 50(4), 987-1007. — The original ARCH paper (Nobel Prize 2003).
- **Bollerslev, T.** (1986). "Generalized Autoregressive Conditional Heteroskedasticity." *Journal of Econometrics*, 31(3), 307-327. — Introduces GARCH.
- **Glosten, L.R., Jagannathan, R., & Runkle, D.E.** (1993). "On the Relation between the Expected Value and the Volatility of the Nominal Excess Return on Stocks." *Journal of Finance*, 48(5), 1779-1801. — Introduces GJR-GARCH with asymmetric leverage effect.
- **Bollerslev, T., Chou, R.Y., & Kroner, K.F.** (1992). "ARCH Modeling in Finance: A Review of the Theory and Empirical Evidence." *Journal of Econometrics*, 52(1-2), 5-59. — Comprehensive survey.
- [arch package documentation](https://arch.readthedocs.io/) — Python implementation used in this toolkit.

### Volatility Estimation

Different estimators trade off bias, efficiency, and data requirements.

- **Yang, D. & Zhang, Q.** (2000). "Drift-Independent Volatility Estimation Based on High, Low, Open, and Close Prices." *Journal of Business*, 73(3), 477-492. — The Yang-Zhang estimator is ~8x more efficient than close-to-close.
- **Parkinson, M.** (1980). "The Extreme Value Method for Estimating the Variance of the Rate of Return." *Journal of Business*, 53(1), 61-65. — High-low range estimator.
- **Garman, M.B. & Klass, M.J.** (1980). "On the Estimation of Security Price Volatilities from Historical Data." *Journal of Business*, 53(1), 67-78. — OHLC-based estimator.
- **RiskMetrics Technical Document** (1996). J.P. Morgan/Reuters. — Introduces EWMA with λ=0.94 for daily data. [PDF](https://www.msci.com/documents/10199/5915b101-4206-4ba0-aee2-3449d5c7e95a)

### Monte Carlo Methods in Finance

- **Glasserman, P.** (2003). *Monte Carlo Methods in Financial Engineering*. Springer. — The standard reference for Monte Carlo in finance.
- **Jäckel, P.** (2002). *Monte Carlo Methods in Finance*. Wiley. — Practical implementation guide.

### Books for Practitioners

- **Sinclair, E.** (2010). *Option Trading: Pricing and Volatility Strategies and Techniques*. Wiley. — Practical volatility trading.
- **Taleb, N.N.** (2007). *The Black Swan*. Random House. — Accessible introduction to fat tails and their consequences.
- **Alexander, C.** (2008). *Market Risk Analysis, Volume II: Practical Financial Econometrics*. Wiley. — GARCH modeling in practice.

## License

MIT License - Use freely for personal investment decisions.
