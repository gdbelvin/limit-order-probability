"""
Limit Order Analyzer
====================

Comprehensive analysis of limit orders for a portfolio.

This script:
1. Fetches current price data for all ETFs
2. Calculates volatility using multiple methods
3. Runs fill probability models (GBM, Student's t, GARCH)
4. Generates P70 limit price recommendations
5. Outputs a summary report

Usage:
    python analyze_orders.py

Or in Python:
    from analyze_orders import analyze_portfolio_orders
    results = analyze_portfolio_orders()
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import warnings

from fill_probability import (
    FillProbabilityCalculator,
    VolatilityEstimator,
    FillDirection,
    calculate_p70_limit,
)
from data_fetcher import DataFetcher, fetch_portfolio_data


# Orders loaded from CSV file (set by load_orders_from_csv)
CURRENT_ORDERS = {}

# Default input file
DEFAULT_ORDERS_FILE = 'open_orders.csv'

# Asset class categorization for different volatility characteristics
ASSET_CLASSES = {
    'equity': ['VOO', 'VBR', 'VSS', 'VXUS', 'VWO'],
    'bond': ['VGIT', 'VGSH', 'VGLT', 'VTIP', 'BNDX'],
    'alternatives': ['DBMF', 'CAOS'],
}


def get_asset_class(symbol: str) -> str:
    """Determine asset class for a symbol"""
    for asset_class, symbols in ASSET_CLASSES.items():
        if symbol in symbols:
            return asset_class
    return 'unknown'


def calculate_all_volatilities(symbol: str, 
                                ohlc: pd.DataFrame, 
                                returns_pct: pd.Series) -> Dict:
    """
    Calculate volatility using all available methods.
    
    Args:
        symbol: Ticker symbol
        ohlc: OHLC DataFrame
        returns_pct: Returns in percentage terms
        
    Returns:
        Dictionary of volatility estimates
    """
    returns_decimal = returns_pct / 100
    
    result = {
        'symbol': symbol,
        'asset_class': get_asset_class(symbol)
    }
    
    # Close-to-close
    result['close_to_close_60d'] = VolatilityEstimator.close_to_close(returns_decimal, 60)
    result['close_to_close_20d'] = VolatilityEstimator.close_to_close(returns_decimal, 20)
    
    # Yang-Zhang (requires OHLC)
    try:
        result['yang_zhang_60d'] = VolatilityEstimator.yang_zhang(ohlc, 60)
    except Exception:
        result['yang_zhang_60d'] = None
    
    # EWMA
    result['ewma_94'] = VolatilityEstimator.ewma(returns_decimal, lambda_param=0.94)
    result['ewma_97'] = VolatilityEstimator.ewma(returns_decimal, lambda_param=0.97)
    
    # GARCH
    try:
        garch_vol, garch_details = VolatilityEstimator.fit_garch(returns_pct, horizon=30)
        result['garch_30d'] = garch_vol
        result['garch_nu'] = garch_details['nu']
        result['garch_persistence'] = garch_details['persistence']
        result['garch_alpha'] = garch_details['alpha']
        result['garch_beta'] = garch_details['beta']
        result['garch_gamma'] = garch_details['gamma']
        result['garch_omega'] = garch_details['omega']
    except Exception as e:
        result['garch_30d'] = None
        result['garch_error'] = str(e)
    
    # Select best estimate
    # Priority: GARCH > Yang-Zhang > EWMA (λ=0.97) > Close-to-close
    if result.get('garch_30d') is not None:
        result['best_estimate'] = result['garch_30d']
        result['best_method'] = 'garch'
    elif result.get('yang_zhang_60d') is not None:
        result['best_estimate'] = result['yang_zhang_60d']
        result['best_method'] = 'yang_zhang'
    else:
        result['best_estimate'] = result['ewma_97']
        result['best_method'] = 'ewma_97'
    
    return result


class InsufficientDataError(Exception):
    """Raised when historical data is insufficient for GARCH analysis."""
    pass


def analyze_single_order(symbol: str,
                         current_price: float,
                         limit_price: float,
                         volatility: float,
                         returns_pct: Optional[pd.Series],
                         horizon_days: int = 60,
                         require_garch: bool = True) -> Dict:
    """
    Analyze a single limit order using multiple models.

    Args:
        symbol: Ticker symbol
        current_price: Current market price
        limit_price: Current limit order price
        volatility: Volatility estimate to use
        returns_pct: Historical returns (for GARCH model)
        horizon_days: Time horizon in trading days
        require_garch: If True, raise error if GARCH cannot be run

    Returns:
        Analysis results dictionary

    Raises:
        InsufficientDataError: If require_garch=True and insufficient data
    """
    result = {
        'symbol': symbol,
        'current_price': current_price,
        'limit_price': limit_price,
        'volatility': volatility,
        'horizon_days': horizon_days,
    }

    # Calculate basic metrics
    direction = FillDirection.BUY if limit_price < current_price else FillDirection.SELL
    result['direction'] = direction.value
    result['distance_pct'] = (current_price - limit_price) / current_price * 100
    result['distance_dollars'] = current_price - limit_price

    # Validate we have sufficient data for GARCH
    if require_garch:
        if returns_pct is None:
            raise InsufficientDataError(
                f"{symbol}: No historical returns data available. "
                "GARCH requires historical data."
            )
        if len(returns_pct) < 100:
            raise InsufficientDataError(
                f"{symbol}: Only {len(returns_pct)} days of data available. "
                "GARCH requires at least 100 days of historical data."
            )

    # Initialize calculator
    calc = FillProbabilityCalculator(
        current_price=current_price,
        limit_price=limit_price,
        horizon_days=horizon_days,
        volatility=volatility
    )

    # Model 1: GBM Closed-Form (baseline - for reference only)
    gbm = calc.gbm_closed_form()
    result['prob_gbm'] = gbm.probability

    # Model 2: Student's t Monte Carlo (for reference only)
    asset_class = get_asset_class(symbol)
    nu = 4.0 if asset_class == 'equity' else 6.0

    t_result = calc.student_t_monte_carlo(nu=nu, n_sims=50000, seed=42)
    result['prob_student_t'] = t_result.probability
    result['prob_student_t_ci'] = t_result.confidence_interval

    # Model 3: GARCH Monte Carlo (PRIMARY MODEL)
    if returns_pct is not None and len(returns_pct) >= 100:
        garch = calc.garch_monte_carlo(returns_pct, n_sims=50000, seed=42)
        result['prob_garch'] = garch.probability
        result['prob_garch_ci'] = garch.confidence_interval
        result['garch_nu'] = garch.details['garch_params']['nu']
    else:
        result['prob_garch'] = None

    # Best probability estimate - ONLY use GARCH when require_garch=True
    if result.get('prob_garch') is not None:
        result['prob_best'] = result['prob_garch']
        result['prob_best_model'] = 'GARCH'
    elif require_garch:
        # This shouldn't happen due to validation above, but be defensive
        raise InsufficientDataError(
            f"{symbol}: GARCH analysis failed but require_garch=True"
        )
    else:
        result['prob_best'] = result['prob_student_t']
        result['prob_best_model'] = 'Student-t (FALLBACK - NO HISTORICAL DATA)'

    # Calculate P70 target price
    p70_price = calculate_p70_limit(
        current_price, volatility, horizon_days, direction
    )
    result['p70_limit'] = p70_price
    result['p70_distance_pct'] = abs(current_price - p70_price) / current_price * 100

    # Adjustment needed
    if direction == FillDirection.BUY:
        result['adjustment'] = p70_price - limit_price
    else:
        result['adjustment'] = limit_price - p70_price

    # Status
    if result['prob_best'] >= 0.65:
        result['status'] = 'GOOD'
    elif result['prob_best'] >= 0.45:
        result['status'] = 'MARGINAL'
    else:
        result['status'] = 'TOO_AGGRESSIVE'

    return result


def analyze_portfolio_orders(horizon_days: int = 60,
                             preferred_source: str = 'yfinance',
                             require_garch: bool = True,
                             **kwargs) -> Dict:
    """
    Analyze all open orders in the portfolio.

    Args:
        horizon_days: Time horizon for fill probability
        preferred_source: Data source ('alpaca', 'tiingo', 'yfinance')
        require_garch: If True (default), require GARCH analysis for all orders.
                      Will error if insufficient historical data.
        **kwargs: Additional arguments for DataFetcher

    Returns:
        Dictionary with analysis results for all orders

    Raises:
        InsufficientDataError: If require_garch=True and any symbol lacks data
    """
    symbols = list(CURRENT_ORDERS.keys())
    
    print("=" * 70)
    print("LIMIT ORDER FILL PROBABILITY ANALYSIS")
    print(f"Horizon: {horizon_days} trading days")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)
    
    # Fetch data
    print("\nFetching market data...")
    try:
        data = fetch_portfolio_data(
            symbols, 
            days=252,  # 1 year of data for volatility estimation
            preferred_source=preferred_source,
            **kwargs
        )
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

    # Calculate volatilities first (needed for data summary)
    print("\nCalculating volatility estimates...")
    volatilities = {}
    for symbol in symbols:
        if symbol in data['ohlc'] and symbol in data['returns_pct']:
            vol_data = calculate_all_volatilities(
                symbol,
                data['ohlc'][symbol],
                data['returns_pct'][symbol]
            )
            volatilities[symbol] = vol_data

    # Print comprehensive data summary for validation
    print("\n" + "=" * 120)
    print("DATA SUMMARY & GARCH PARAMETERS")
    print("=" * 120)
    print(f"{'Symbol':<6} {'Days':>5} {'Current':>9} {'Yr Low':>9} {'Yr High':>9} {'Max Drop':>9} "
          f"{'Vol %':>7} {'ν':>6} {'α':>6} {'β':>6} {'γ':>6} {'Persist':>7}")
    print("-" * 120)
    for symbol in symbols:
        ohlc = data['ohlc'].get(symbol)
        vol_data = volatilities.get(symbol, {})
        if ohlc is not None and len(ohlc) > 0:
            days = len(ohlc)
            current = ohlc['close'].iloc[-1]
            yr_low = ohlc['low'].min()
            yr_high = ohlc['high'].max()
            daily_returns = ohlc['close'].pct_change()
            max_drop = daily_returns.min() * 100

            # Volatility and GARCH params
            vol_pct = vol_data.get('best_estimate', 0) * 100
            nu = vol_data.get('garch_nu', 0)
            persistence = vol_data.get('garch_persistence', 0)

            # Get detailed GARCH params if available
            alpha = vol_data.get('garch_alpha', 0)
            beta = vol_data.get('garch_beta', 0)
            gamma = vol_data.get('garch_gamma', 0)

            print(f"{symbol:<6} {days:>5} ${current:>8.2f} ${yr_low:>8.2f} ${yr_high:>8.2f} {max_drop:>8.1f}% "
                  f"{vol_pct:>6.1f}% {nu:>6.2f} {alpha:>6.3f} {beta:>6.3f} {gamma:>6.3f} {persistence:>6.3f}")
        else:
            print(f"{symbol:<6} NO DATA")
    print("-" * 120)
    print("ν=degrees of freedom (fat tails), α=ARCH effect, β=GARCH persistence, γ=leverage (asymmetry)")
    print("=" * 120)

    # Analyze orders
    print("\nAnalyzing orders...")
    analyses = {}
    errors = []

    for symbol, order in CURRENT_ORDERS.items():
        # Validate we have price data
        if symbol not in data['current_prices']:
            error_msg = f"{symbol}: No current price data available"
            if require_garch:
                errors.append(error_msg)
                continue
            else:
                print(f"  WARNING: {error_msg} - skipping")
                continue

        # Validate we have volatility data (not using defaults!)
        if symbol not in volatilities:
            error_msg = f"{symbol}: No volatility data available"
            if require_garch:
                errors.append(error_msg)
                continue
            else:
                print(f"  WARNING: {error_msg} - skipping")
                continue

        vol_data = volatilities[symbol]
        if vol_data.get('best_estimate') is None:
            error_msg = f"{symbol}: Could not estimate volatility"
            if require_garch:
                errors.append(error_msg)
                continue
            else:
                print(f"  WARNING: {error_msg} - skipping")
                continue

        current_price = data['current_prices'][symbol]
        limit_price = order['limit']
        vol = vol_data['best_estimate']
        returns_pct = data['returns_pct'].get(symbol)

        try:
            analysis = analyze_single_order(
                symbol=symbol,
                current_price=current_price,
                limit_price=limit_price,
                volatility=vol,
                returns_pct=returns_pct,
                horizon_days=horizon_days,
                require_garch=require_garch
            )

            # Add order details
            analysis['shares'] = order['shares']
            analysis['account'] = order['account']
            analysis['order_value'] = order['limit'] * order['shares']

            analyses[symbol] = analysis

        except InsufficientDataError as e:
            errors.append(str(e))
            continue

    # Fail if any errors and require_garch is True
    if errors and require_garch:
        print("\n" + "=" * 70)
        print("ERROR: Insufficient data for GARCH analysis")
        print("=" * 70)
        for error in errors:
            print(f"  - {error}")
        print("\nTo proceed without GARCH (NOT RECOMMENDED), use require_garch=False")
        raise InsufficientDataError(
            f"Insufficient data for {len(errors)} symbol(s). See errors above."
        )
    
    # Generate summary report
    print("\n" + "=" * 70)
    print("SUMMARY REPORT")
    print("=" * 70)
    
    # Group by status
    good = [s for s, a in analyses.items() if a['status'] == 'GOOD']
    marginal = [s for s, a in analyses.items() if a['status'] == 'MARGINAL']
    aggressive = [s for s, a in analyses.items() if a['status'] == 'TOO_AGGRESSIVE']
    
    print(f"\n✅ GOOD ({len(good)}): {', '.join(good) if good else 'None'}")
    print(f"⚠️  MARGINAL ({len(marginal)}): {', '.join(marginal) if marginal else 'None'}")
    print(f"❌ TOO AGGRESSIVE ({len(aggressive)}): {', '.join(aggressive) if aggressive else 'None'}")
    
    # Detailed table
    print("\n" + "-" * 70)
    print(f"{'Symbol':<6} {'Current':>8} {'Limit':>8} {'Dist%':>7} {'Fill%':>7} {'P70':>8} {'Adj':>7} {'Status':<12}")
    print("-" * 70)
    
    for symbol in symbols:
        if symbol not in analyses:
            continue
        a = analyses[symbol]
        print(f"{symbol:<6} ${a['current_price']:>7.2f} ${a['limit_price']:>7.2f} "
              f"{a['distance_pct']:>6.2f}% {a['prob_best']*100:>6.1f}% "
              f"${a['p70_limit']:>7.2f} ${a['adjustment']:>+6.2f} {a['status']:<12}")
    
    print("-" * 70)
    
    # Recommendations
    print("\nRECOMMENDATIONS:")
    
    if aggressive:
        print("\n  Orders requiring repricing (>$1 adjustment):")
        for symbol in aggressive:
            a = analyses[symbol]
            if abs(a['adjustment']) > 1:
                print(f"    {symbol}: Raise limit from ${a['limit_price']:.2f} to ${a['p70_limit']:.2f} "
                      f"(+${a['adjustment']:.2f})")
    
    if marginal:
        print("\n  Marginal orders (consider adjustment):")
        for symbol in marginal:
            a = analyses[symbol]
            print(f"    {symbol}: Current fill probability {a['prob_best']*100:.1f}%, "
                  f"P70 would be ${a['p70_limit']:.2f}")
    
    if good:
        print("\n  Well-positioned orders (no action needed):")
        for symbol in good:
            a = analyses[symbol]
            print(f"    {symbol}: {a['prob_best']*100:.1f}% fill probability")
    
    return {
        'analyses': analyses,
        'volatilities': volatilities,
        'data': data,
        'summary': {
            'good': good,
            'marginal': marginal,
            'aggressive': aggressive,
            'total_orders': len(analyses),
            'horizon_days': horizon_days,
            'analysis_date': datetime.now().isoformat()
        }
    }


def export_to_csv(results: Dict, filename: str = 'order_analysis.csv'):
    """Export analysis results to CSV"""
    rows = []
    
    for symbol, analysis in results['analyses'].items():
        row = {
            'Symbol': symbol,
            'Asset Class': get_asset_class(symbol),
            'Account': analysis['account'],
            'Shares': analysis['shares'],
            'Current Price': analysis['current_price'],
            'Limit Price': analysis['limit_price'],
            'Distance %': analysis['distance_pct'],
            'Volatility': analysis['volatility'],
            'Fill Prob (GBM)': analysis['prob_gbm'],
            'Fill Prob (Student-t)': analysis['prob_student_t'],
            'Fill Prob (GARCH)': analysis.get('prob_garch'),
            'Best Prob': analysis['prob_best'],
            'Best Model': analysis['prob_best_model'],
            'P70 Limit': analysis['p70_limit'],
            'Adjustment': analysis['adjustment'],
            'Status': analysis['status']
        }
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(filename, index=False)
    print(f"\nResults exported to {filename}")
    return df


def load_orders_from_csv(filename: str) -> Dict:
    """
    Load orders from a CSV file.

    Expected columns (case-insensitive):
    - Symbol: ticker symbol
    - Limit Price or Limit: limit order price
    - Shares: number of shares (optional, defaults to 1)
    - Account: account name (optional)

    Args:
        filename: Path to CSV file

    Returns:
        Dictionary of orders in CURRENT_ORDERS format
    """
    df = pd.read_csv(filename)

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower()

    # Find the limit price column
    limit_col = None
    for col in ['limit price', 'limit', 'limit_price']:
        if col in df.columns:
            limit_col = col
            break

    if limit_col is None:
        raise ValueError("CSV must have a 'Limit Price' or 'Limit' column")

    if 'symbol' not in df.columns:
        raise ValueError("CSV must have a 'Symbol' column")

    orders = {}
    for _, row in df.iterrows():
        symbol = row['symbol'].strip().upper()
        orders[symbol] = {
            'limit': float(row[limit_col]),
            'shares': int(row.get('shares', 1)) if pd.notna(row.get('shares')) else 1,
            'account': row.get('account', 'Unknown') if pd.notna(row.get('account')) else 'Unknown',
        }

    return orders


def analyze_orders_from_csv(input_csv: str,
                            output_csv: str = None,
                            horizon_days: int = 60,
                            preferred_source: str = 'yfinance',
                            require_garch: bool = True,
                            **kwargs) -> Dict:
    """
    Load orders from CSV and run full analysis.

    Args:
        input_csv: Path to input CSV with orders
        output_csv: Path for output CSV (default: input_csv with '_analyzed' suffix)
        horizon_days: Time horizon for fill probability
        preferred_source: Data source ('alpaca', 'tiingo', 'yfinance')
        require_garch: If True (default), require GARCH analysis. Error if data unavailable.

    Returns:
        Analysis results dictionary

    Raises:
        InsufficientDataError: If require_garch=True and insufficient data
    """
    global CURRENT_ORDERS

    # Load orders from CSV
    print(f"Loading orders from {input_csv}...")
    CURRENT_ORDERS = load_orders_from_csv(input_csv)
    print(f"Loaded {len(CURRENT_ORDERS)} orders: {', '.join(CURRENT_ORDERS.keys())}")

    # Run analysis
    results = analyze_portfolio_orders(
        horizon_days=horizon_days,
        preferred_source=preferred_source,
        require_garch=require_garch,
        **kwargs
    )

    if results and output_csv:
        export_to_csv(results, output_csv)
    elif results:
        # Default output name
        base = input_csv.rsplit('.', 1)[0]
        export_to_csv(results, f"{base}_analyzed.csv")

    return results


if __name__ == "__main__":
    import sys
    import os

    # Determine input file: command line arg or default
    if len(sys.argv) > 1:
        input_csv = sys.argv[1]
    else:
        input_csv = DEFAULT_ORDERS_FILE

    # Check if input file exists
    if not os.path.exists(input_csv):
        print(f"Error: Input file '{input_csv}' not found.")
        print(f"Please create {DEFAULT_ORDERS_FILE} with columns: Symbol, Limit, Shares, Account")
        sys.exit(1)

    output_csv = sys.argv[2] if len(sys.argv) > 2 else 'order_analysis.csv'

    results = analyze_orders_from_csv(
        input_csv=input_csv,
        output_csv=output_csv,
        horizon_days=60,
        preferred_source='yfinance'  # Change to 'alpaca' or 'tiingo' if configured
    )

    if results:
        print("\n" + "=" * 70)
        print("Analysis complete!")
        print("=" * 70)
