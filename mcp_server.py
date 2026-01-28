"""
MCP Server for Limit Order Fill Probability Calculator
======================================================

Exposes the limit order fill probability calculator as MCP tools for Claude
and other MCP clients.

Usage:
    # Run standalone
    python mcp_server.py

    # Test with MCP inspector
    npx @anthropics/mcp-inspector python mcp_server.py

    # Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
        "mcpServers": {
            "limit-order-probability": {
                "command": "python",
                "args": ["/path/to/mcp_server.py"]
            }
        }
    }
"""

import time
from typing import Optional, Literal
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

# Import core modules
from fill_probability import (
    FillProbabilityCalculator,
    VolatilityEstimator,
    FillDirection,
    calculate_p70_limit,
    analyze_order,
    InsufficientDataError,
)
from data_fetcher import DataFetcher


# Initialize MCP server
mcp = FastMCP("limit-order-probability")


# Simple TTL cache for market data
@dataclass
class CacheEntry:
    data: dict
    timestamp: float


_cache: dict[str, CacheEntry] = {}
CACHE_TTL_SECONDS = 300  # 5 minutes


def _get_cached_data(symbol: str, days: int = 252) -> Optional[dict]:
    """Get cached market data if still valid."""
    cache_key = f"{symbol}:{days}"
    if cache_key in _cache:
        entry = _cache[cache_key]
        if time.time() - entry.timestamp < CACHE_TTL_SECONDS:
            return entry.data
    return None


def _set_cached_data(symbol: str, days: int, data: dict) -> None:
    """Cache market data."""
    cache_key = f"{symbol}:{days}"
    _cache[cache_key] = CacheEntry(data=data, timestamp=time.time())


def _fetch_market_data(symbol: str, days: int = 252) -> dict:
    """
    Fetch market data for a symbol, with caching.

    Returns dict with: ohlc, returns_pct, current_price, error
    """
    # Check cache first
    cached = _get_cached_data(symbol, days)
    if cached is not None:
        return cached

    result = {
        "ohlc": None,
        "returns_pct": None,
        "current_price": None,
        "error": None,
    }

    try:
        fetcher = DataFetcher(preferred_source='yfinance')
        ohlc = fetcher.get_ohlc(symbol, days=days)
        returns_pct = fetcher.get_returns(symbol, days=days, as_percentage=True)

        result["ohlc"] = ohlc
        result["returns_pct"] = returns_pct
        result["current_price"] = float(ohlc['close'].iloc[-1])

        # Cache the result
        _set_cached_data(symbol, days, result)

    except Exception as e:
        result["error"] = str(e)

    return result


@mcp.tool()
def calculate_fill_probability(
    symbol: str,
    limit_price: float,
    horizon_days: int = 60,
    model: Literal["garch", "student_t", "gbm", "all"] = "garch",
) -> dict:
    """
    Calculate the probability that a limit order will fill within a time horizon.

    Uses sophisticated models that account for fat tails and volatility clustering
    in asset returns for more accurate probability estimates.

    Args:
        symbol: Ticker symbol (e.g., "VOO", "AAPL")
        limit_price: Limit order price
        horizon_days: Time horizon in trading days (default 60, ~3 months)
        model: Model to use - "garch" (recommended), "student_t", "gbm", or "all"

    Returns:
        Dictionary with fill probability, confidence interval, model details,
        and recommendation (GOOD/MARGINAL/TOO_AGGRESSIVE)
    """
    # Fetch market data
    data = _fetch_market_data(symbol)

    if data["error"]:
        return {"error": f"Failed to fetch data for {symbol}: {data['error']}"}

    current_price = data["current_price"]
    returns_pct = data["returns_pct"]
    ohlc = data["ohlc"]

    # Calculate volatility
    returns_decimal = returns_pct / 100

    try:
        # Try GARCH first (requires 100+ days)
        if len(returns_pct) >= 100:
            garch_vol, garch_details = VolatilityEstimator.fit_garch(returns_pct)
            volatility = garch_vol
            vol_method = "GARCH"
        else:
            # Fall back to Yang-Zhang
            volatility = VolatilityEstimator.yang_zhang(ohlc)
            vol_method = "Yang-Zhang"
            garch_details = None
    except Exception:
        # Final fallback to close-to-close
        volatility = VolatilityEstimator.close_to_close(returns_decimal)
        vol_method = "close-to-close"
        garch_details = None

    # Determine direction
    direction = FillDirection.BUY if limit_price < current_price else FillDirection.SELL
    distance_pct = abs(current_price - limit_price) / current_price * 100

    # Initialize calculator
    calc = FillProbabilityCalculator(
        current_price=current_price,
        limit_price=limit_price,
        horizon_days=horizon_days,
        volatility=volatility,
    )

    results = {
        "symbol": symbol,
        "current_price": current_price,
        "limit_price": limit_price,
        "direction": direction.value,
        "distance_pct": round(distance_pct, 2),
        "horizon_days": horizon_days,
        "volatility": round(volatility * 100, 2),
        "volatility_method": vol_method,
    }

    # Run requested model(s)
    if model in ("gbm", "all"):
        gbm = calc.gbm_closed_form()
        results["gbm"] = {
            "probability": round(gbm.probability, 4),
            "note": "Baseline - underestimates fill probability by 30-100%",
        }

    if model in ("student_t", "all"):
        # Use nu=4 for equities (fat tails)
        t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
        results["student_t"] = {
            "probability": round(t_result.probability, 4),
            "confidence_interval": [round(x, 4) for x in t_result.confidence_interval],
            "degrees_of_freedom": 4.0,
            "note": "Captures fat tails, not volatility clustering",
        }

    if model in ("garch", "all"):
        if len(returns_pct) >= 100:
            try:
                garch_result = calc.garch_monte_carlo(returns_pct, n_sims=50000)
                results["garch"] = {
                    "probability": round(garch_result.probability, 4),
                    "confidence_interval": [round(x, 4) for x in garch_result.confidence_interval],
                    "garch_nu": round(garch_result.details["garch_params"]["nu"], 2),
                    "persistence": round(garch_result.details["persistence"], 4),
                    "note": "RECOMMENDED - captures fat tails and volatility clustering",
                }
            except Exception as e:
                # Fall back to Student's t if GARCH fails
                t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
                results["garch"] = {
                    "probability": round(t_result.probability, 4),
                    "confidence_interval": [round(x, 4) for x in t_result.confidence_interval],
                    "note": f"Fallback to Student's t (GARCH failed: {e})",
                }
        else:
            # Insufficient data for GARCH
            t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
            results["garch"] = {
                "probability": round(t_result.probability, 4),
                "confidence_interval": [round(x, 4) for x in t_result.confidence_interval],
                "note": f"Fallback to Student's t (only {len(returns_pct)} days of data, GARCH needs 100+)",
            }

    # Determine best probability for recommendation
    if "garch" in results:
        best_prob = results["garch"]["probability"]
    elif "student_t" in results:
        best_prob = results["student_t"]["probability"]
    else:
        best_prob = results["gbm"]["probability"]

    # Status and recommendation
    if best_prob >= 0.65:
        status = "GOOD"
        recommendation = "Order is well-positioned with high fill probability."
    elif best_prob >= 0.45:
        status = "MARGINAL"
        recommendation = "Fill probability is moderate. Consider adjusting limit price closer to current price."
    else:
        status = "TOO_AGGRESSIVE"
        recommendation = "Fill probability is low. Strongly recommend adjusting limit price."

    results["status"] = status
    results["recommendation"] = recommendation
    results["fill_probability"] = best_prob

    return results


@mcp.tool()
def find_limit_for_probability(
    symbol: str,
    direction: Literal["buy", "sell"],
    target_probability: float = 0.70,
    horizon_days: int = 60,
) -> dict:
    """
    Find the limit price that achieves a target fill probability.

    This is the inverse function - given a desired fill probability,
    calculate what limit price to set.

    Args:
        symbol: Ticker symbol (e.g., "VOO", "AAPL")
        direction: "buy" (limit below current) or "sell" (limit above current)
        target_probability: Target fill probability (default 0.70 = 70%)
        horizon_days: Time horizon in trading days (default 60)

    Returns:
        Dictionary with recommended limit price, distance from current price,
        and current market data
    """
    # Fetch market data
    data = _fetch_market_data(symbol)

    if data["error"]:
        return {"error": f"Failed to fetch data for {symbol}: {data['error']}"}

    current_price = data["current_price"]
    returns_pct = data["returns_pct"]
    ohlc = data["ohlc"]

    # Calculate volatility
    returns_decimal = returns_pct / 100

    try:
        if len(returns_pct) >= 100:
            garch_vol, _ = VolatilityEstimator.fit_garch(returns_pct)
            volatility = garch_vol
            vol_method = "GARCH"
        else:
            volatility = VolatilityEstimator.yang_zhang(ohlc)
            vol_method = "Yang-Zhang"
    except Exception:
        volatility = VolatilityEstimator.close_to_close(returns_decimal)
        vol_method = "close-to-close"

    # Convert direction string to enum
    fill_direction = FillDirection.BUY if direction == "buy" else FillDirection.SELL

    # Calculate target limit price
    target_price = calculate_p70_limit(
        current_price=current_price,
        volatility=volatility,
        horizon_days=horizon_days,
        direction=fill_direction,
        target_prob=target_probability,
    )

    # Calculate distance
    distance = abs(current_price - target_price)
    distance_pct = distance / current_price * 100

    return {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "recommended_limit_price": round(target_price, 2),
        "direction": direction,
        "target_probability": target_probability,
        "distance_from_current": round(distance, 2),
        "distance_percent": round(distance_pct, 2),
        "horizon_days": horizon_days,
        "volatility": round(volatility * 100, 2),
        "volatility_method": vol_method,
        "note": f"Set limit at ${target_price:.2f} for ~{target_probability*100:.0f}% fill probability over {horizon_days} trading days",
    }


@mcp.tool()
def get_volatility_estimates(
    symbol: str,
    lookback_days: int = 252,
) -> dict:
    """
    Get volatility estimates for a symbol using multiple estimation methods.

    Returns annualized volatility estimates from:
    - Close-to-close (standard historical volatility)
    - Yang-Zhang (optimal OHLC estimator, ~8x more efficient)
    - EWMA (exponentially weighted, more responsive to recent moves)
    - GJR-GARCH (captures volatility clustering and leverage effect)

    Args:
        symbol: Ticker symbol (e.g., "VOO", "AAPL")
        lookback_days: Historical lookback period (default 252 = 1 year)

    Returns:
        Dictionary with volatility estimates from each method and recommendation
    """
    # Fetch market data
    data = _fetch_market_data(symbol, days=lookback_days)

    if data["error"]:
        return {"error": f"Failed to fetch data for {symbol}: {data['error']}"}

    returns_pct = data["returns_pct"]
    ohlc = data["ohlc"]
    current_price = data["current_price"]

    returns_decimal = returns_pct / 100

    results = {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "data_days": len(returns_pct),
        "estimates": {},
    }

    # Close-to-close (60-day and 20-day)
    results["estimates"]["close_to_close_60d"] = round(
        VolatilityEstimator.close_to_close(returns_decimal, window=60) * 100, 2
    )
    results["estimates"]["close_to_close_20d"] = round(
        VolatilityEstimator.close_to_close(returns_decimal, window=20) * 100, 2
    )

    # Yang-Zhang
    try:
        yz_vol = VolatilityEstimator.yang_zhang(ohlc, window=60)
        results["estimates"]["yang_zhang_60d"] = round(yz_vol * 100, 2)
    except Exception as e:
        results["estimates"]["yang_zhang_60d"] = f"Error: {e}"

    # EWMA
    results["estimates"]["ewma_lambda_94"] = round(
        VolatilityEstimator.ewma(returns_decimal, lambda_param=0.94) * 100, 2
    )
    results["estimates"]["ewma_lambda_97"] = round(
        VolatilityEstimator.ewma(returns_decimal, lambda_param=0.97) * 100, 2
    )

    # GARCH
    if len(returns_pct) >= 100:
        try:
            garch_vol, garch_details = VolatilityEstimator.fit_garch(returns_pct)
            results["estimates"]["garch"] = round(garch_vol * 100, 2)
            results["garch_details"] = {
                "degrees_of_freedom": round(garch_details["nu"], 2),
                "persistence": round(garch_details["persistence"], 4),
                "alpha": round(garch_details["alpha"], 4),
                "beta": round(garch_details["beta"], 4),
                "gamma": round(garch_details["gamma"], 4),
                "long_run_vol": round(garch_details["long_run_vol"] * 100, 2),
            }
        except Exception as e:
            results["estimates"]["garch"] = f"Error: {e}"
    else:
        results["estimates"]["garch"] = f"Insufficient data ({len(returns_pct)} days, need 100+)"

    # Best estimate recommendation
    if isinstance(results["estimates"].get("garch"), float):
        results["best_estimate"] = results["estimates"]["garch"]
        results["best_method"] = "GARCH (recommended)"
    elif isinstance(results["estimates"].get("yang_zhang_60d"), float):
        results["best_estimate"] = results["estimates"]["yang_zhang_60d"]
        results["best_method"] = "Yang-Zhang"
    else:
        results["best_estimate"] = results["estimates"]["ewma_lambda_97"]
        results["best_method"] = "EWMA (λ=0.97)"

    return results


@mcp.tool()
def analyze_order(
    symbol: str,
    limit_price: float,
    horizon_days: int = 60,
) -> dict:
    """
    Comprehensive analysis of a limit order with all models and detailed recommendations.

    Runs all probability models, calculates volatility estimates, and provides
    detailed status and recommendations for the order.

    Args:
        symbol: Ticker symbol (e.g., "VOO", "AAPL")
        limit_price: Limit order price
        horizon_days: Time horizon in trading days (default 60)

    Returns:
        Dictionary with all model results, volatility data, status, and recommendations
    """
    # Fetch market data
    data = _fetch_market_data(symbol)

    if data["error"]:
        return {"error": f"Failed to fetch data for {symbol}: {data['error']}"}

    current_price = data["current_price"]
    returns_pct = data["returns_pct"]
    ohlc = data["ohlc"]

    # Determine direction and distance
    direction = FillDirection.BUY if limit_price < current_price else FillDirection.SELL
    distance_pct = abs(current_price - limit_price) / current_price * 100

    results = {
        "symbol": symbol,
        "current_price": round(current_price, 2),
        "limit_price": limit_price,
        "direction": direction.value,
        "distance_pct": round(distance_pct, 2),
        "horizon_days": horizon_days,
        "data_days": len(returns_pct),
    }

    # Calculate all volatility estimates
    returns_decimal = returns_pct / 100

    vol_estimates = {
        "close_to_close_60d": round(VolatilityEstimator.close_to_close(returns_decimal, 60) * 100, 2),
        "ewma_97": round(VolatilityEstimator.ewma(returns_decimal, lambda_param=0.97) * 100, 2),
    }

    try:
        vol_estimates["yang_zhang_60d"] = round(VolatilityEstimator.yang_zhang(ohlc, 60) * 100, 2)
    except Exception:
        pass

    garch_details = None
    if len(returns_pct) >= 100:
        try:
            garch_vol, garch_details = VolatilityEstimator.fit_garch(returns_pct)
            vol_estimates["garch"] = round(garch_vol * 100, 2)
            volatility = garch_vol
            vol_method = "GARCH"
        except Exception:
            volatility = VolatilityEstimator.yang_zhang(ohlc)
            vol_method = "Yang-Zhang"
    else:
        try:
            volatility = VolatilityEstimator.yang_zhang(ohlc)
            vol_method = "Yang-Zhang"
        except Exception:
            volatility = VolatilityEstimator.close_to_close(returns_decimal)
            vol_method = "close-to-close"

    results["volatility_estimates"] = vol_estimates
    results["volatility_used"] = round(volatility * 100, 2)
    results["volatility_method"] = vol_method

    if garch_details:
        results["garch_parameters"] = {
            "degrees_of_freedom": round(garch_details["nu"], 2),
            "persistence": round(garch_details["persistence"], 4),
            "alpha": round(garch_details["alpha"], 4),
            "beta": round(garch_details["beta"], 4),
            "gamma": round(garch_details["gamma"], 4),
        }

    # Initialize calculator
    calc = FillProbabilityCalculator(
        current_price=current_price,
        limit_price=limit_price,
        horizon_days=horizon_days,
        volatility=volatility,
    )

    # Run all models
    model_results = {}

    # GBM
    gbm = calc.gbm_closed_form()
    model_results["gbm"] = {
        "probability": round(gbm.probability, 4),
        "note": "Baseline - underestimates by 30-100%",
    }

    # Student's t
    t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
    model_results["student_t"] = {
        "probability": round(t_result.probability, 4),
        "confidence_interval": [round(x, 4) for x in t_result.confidence_interval],
    }

    # GARCH Monte Carlo
    if len(returns_pct) >= 100:
        try:
            garch_mc = calc.garch_monte_carlo(returns_pct, n_sims=50000)
            model_results["garch_mc"] = {
                "probability": round(garch_mc.probability, 4),
                "confidence_interval": [round(x, 4) for x in garch_mc.confidence_interval],
                "note": "PRIMARY MODEL - recommended",
            }
            best_prob = garch_mc.probability
        except Exception as e:
            model_results["garch_mc"] = {"error": str(e)}
            best_prob = t_result.probability
    else:
        model_results["garch_mc"] = {"error": f"Insufficient data ({len(returns_pct)} days)"}
        best_prob = t_result.probability

    results["model_results"] = model_results
    results["fill_probability"] = round(best_prob, 4)

    # Calculate P70 target
    p70_price = calculate_p70_limit(current_price, volatility, horizon_days, direction, 0.70)
    results["p70_target"] = {
        "limit_price": round(p70_price, 2),
        "distance_pct": round(abs(current_price - p70_price) / current_price * 100, 2),
    }

    # Adjustment calculation
    if direction == FillDirection.BUY:
        adjustment = p70_price - limit_price
    else:
        adjustment = limit_price - p70_price
    results["adjustment_to_p70"] = round(adjustment, 2)

    # Status and recommendation
    if best_prob >= 0.65:
        status = "GOOD"
        recommendation = "Order is well-positioned. No adjustment needed."
    elif best_prob >= 0.45:
        status = "MARGINAL"
        recommendation = f"Consider raising limit by ${abs(adjustment):.2f} to ${p70_price:.2f} for ~70% fill probability."
    else:
        status = "TOO_AGGRESSIVE"
        recommendation = f"Limit is too aggressive. Raise to ${p70_price:.2f} (+${abs(adjustment):.2f}) for ~70% fill probability."

    results["status"] = status
    results["recommendation"] = recommendation

    return results


if __name__ == "__main__":
    mcp.run()
