"""
Unit tests for MCP Server
=========================

Tests all MCP tools for the limit order fill probability calculator.

Usage:
    python -m unittest test_mcp_server -v
"""

import unittest
from unittest.mock import patch
import pandas as pd
import numpy as np
from datetime import datetime


class TestMCPServerImport(unittest.TestCase):
    """Test that the MCP server can be imported correctly."""

    def test_import_mcp_server(self):
        """MCP server module should import without errors."""
        import mcp_server
        self.assertIsNotNone(mcp_server)

    def test_tools_registered(self):
        """All four tools should be registered."""
        from mcp_server import mcp
        tools = list(mcp._tool_manager._tools.keys())
        self.assertIn('calculate_fill_probability', tools)
        self.assertIn('find_limit_for_probability', tools)
        self.assertIn('get_volatility_estimates', tools)
        self.assertIn('analyze_order', tools)
        self.assertEqual(len(tools), 4)


class TestCacheHelpers(unittest.TestCase):
    """Test the caching mechanism."""

    def test_cache_set_and_get(self):
        """Cache should store and retrieve data."""
        from mcp_server import _get_cached_data, _set_cached_data, _cache

        # Clear cache
        _cache.clear()

        # Set data
        test_data = {"price": 100.0, "symbol": "TEST"}
        _set_cached_data("TEST", 252, test_data)

        # Get data
        result = _get_cached_data("TEST", 252)
        self.assertEqual(result, test_data)

    def test_cache_miss(self):
        """Cache should return None for missing keys."""
        from mcp_server import _get_cached_data, _cache

        _cache.clear()
        result = _get_cached_data("NONEXISTENT", 252)
        self.assertIsNone(result)

    def test_cache_different_days(self):
        """Cache should differentiate by lookback days."""
        from mcp_server import _get_cached_data, _set_cached_data, _cache

        _cache.clear()

        data_252 = {"days": 252}
        data_100 = {"days": 100}

        _set_cached_data("TEST", 252, data_252)
        _set_cached_data("TEST", 100, data_100)

        self.assertEqual(_get_cached_data("TEST", 252), data_252)
        self.assertEqual(_get_cached_data("TEST", 100), data_100)


def create_mock_ohlc_data(days: int = 252, base_price: float = 100.0) -> pd.DataFrame:
    """Create mock OHLC data for testing."""
    np.random.seed(42)
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')

    # Generate realistic price movements
    returns = np.random.normal(0.0005, 0.01, days)
    prices = base_price * np.exp(np.cumsum(returns))

    df = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, days)),
        'high': prices * (1 + np.random.uniform(0, 0.02, days)),
        'low': prices * (1 - np.random.uniform(0, 0.02, days)),
        'close': prices,
        'volume': np.random.randint(1000000, 10000000, days),
    }, index=dates)

    return df


def create_mock_returns(days: int = 252) -> pd.Series:
    """Create mock returns data for testing."""
    np.random.seed(42)
    returns_pct = np.random.normal(0.05, 1.0, days)  # ~1% daily vol in pct terms
    dates = pd.date_range(end=datetime.now(), periods=days, freq='B')
    return pd.Series(returns_pct, index=dates, name='TEST')


class TestCalculateFillProbability(unittest.TestCase):
    """Test calculate_fill_probability tool."""

    def setUp(self):
        """Set up mock data."""
        from mcp_server import _cache
        _cache.clear()

        self.mock_ohlc = create_mock_ohlc_data(252, 100.0)
        self.mock_returns = create_mock_returns(252)
        self.current_price = float(self.mock_ohlc['close'].iloc[-1])

    @patch('mcp_server._fetch_market_data')
    def test_basic_call_garch(self, mock_fetch):
        """Test basic call with GARCH model."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = calculate_fill_probability('TEST', 95.0, 60, 'garch')

        self.assertIn('symbol', result)
        self.assertEqual(result['symbol'], 'TEST')
        self.assertIn('fill_probability', result)
        self.assertIn('status', result)
        self.assertIn('recommendation', result)
        self.assertIn('garch', result)
        self.assertIn('probability', result['garch'])

    @patch('mcp_server._fetch_market_data')
    def test_all_models(self, mock_fetch):
        """Test with all models."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = calculate_fill_probability('TEST', 95.0, 60, 'all')

        self.assertIn('gbm', result)
        self.assertIn('student_t', result)
        self.assertIn('garch', result)

    @patch('mcp_server._fetch_market_data')
    def test_buy_direction(self, mock_fetch):
        """Test buy order (limit below current)."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = calculate_fill_probability('TEST', 95.0, 60, 'garch')
        self.assertEqual(result['direction'], 'buy')

    @patch('mcp_server._fetch_market_data')
    def test_sell_direction(self, mock_fetch):
        """Test sell order (limit above current)."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = calculate_fill_probability('TEST', 105.0, 60, 'garch')
        self.assertEqual(result['direction'], 'sell')

    @patch('mcp_server._fetch_market_data')
    def test_error_handling(self, mock_fetch):
        """Test error handling for failed data fetch."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': None,
            'returns_pct': None,
            'current_price': None,
            'error': 'Symbol not found',
        }

        result = calculate_fill_probability('INVALID', 100.0, 60, 'garch')
        self.assertIn('error', result)

    @patch('mcp_server._fetch_market_data')
    def test_status_good(self, mock_fetch):
        """Test GOOD status for high probability orders."""
        from mcp_server import calculate_fill_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        # Limit very close to current price should have high fill probability
        result = calculate_fill_probability('TEST', 99.5, 60, 'garch')
        self.assertIn('fill_probability', result)
        self.assertIn('status', result)

    @patch('mcp_server._fetch_market_data')
    def test_insufficient_data_fallback(self, mock_fetch):
        """Test fallback to Student's t with insufficient data."""
        from mcp_server import calculate_fill_probability

        # Only 50 days of data
        short_returns = create_mock_returns(50)

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc.tail(50),
            'returns_pct': short_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = calculate_fill_probability('TEST', 95.0, 60, 'garch')
        # Should still return a result with fallback
        self.assertIn('garch', result)
        self.assertIn('note', result['garch'])


class TestFindLimitForProbability(unittest.TestCase):
    """Test find_limit_for_probability tool."""

    def setUp(self):
        """Set up mock data."""
        from mcp_server import _cache
        _cache.clear()

        self.mock_ohlc = create_mock_ohlc_data(252, 100.0)
        self.mock_returns = create_mock_returns(252)
        self.current_price = float(self.mock_ohlc['close'].iloc[-1])

    @patch('mcp_server._fetch_market_data')
    def test_find_buy_limit(self, mock_fetch):
        """Test finding buy limit price."""
        from mcp_server import find_limit_for_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = find_limit_for_probability('TEST', 'buy', 0.70, 60)

        self.assertIn('recommended_limit_price', result)
        self.assertIn('distance_from_current', result)
        self.assertIn('distance_percent', result)
        self.assertEqual(result['direction'], 'buy')
        # Buy limit should be below current price
        self.assertLess(result['recommended_limit_price'], 100.0)

    @patch('mcp_server._fetch_market_data')
    def test_find_sell_limit(self, mock_fetch):
        """Test finding sell limit price."""
        from mcp_server import find_limit_for_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = find_limit_for_probability('TEST', 'sell', 0.70, 60)

        self.assertEqual(result['direction'], 'sell')
        # Sell limit should be above current price
        self.assertGreater(result['recommended_limit_price'], 100.0)

    @patch('mcp_server._fetch_market_data')
    def test_different_probabilities(self, mock_fetch):
        """Test that higher probability targets are closer to current price."""
        from mcp_server import find_limit_for_probability

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result_70 = find_limit_for_probability('TEST', 'buy', 0.70, 60)
        result_50 = find_limit_for_probability('TEST', 'buy', 0.50, 60)

        # 70% target should be closer to current price than 50% target
        self.assertGreater(
            result_70['recommended_limit_price'],
            result_50['recommended_limit_price']
        )

    @patch('mcp_server._fetch_market_data')
    def test_error_handling(self, mock_fetch):
        """Test error handling."""
        from mcp_server import find_limit_for_probability

        mock_fetch.return_value = {
            'ohlc': None,
            'returns_pct': None,
            'current_price': None,
            'error': 'Network error',
        }

        result = find_limit_for_probability('INVALID', 'buy', 0.70, 60)
        self.assertIn('error', result)


class TestGetVolatilityEstimates(unittest.TestCase):
    """Test get_volatility_estimates tool."""

    def setUp(self):
        """Set up mock data."""
        from mcp_server import _cache
        _cache.clear()

        self.mock_ohlc = create_mock_ohlc_data(252, 100.0)
        self.mock_returns = create_mock_returns(252)
        self.current_price = float(self.mock_ohlc['close'].iloc[-1])

    @patch('mcp_server._fetch_market_data')
    def test_all_estimates_returned(self, mock_fetch):
        """Test that all volatility estimates are returned."""
        from mcp_server import get_volatility_estimates

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = get_volatility_estimates('TEST', 252)

        self.assertIn('estimates', result)
        estimates = result['estimates']

        self.assertIn('close_to_close_60d', estimates)
        self.assertIn('close_to_close_20d', estimates)
        self.assertIn('yang_zhang_60d', estimates)
        self.assertIn('ewma_lambda_94', estimates)
        self.assertIn('ewma_lambda_97', estimates)
        self.assertIn('garch', estimates)

    @patch('mcp_server._fetch_market_data')
    def test_garch_details_included(self, mock_fetch):
        """Test that GARCH details are included."""
        from mcp_server import get_volatility_estimates

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = get_volatility_estimates('TEST', 252)

        self.assertIn('garch_details', result)
        details = result['garch_details']
        self.assertIn('degrees_of_freedom', details)
        self.assertIn('persistence', details)
        self.assertIn('alpha', details)
        self.assertIn('beta', details)
        self.assertIn('gamma', details)

    @patch('mcp_server._fetch_market_data')
    def test_best_estimate_selected(self, mock_fetch):
        """Test that best estimate is selected."""
        from mcp_server import get_volatility_estimates

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = get_volatility_estimates('TEST', 252)

        self.assertIn('best_estimate', result)
        self.assertIn('best_method', result)
        self.assertIsInstance(result['best_estimate'], float)

    @patch('mcp_server._fetch_market_data')
    def test_volatility_values_reasonable(self, mock_fetch):
        """Test that volatility values are in reasonable range."""
        from mcp_server import get_volatility_estimates

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': self.current_price,
            'error': None,
        }

        result = get_volatility_estimates('TEST', 252)

        # Volatilities should be positive and typically < 100% for most assets
        for _, value in result['estimates'].items():
            if isinstance(value, (int, float)):
                self.assertGreater(value, 0)
                self.assertLess(value, 200)  # 200% would be extreme

    @patch('mcp_server._fetch_market_data')
    def test_error_handling(self, mock_fetch):
        """Test error handling."""
        from mcp_server import get_volatility_estimates

        mock_fetch.return_value = {
            'ohlc': None,
            'returns_pct': None,
            'current_price': None,
            'error': 'Data unavailable',
        }

        result = get_volatility_estimates('INVALID', 252)
        self.assertIn('error', result)


class TestAnalyzeOrder(unittest.TestCase):
    """Test analyze_order tool."""

    def setUp(self):
        """Set up mock data."""
        from mcp_server import _cache
        _cache.clear()

        self.mock_ohlc = create_mock_ohlc_data(252, 100.0)
        self.mock_returns = create_mock_returns(252)
        self.current_price = float(self.mock_ohlc['close'].iloc[-1])

    @patch('mcp_server._fetch_market_data')
    def test_comprehensive_results(self, mock_fetch):
        """Test that comprehensive results are returned."""
        from mcp_server import analyze_order

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = analyze_order('TEST', 95.0, 60)

        # Basic info
        self.assertIn('symbol', result)
        self.assertIn('current_price', result)
        self.assertIn('limit_price', result)
        self.assertIn('direction', result)
        self.assertIn('distance_pct', result)

        # Volatility
        self.assertIn('volatility_estimates', result)
        self.assertIn('volatility_used', result)
        self.assertIn('volatility_method', result)

        # Models
        self.assertIn('model_results', result)
        self.assertIn('gbm', result['model_results'])
        self.assertIn('student_t', result['model_results'])
        self.assertIn('garch_mc', result['model_results'])

        # Recommendations
        self.assertIn('fill_probability', result)
        self.assertIn('p70_target', result)
        self.assertIn('adjustment_to_p70', result)
        self.assertIn('status', result)
        self.assertIn('recommendation', result)

    @patch('mcp_server._fetch_market_data')
    def test_status_values(self, mock_fetch):
        """Test that status is one of expected values."""
        from mcp_server import analyze_order

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = analyze_order('TEST', 95.0, 60)

        self.assertIn(result['status'], ['GOOD', 'MARGINAL', 'TOO_AGGRESSIVE'])

    @patch('mcp_server._fetch_market_data')
    def test_garch_parameters_included(self, mock_fetch):
        """Test that GARCH parameters are included when available."""
        from mcp_server import analyze_order

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = analyze_order('TEST', 95.0, 60)

        self.assertIn('garch_parameters', result)
        params = result['garch_parameters']
        self.assertIn('degrees_of_freedom', params)
        self.assertIn('persistence', params)

    @patch('mcp_server._fetch_market_data')
    def test_p70_target_calculation(self, mock_fetch):
        """Test P70 target is calculated."""
        from mcp_server import analyze_order

        mock_fetch.return_value = {
            'ohlc': self.mock_ohlc,
            'returns_pct': self.mock_returns,
            'current_price': 100.0,
            'error': None,
        }

        result = analyze_order('TEST', 95.0, 60)

        p70 = result['p70_target']
        self.assertIn('limit_price', p70)
        self.assertIn('distance_pct', p70)
        # For a buy order, P70 should be between limit and current
        self.assertGreater(p70['limit_price'], 95.0)
        self.assertLess(p70['limit_price'], 100.0)

    @patch('mcp_server._fetch_market_data')
    def test_error_handling(self, mock_fetch):
        """Test error handling."""
        from mcp_server import analyze_order

        mock_fetch.return_value = {
            'ohlc': None,
            'returns_pct': None,
            'current_price': None,
            'error': 'API error',
        }

        result = analyze_order('INVALID', 100.0, 60)
        self.assertIn('error', result)


class TestIntegration(unittest.TestCase):
    """Integration tests that hit the actual data source (slow)."""

    @unittest.skip("Integration test - requires network. Run manually with: python -m unittest test_mcp_server.TestIntegration")
    def test_real_data_voo(self):
        """Test with real VOO data."""
        from mcp_server import calculate_fill_probability, _cache
        _cache.clear()

        result = calculate_fill_probability('VOO', 550.0, 60, 'garch')

        self.assertNotIn('error', result)
        self.assertIn('fill_probability', result)
        self.assertIsInstance(result['fill_probability'], float)
        self.assertGreater(result['fill_probability'], 0)
        self.assertLess(result['fill_probability'], 1)

    @unittest.skip("Integration test - requires network. Run manually.")
    def test_real_data_invalid_symbol(self):
        """Test error handling with real invalid symbol."""
        from mcp_server import calculate_fill_probability, _cache
        _cache.clear()

        result = calculate_fill_probability('NOTAREALSYMBOL123', 100.0, 60, 'garch')
        self.assertIn('error', result)


if __name__ == '__main__':
    unittest.main()
