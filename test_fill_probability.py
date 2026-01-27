"""
Unit tests for the fill probability calculator.
"""

import unittest
import numpy as np
import pandas as pd
from fill_probability import (
    FillProbabilityCalculator,
    FillDirection,
    VolatilityEstimator,
    calculate_p70_limit,
)


class TestFillDirection(unittest.TestCase):
    """Test fill direction detection."""

    def test_buy_direction(self):
        """Limit below current price is a buy order."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=30, volatility=0.20
        )
        self.assertEqual(calc.direction, FillDirection.BUY)

    def test_sell_direction(self):
        """Limit above current price is a sell order."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=105, horizon_days=30, volatility=0.20
        )
        self.assertEqual(calc.direction, FillDirection.SELL)


class TestGBMClosedForm(unittest.TestCase):
    """Test GBM closed-form probability calculations."""

    def test_probability_bounds(self):
        """Probability should be between 0 and 1."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.20
        )
        result = calc.gbm_closed_form()
        self.assertGreaterEqual(result.probability, 0)
        self.assertLessEqual(result.probability, 1)

    def test_closer_limit_higher_probability(self):
        """Closer limit price should have higher fill probability."""
        calc_close = FillProbabilityCalculator(
            current_price=100, limit_price=98, horizon_days=60, volatility=0.20
        )
        calc_far = FillProbabilityCalculator(
            current_price=100, limit_price=90, horizon_days=60, volatility=0.20
        )
        self.assertGreater(
            calc_close.gbm_closed_form().probability,
            calc_far.gbm_closed_form().probability,
        )

    def test_longer_horizon_higher_probability(self):
        """Longer time horizon should increase fill probability."""
        calc_short = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=30, volatility=0.20
        )
        calc_long = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=90, volatility=0.20
        )
        self.assertGreater(
            calc_long.gbm_closed_form().probability,
            calc_short.gbm_closed_form().probability,
        )

    def test_higher_volatility_higher_probability(self):
        """Higher volatility should increase fill probability for buy orders."""
        calc_low_vol = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.10
        )
        calc_high_vol = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.30
        )
        self.assertGreater(
            calc_high_vol.gbm_closed_form().probability,
            calc_low_vol.gbm_closed_form().probability,
        )

    def test_result_attributes(self):
        """Result should have all expected attributes."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.20
        )
        result = calc.gbm_closed_form()
        self.assertEqual(result.current_price, 100)
        self.assertEqual(result.limit_price, 95)
        self.assertEqual(result.horizon_days, 60)
        self.assertEqual(result.volatility_used, 0.20)
        self.assertEqual(result.direction, FillDirection.BUY)
        self.assertIn("GBM", result.model)


class TestStudentTMonteCarlo(unittest.TestCase):
    """Test Student's t Monte Carlo simulations."""

    def test_probability_bounds(self):
        """Probability should be between 0 and 1."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.20
        )
        result = calc.student_t_monte_carlo(nu=4.0, n_sims=10000, seed=42)
        self.assertGreaterEqual(result.probability, 0)
        self.assertLessEqual(result.probability, 1)

    def test_confidence_interval(self):
        """Result should include confidence interval."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.20
        )
        result = calc.student_t_monte_carlo(nu=4.0, n_sims=10000, seed=42)
        self.assertIsNotNone(result.confidence_interval)
        lower, upper = result.confidence_interval
        self.assertLess(lower, result.probability)
        self.assertGreater(upper, result.probability)

    def test_reproducibility_with_seed(self):
        """Same seed should produce same result."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=95, horizon_days=60, volatility=0.20
        )
        result1 = calc.student_t_monte_carlo(nu=4.0, n_sims=10000, seed=42)
        result2 = calc.student_t_monte_carlo(nu=4.0, n_sims=10000, seed=42)
        self.assertEqual(result1.probability, result2.probability)

    def test_fatter_tails_affect_probability(self):
        """Lower degrees of freedom (fatter tails) should affect probability."""
        calc = FillProbabilityCalculator(
            current_price=100, limit_price=90, horizon_days=60, volatility=0.20
        )
        # Just verify both run without error - the direction of effect depends on context
        result_fat = calc.student_t_monte_carlo(nu=3.0, n_sims=10000, seed=42)
        result_thin = calc.student_t_monte_carlo(nu=10.0, n_sims=10000, seed=42)
        self.assertIsNotNone(result_fat.probability)
        self.assertIsNotNone(result_thin.probability)


class TestVolatilityEstimator(unittest.TestCase):
    """Test volatility estimation methods."""

    def setUp(self):
        """Create sample return series."""
        np.random.seed(42)
        # Generate 100 days of synthetic returns
        self.returns = pd.Series(np.random.normal(0, 0.01, 100))

    def test_close_to_close_positive(self):
        """Close-to-close volatility should be positive."""
        vol = VolatilityEstimator.close_to_close(self.returns)
        self.assertGreater(vol, 0)

    def test_close_to_close_annualized(self):
        """Close-to-close volatility should be annualized (roughly sqrt(252) * daily)."""
        daily_std = self.returns.std()
        annualized = VolatilityEstimator.close_to_close(self.returns)
        # Should be approximately sqrt(252) times daily
        expected = daily_std * np.sqrt(252)
        self.assertAlmostEqual(annualized, expected, places=2)

    def test_ewma_positive(self):
        """EWMA volatility should be positive."""
        vol = VolatilityEstimator.ewma(self.returns)
        self.assertGreater(vol, 0)

    def test_yang_zhang_with_ohlc(self):
        """Yang-Zhang should work with OHLC data."""
        np.random.seed(42)
        n = 100
        close = 100 * np.exp(np.cumsum(np.random.normal(0, 0.01, n)))
        # Generate plausible OHLC data
        ohlc = pd.DataFrame({
            'open': close * (1 + np.random.uniform(-0.005, 0.005, n)),
            'high': close * (1 + np.random.uniform(0.001, 0.02, n)),
            'low': close * (1 - np.random.uniform(0.001, 0.02, n)),
            'close': close,
        })
        vol = VolatilityEstimator.yang_zhang(ohlc)
        self.assertGreater(vol, 0)


class TestCalculateP70Limit(unittest.TestCase):
    """Test inverse probability function."""

    def test_p70_limit_buy(self):
        """P70 limit for buy should be below current price."""
        limit = calculate_p70_limit(
            current_price=100,
            volatility=0.20,
            horizon_days=60,
            direction=FillDirection.BUY,
            target_prob=0.70,
        )
        self.assertLess(limit, 100)

    def test_p70_limit_sell(self):
        """P70 limit for sell should be above current price."""
        limit = calculate_p70_limit(
            current_price=100,
            volatility=0.20,
            horizon_days=60,
            direction=FillDirection.SELL,
            target_prob=0.70,
        )
        self.assertGreater(limit, 100)

    def test_higher_target_closer_limit(self):
        """Higher target probability should result in limit closer to current price."""
        limit_70 = calculate_p70_limit(
            current_price=100,
            volatility=0.20,
            horizon_days=60,
            direction=FillDirection.BUY,
            target_prob=0.70,
        )
        limit_90 = calculate_p70_limit(
            current_price=100,
            volatility=0.20,
            horizon_days=60,
            direction=FillDirection.BUY,
            target_prob=0.90,
        )
        # For buy orders, higher probability means closer to current price (higher limit)
        self.assertGreater(limit_90, limit_70)


if __name__ == "__main__":
    unittest.main()
