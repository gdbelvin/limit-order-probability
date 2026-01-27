"""
Limit Order Fill Probability Calculator
========================================

A sophisticated toolkit for calculating the probability that a limit order
will fill within a specified time horizon (30-60 days).

Models implemented:
1. GBM (Geometric Brownian Motion) - Closed-form baseline
2. Student's t Monte Carlo - Captures fat tails
3. GJR-GARCH with t-innovations - Captures fat tails + volatility clustering

Supports multiple volatility models for robust probability estimation.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm, t as t_dist
from scipy.optimize import minimize_scalar
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
from enum import Enum
import warnings


class FillDirection(Enum):
    """Direction of the limit order"""
    BUY = "buy"   # Limit price below current price
    SELL = "sell" # Limit price above current price


@dataclass
class VolatilityEstimate:
    """Container for volatility estimates from different methods"""
    close_to_close: float      # Standard close-to-close volatility
    yang_zhang: Optional[float] = None  # Yang-Zhang OHLC estimator
    ewma: Optional[float] = None        # EWMA volatility
    garch_forecast: Optional[float] = None  # GARCH conditional forecast
    implied: Optional[float] = None     # Market implied volatility
    
    @property
    def best_estimate(self) -> float:
        """Return the best available volatility estimate"""
        # Priority: implied > garch > yang_zhang > ewma > close_to_close
        if self.implied is not None:
            return self.implied * 0.85  # Adjust for vol risk premium
        if self.garch_forecast is not None:
            return self.garch_forecast
        if self.yang_zhang is not None:
            return self.yang_zhang
        if self.ewma is not None:
            return self.ewma
        return self.close_to_close


@dataclass
class FillProbabilityResult:
    """Results from fill probability calculation"""
    probability: float
    model: str
    volatility_used: float
    current_price: float
    limit_price: float
    horizon_days: int
    direction: FillDirection
    confidence_interval: Optional[Tuple[float, float]] = None
    details: Optional[Dict] = None


class VolatilityEstimator:
    """
    Calculate volatility using multiple methods.
    
    Implements:
    - Close-to-close (standard)
    - Yang-Zhang (optimal OHLC estimator)
    - EWMA (Exponentially Weighted Moving Average)
    - GJR-GARCH (with asymmetric volatility)
    """
    
    @staticmethod
    def close_to_close(returns: pd.Series, window: int = 60) -> float:
        """
        Standard close-to-close volatility estimator.
        
        Args:
            returns: Series of log returns
            window: Lookback window in days
            
        Returns:
            Annualized volatility
        """
        return returns.tail(window).std() * np.sqrt(252)
    
    @staticmethod
    def yang_zhang(df: pd.DataFrame, window: int = 60) -> float:
        """
        Yang-Zhang volatility estimator using OHLC data.
        
        This is the optimal estimator for OHLC data - approximately
        8x more efficient than close-to-close.
        
        Args:
            df: DataFrame with 'open', 'high', 'low', 'close' columns
            window: Lookback window in days
            
        Returns:
            Annualized volatility
        """
        df = df.tail(window + 1).copy()
        
        # Normalize column names
        df.columns = df.columns.str.lower()
        
        log_ho = np.log(df['high'] / df['open'])
        log_lo = np.log(df['low'] / df['open'])
        log_co = np.log(df['close'] / df['open'])
        log_oc = np.log(df['open'] / df['close'].shift(1))
        
        # Rogers-Satchell volatility component
        rs = log_ho * (log_ho - log_co) + log_lo * (log_lo - log_co)
        
        # Yang-Zhang weighting factor
        k = 0.34 / (1.34 + (window + 1) / (window - 1))
        
        # Overnight variance
        overnight_var = log_oc.var()
        
        # Open-to-close variance
        open_close_var = log_co.var()
        
        # Rogers-Satchell variance
        rs_var = rs.mean()
        
        # Yang-Zhang variance
        yz_var = overnight_var + k * open_close_var + (1 - k) * rs_var
        
        return np.sqrt(yz_var * 252)
    
    @staticmethod
    def ewma(returns: pd.Series, lambda_param: float = 0.94, 
             window: int = 60) -> float:
        """
        Exponentially Weighted Moving Average volatility.
        
        RiskMetrics standard uses lambda=0.94 for daily data.
        For monthly forecasting, lambda=0.97 is more appropriate.
        
        Args:
            returns: Series of log returns
            lambda_param: Decay factor (0.94 for daily, 0.97 for monthly)
            window: Number of days to use
            
        Returns:
            Annualized volatility
        """
        returns = returns.tail(window)
        
        # Initialize with sample variance
        var = returns.iloc[:10].var()
        
        # EWMA recursion
        for r in returns.iloc[10:]:
            var = lambda_param * var + (1 - lambda_param) * r**2
        
        return np.sqrt(var * 252)
    
    @staticmethod
    def fit_garch(returns_pct: pd.Series, 
                  horizon: int = 1) -> Tuple[float, Dict]:
        """
        Fit GJR-GARCH(1,1) model with Student's t innovations.
        
        The GJR variant captures the leverage effect (negative returns
        increase volatility more than positive returns).
        
        Args:
            returns_pct: Returns in PERCENTAGE terms (multiply decimal by 100)
            horizon: Forecast horizon in days
            
        Returns:
            Tuple of (annualized volatility forecast, model details)
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "arch package required for GARCH models. "
                "Install with: pip install arch"
            )
        
        # Fit GJR-GARCH with t-distributed innovations
        model = arch_model(
            returns_pct, 
            vol='GARCH', 
            p=1, o=1, q=1,  # GJR-GARCH(1,1)
            dist='t'
        )
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp='off')
        
        # Extract parameters
        omega = result.params['omega']
        alpha = result.params['alpha[1]']
        gamma = result.params.get('gamma[1]', 0)  # Asymmetry term
        beta = result.params['beta[1]']
        nu = result.params['nu']  # Degrees of freedom
        
        # Current conditional volatility
        current_vol = result.conditional_volatility.iloc[-1]
        
        # Long-run (unconditional) variance
        persistence = alpha + gamma/2 + beta
        if persistence < 1:
            long_run_var = omega / (1 - persistence)
        else:
            long_run_var = current_vol**2
        
        # Multi-step forecast (converges to unconditional)
        if horizon == 1:
            forecast_var = current_vol**2
        else:
            # Variance converges to long-run at rate of persistence
            forecast_var = long_run_var + (current_vol**2 - long_run_var) * (persistence ** horizon)
        
        # Convert from daily percentage to annualized decimal
        annualized_vol = np.sqrt(forecast_var * 252) / 100
        
        details = {
            'omega': omega,
            'alpha': alpha,
            'gamma': gamma,
            'beta': beta,
            'nu': nu,
            'persistence': persistence,
            'current_daily_vol': current_vol / 100,
            'long_run_vol': np.sqrt(long_run_var * 252) / 100
        }
        
        return annualized_vol, details


class FillProbabilityCalculator:
    """
    Calculate limit order fill probabilities using multiple models.
    """
    
    def __init__(self, 
                 current_price: float,
                 limit_price: float,
                 horizon_days: int,
                 volatility: float,
                 risk_free_rate: float = 0.04):
        """
        Initialize the calculator.
        
        Args:
            current_price: Current market price
            limit_price: Limit order price
            horizon_days: Time horizon in trading days
            volatility: Annualized volatility (decimal, e.g., 0.20 for 20%)
            risk_free_rate: Annualized risk-free rate (default 4%)
        """
        self.S0 = current_price
        self.B = limit_price
        self.T = horizon_days / 252  # Convert to years
        self.horizon_days = horizon_days
        self.sigma = volatility
        self.r = risk_free_rate
        
        # Determine direction
        if limit_price < current_price:
            self.direction = FillDirection.BUY
        else:
            self.direction = FillDirection.SELL
    
    def gbm_closed_form(self) -> FillProbabilityResult:
        """
        Closed-form first-passage probability under GBM.
        
        Uses the reflection principle for Brownian motion.
        This is a BASELINE that underestimates fat-tail events.
        
        Formula:
        P(τ_B ≤ T) = Φ(d₁) + (B/S₀)^(2μ/σ²) × Φ(d₂)
        
        where:
        - d₁ = [ln(B/S₀) - μT] / (σ√T)
        - d₂ = [ln(B/S₀) + μT] / (σ√T)
        - μ = r - σ²/2 (drift under GBM)
        """
        mu = self.r - 0.5 * self.sigma**2
        sigma_sqrt_T = self.sigma * np.sqrt(self.T)
        
        log_ratio = np.log(self.B / self.S0)
        
        d1 = (log_ratio - mu * self.T) / sigma_sqrt_T
        d2 = (log_ratio + mu * self.T) / sigma_sqrt_T
        
        # Exponent for reflection term
        if self.sigma > 0:
            exponent = 2 * mu / (self.sigma**2)
        else:
            exponent = 0
        
        # First-passage probability
        if self.direction == FillDirection.BUY:
            # P(min S_t <= B)
            prob = norm.cdf(d1) + (self.B / self.S0)**exponent * norm.cdf(d2)
        else:
            # P(max S_t >= B) - use symmetry
            prob = norm.cdf(-d1) + (self.B / self.S0)**exponent * norm.cdf(-d2)
        
        # Ensure probability is in [0, 1]
        prob = np.clip(prob, 0, 1)
        
        return FillProbabilityResult(
            probability=prob,
            model="GBM (Closed-Form)",
            volatility_used=self.sigma,
            current_price=self.S0,
            limit_price=self.B,
            horizon_days=self.horizon_days,
            direction=self.direction,
            details={
                'd1': d1,
                'd2': d2,
                'drift': mu,
                'note': 'Baseline model - underestimates fat-tail events by ~30-100%'
            }
        )
    
    def student_t_monte_carlo(self,
                              nu: float = 4.0,
                              n_sims: int = 50000,
                              seed: Optional[int] = None) -> FillProbabilityResult:
        """
        Vectorized Monte Carlo simulation with Student's t-distributed returns.

        Captures fat tails but not volatility clustering.
        Typical degrees of freedom for stocks: 3-5

        Args:
            nu: Degrees of freedom (lower = fatter tails)
            n_sims: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            FillProbabilityResult
        """
        if seed is not None:
            np.random.seed(seed)

        # Daily volatility
        daily_vol = self.sigma / np.sqrt(252)

        # Scale factor for unit variance t-distribution
        # t-distribution with nu df has variance nu/(nu-2) for nu > 2
        if nu > 2:
            scale = np.sqrt((nu - 2) / nu)
        else:
            scale = 1.0

        # Generate all random innovations at once: (n_sims, horizon_days)
        z = t_dist.rvs(nu, size=(n_sims, self.horizon_days)) * scale

        # Compute log returns
        log_returns = daily_vol * z

        # Cumulative log returns -> price paths
        cumulative_returns = np.cumsum(log_returns, axis=1)
        price_paths = self.S0 * np.exp(cumulative_returns)

        # Check fill condition (vectorized)
        if self.direction == FillDirection.BUY:
            # Check if minimum price along each path <= limit
            fills = np.min(price_paths, axis=1) <= self.B
        else:
            # Check if maximum price along each path >= limit
            fills = np.max(price_paths, axis=1) >= self.B

        prob = np.mean(fills)

        # Confidence interval (Wilson score interval)
        z_alpha = 1.96
        n = n_sims
        p_hat = prob

        denom = 1 + z_alpha**2 / n
        center = (p_hat + z_alpha**2 / (2*n)) / denom
        spread = z_alpha * np.sqrt((p_hat * (1-p_hat) + z_alpha**2 / (4*n)) / n) / denom

        ci = (max(0, center - spread), min(1, center + spread))

        return FillProbabilityResult(
            probability=prob,
            model=f"Student's t Monte Carlo (ν={nu})",
            volatility_used=self.sigma,
            current_price=self.S0,
            limit_price=self.B,
            horizon_days=self.horizon_days,
            direction=self.direction,
            confidence_interval=ci,
            details={
                'degrees_of_freedom': nu,
                'n_simulations': n_sims,
                'excess_kurtosis': 6/(nu-4) if nu > 4 else float('inf'),
                'note': 'Captures fat tails, not volatility clustering'
            }
        )
    
    def garch_monte_carlo(self,
                          returns_pct: pd.Series,
                          n_sims: int = 50000,
                          seed: Optional[int] = None) -> FillProbabilityResult:
        """
        Vectorized Monte Carlo simulation with GJR-GARCH volatility dynamics.

        This is the RECOMMENDED model - captures both fat tails
        (via Student's t innovations) and volatility clustering
        (via GARCH dynamics).

        Args:
            returns_pct: Historical returns in PERCENTAGE terms
            n_sims: Number of simulation paths
            seed: Random seed for reproducibility

        Returns:
            FillProbabilityResult
        """
        try:
            from arch import arch_model
        except ImportError:
            raise ImportError(
                "arch package required for GARCH models. "
                "Install with: pip install arch"
            )

        if seed is not None:
            np.random.seed(seed)

        # Fit GJR-GARCH model
        model = arch_model(
            returns_pct,
            vol='GARCH',
            p=1, o=1, q=1,
            dist='t'
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = model.fit(disp='off')

        # Extract parameters
        omega = result.params['omega']
        alpha = result.params['alpha[1]']
        gamma = result.params.get('gamma[1]', 0)
        beta = result.params['beta[1]']
        nu = result.params['nu']

        # Current state
        current_vol = result.conditional_volatility.iloc[-1]

        # Scale factor for t-distribution
        if nu > 2:
            scale = np.sqrt((nu - 2) / nu)
        else:
            scale = 1.0

        # Pre-generate all random innovations: (n_sims, horizon_days)
        z = t_dist.rvs(nu, size=(n_sims, self.horizon_days)) * scale

        # Initialize arrays for vectorized simulation
        # All paths start at current price and volatility
        prices = np.full(n_sims, self.S0)
        vols = np.full(n_sims, current_vol)
        filled = np.zeros(n_sims, dtype=bool)

        # Simulate day by day (loop over time, vectorized across paths)
        for t in range(self.horizon_days):
            # Only simulate paths that haven't filled yet
            active = ~filled

            if not np.any(active):
                break

            # Get innovations for this day
            z_t = z[active, t]

            # Returns (vol is in percentage terms)
            ret_pct = vols[active] * z_t
            ret = ret_pct / 100  # Convert to decimal

            # Update prices
            prices[active] = prices[active] * np.exp(ret)

            # Check fill condition
            if self.direction == FillDirection.BUY:
                new_fills = prices <= self.B
            else:
                new_fills = prices >= self.B

            filled = filled | new_fills

            # Update GARCH volatility for paths still active
            still_active = ~filled
            if np.any(still_active):
                # GJR asymmetric term: indicator for negative returns
                # Need to recompute ret_pct for still_active paths
                active_mask = active & still_active
                ret_pct_active = vols[active_mask] * z[active_mask, t]
                indicator = (ret_pct_active < 0).astype(float)
                vols[active_mask] = np.sqrt(
                    omega +
                    (alpha + gamma * indicator) * ret_pct_active**2 +
                    beta * vols[active_mask]**2
                )

        prob = np.mean(filled)

        # Confidence interval
        z_alpha = 1.96
        n = n_sims
        p_hat = prob

        denom = 1 + z_alpha**2 / n
        center = (p_hat + z_alpha**2 / (2*n)) / denom
        spread = z_alpha * np.sqrt((p_hat * (1-p_hat) + z_alpha**2 / (4*n)) / n) / denom

        ci = (max(0, center - spread), min(1, center + spread))

        # Calculate annualized vol used
        persistence = alpha + gamma/2 + beta
        annualized_vol = current_vol * np.sqrt(252) / 100

        return FillProbabilityResult(
            probability=prob,
            model="GJR-GARCH Monte Carlo",
            volatility_used=annualized_vol,
            current_price=self.S0,
            limit_price=self.B,
            horizon_days=self.horizon_days,
            direction=self.direction,
            confidence_interval=ci,
            details={
                'garch_params': {
                    'omega': omega,
                    'alpha': alpha,
                    'gamma': gamma,
                    'beta': beta,
                    'nu': nu
                },
                'persistence': persistence,
                'current_daily_vol_pct': current_vol,
                'n_simulations': n_sims,
                'note': 'RECOMMENDED - captures fat tails and vol clustering'
            }
        )
    
    def bootstrap_historical(self,
                             returns: pd.Series,
                             block_size: int = 5,
                             n_sims: int = 10000,
                             seed: Optional[int] = None) -> FillProbabilityResult:
        """
        Block bootstrap simulation using historical returns.
        
        Preserves the empirical distribution and short-term
        autocorrelation structure without parametric assumptions.
        
        Args:
            returns: Historical log returns (decimal)
            block_size: Block size for preserving autocorrelation
            n_sims: Number of simulations
            seed: Random seed
            
        Returns:
            FillProbabilityResult
        """
        if seed is not None:
            np.random.seed(seed)
        
        returns_array = returns.values
        n = len(returns_array)
        
        if n < block_size * 2:
            raise ValueError(f"Need at least {block_size * 2} returns for bootstrap")
        
        fill_count = 0
        
        for _ in range(n_sims):
            # Build path by sampling blocks
            sampled_returns = []
            
            while len(sampled_returns) < self.horizon_days:
                # Random block start
                start = np.random.randint(0, n - block_size)
                block = returns_array[start:start + block_size].tolist()
                sampled_returns.extend(block)
            
            # Trim to exact horizon
            sampled_returns = sampled_returns[:self.horizon_days]
            
            # Compute price path
            cumulative_returns = np.cumsum(sampled_returns)
            price_path = self.S0 * np.exp(cumulative_returns)
            
            # Check fill condition
            if self.direction == FillDirection.BUY:
                if np.min(price_path) <= self.B:
                    fill_count += 1
            else:
                if np.max(price_path) >= self.B:
                    fill_count += 1
        
        prob = fill_count / n_sims
        
        # Confidence interval
        z_alpha = 1.96
        n_sim = n_sims
        p_hat = prob
        
        denom = 1 + z_alpha**2 / n_sim
        center = (p_hat + z_alpha**2 / (2*n_sim)) / denom
        spread = z_alpha * np.sqrt((p_hat * (1-p_hat) + z_alpha**2 / (4*n_sim)) / n_sim) / denom
        
        ci = (max(0, center - spread), min(1, center + spread))
        
        return FillProbabilityResult(
            probability=prob,
            model="Block Bootstrap",
            volatility_used=self.sigma,
            current_price=self.S0,
            limit_price=self.B,
            horizon_days=self.horizon_days,
            direction=self.direction,
            confidence_interval=ci,
            details={
                'block_size': block_size,
                'n_simulations': n_sims,
                'history_length': n,
                'note': 'Non-parametric - uses actual return distribution'
            }
        )


def calculate_p70_limit(current_price: float,
                        volatility: float,
                        horizon_days: int,
                        direction: FillDirection,
                        target_prob: float = 0.70) -> float:
    """
    Calculate the limit price that gives target fill probability.
    
    This inverts the fill probability function to find the price
    that corresponds to a specific fill probability (default 70%).
    
    Uses GBM closed-form as it's fast and provides a good approximation.
    For more accuracy, adjust the result by ~10-20% to account for
    fat tails (which increase fill probability for same distance).
    
    Args:
        current_price: Current market price
        volatility: Annualized volatility
        horizon_days: Time horizon in trading days
        direction: BUY or SELL
        target_prob: Target fill probability (default 0.70)
        
    Returns:
        Limit price that achieves approximately target_prob fill probability
    """
    T = horizon_days / 252
    mu = 0.04 - 0.5 * volatility**2  # Assume 4% risk-free rate
    sigma_sqrt_T = volatility * np.sqrt(T)
    
    def prob_error(limit_price: float) -> float:
        """Error function to minimize"""
        calc = FillProbabilityCalculator(
            current_price=current_price,
            limit_price=limit_price,
            horizon_days=horizon_days,
            volatility=volatility
        )
        result = calc.gbm_closed_form()
        return (result.probability - target_prob)**2
    
    # Set bounds based on direction
    if direction == FillDirection.BUY:
        # Search below current price
        lower = current_price * 0.5
        upper = current_price * 0.999
    else:
        # Search above current price
        lower = current_price * 1.001
        upper = current_price * 1.5
    
    # Find optimal limit price
    result = minimize_scalar(prob_error, bounds=(lower, upper), method='bounded')
    
    return result.x


class InsufficientDataError(Exception):
    """Raised when historical data is insufficient for proper analysis."""
    pass


def analyze_order(symbol: str,
                  current_price: float,
                  limit_price: float,
                  horizon_days: int,
                  returns_pct: pd.Series,
                  ohlc_df: Optional[pd.DataFrame] = None,
                  implied_vol: Optional[float] = None,
                  require_garch: bool = True) -> Dict:
    """
    Comprehensive analysis of a limit order using GJR-GARCH.

    Args:
        symbol: Ticker symbol (for labeling)
        current_price: Current market price
        limit_price: Limit order price
        horizon_days: Time horizon in trading days
        returns_pct: Historical returns in percentage (REQUIRED for GARCH)
        ohlc_df: OHLC DataFrame (for Yang-Zhang vol)
        implied_vol: Market implied volatility (optional)
        require_garch: If True (default), require GARCH analysis

    Returns:
        Dictionary with analysis results and recommendations

    Raises:
        InsufficientDataError: If require_garch=True and insufficient data
    """
    # Validate data upfront
    if require_garch:
        if returns_pct is None:
            raise InsufficientDataError(
                f"{symbol}: returns_pct is required for GARCH analysis"
            )
        if len(returns_pct) < 100:
            raise InsufficientDataError(
                f"{symbol}: Need at least 100 days of returns, got {len(returns_pct)}"
            )

    results = {'symbol': symbol}

    # Calculate volatility estimates
    vol_estimates = {}

    if returns_pct is not None:
        returns_decimal = returns_pct / 100
        vol_estimates['close_to_close'] = VolatilityEstimator.close_to_close(returns_decimal)
        vol_estimates['ewma'] = VolatilityEstimator.ewma(returns_decimal, lambda_param=0.97)

        vol_estimates['garch'], garch_details = VolatilityEstimator.fit_garch(returns_pct)
        results['garch_params'] = garch_details

    if ohlc_df is not None:
        vol_estimates['yang_zhang'] = VolatilityEstimator.yang_zhang(ohlc_df)

    if implied_vol is not None:
        vol_estimates['implied'] = implied_vol

    results['volatility_estimates'] = vol_estimates

    # Select best volatility - GARCH is required when require_garch=True
    if implied_vol is not None:
        best_vol = implied_vol * 0.85
        vol_source = 'implied (adjusted)'
    elif vol_estimates.get('garch') is not None:
        best_vol = vol_estimates['garch']
        vol_source = 'garch'
    elif require_garch:
        raise InsufficientDataError(
            f"{symbol}: GARCH volatility estimation failed"
        )
    elif vol_estimates.get('yang_zhang') is not None:
        best_vol = vol_estimates['yang_zhang']
        vol_source = 'yang_zhang (FALLBACK)'
    elif vol_estimates.get('ewma') is not None:
        best_vol = vol_estimates['ewma']
        vol_source = 'ewma (FALLBACK)'
    elif vol_estimates.get('close_to_close') is not None:
        best_vol = vol_estimates['close_to_close']
        vol_source = 'close_to_close (FALLBACK)'
    else:
        raise InsufficientDataError(
            f"{symbol}: No volatility data available"
        )

    results['volatility_used'] = best_vol
    results['volatility_source'] = vol_source

    # Initialize calculator
    calc = FillProbabilityCalculator(
        current_price=current_price,
        limit_price=limit_price,
        horizon_days=horizon_days,
        volatility=best_vol
    )

    # Run models
    model_results = {}

    # 1. GBM Closed-form (baseline - for reference only)
    gbm_result = calc.gbm_closed_form()
    model_results['gbm'] = {
        'probability': gbm_result.probability,
        'model': gbm_result.model
    }

    # 2. Student's t Monte Carlo (for reference only)
    t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
    model_results['student_t'] = {
        'probability': t_result.probability,
        'confidence_interval': t_result.confidence_interval,
        'model': t_result.model
    }

    # 3. GARCH Monte Carlo (PRIMARY MODEL)
    if returns_pct is not None and len(returns_pct) >= 100:
        garch_result = calc.garch_monte_carlo(returns_pct, n_sims=50000)
        model_results['garch_mc'] = {
            'probability': garch_result.probability,
            'confidence_interval': garch_result.confidence_interval,
            'model': garch_result.model,
            'nu': garch_result.details['garch_params']['nu']
        }

    # 4. Bootstrap (if returns available)
    if returns_pct is not None:
        returns_decimal = returns_pct / 100
        bootstrap_result = calc.bootstrap_historical(returns_decimal, n_sims=10000)
        model_results['bootstrap'] = {
            'probability': bootstrap_result.probability,
            'confidence_interval': bootstrap_result.confidence_interval,
            'model': bootstrap_result.model
        }

    results['model_results'] = model_results

    # Summary statistics
    probs = [r.get('probability', 0) for r in model_results.values()
             if isinstance(r, dict) and 'probability' in r]

    if probs:
        results['summary'] = {
            'mean_probability': np.mean(probs),
            'min_probability': np.min(probs),
            'max_probability': np.max(probs),
            'spread': np.max(probs) - np.min(probs)
        }

    # Recommendation - use GARCH only
    direction = calc.direction
    distance_pct = abs(limit_price - current_price) / current_price * 100

    if 'garch_mc' in model_results and 'probability' in model_results['garch_mc']:
        recommended_prob = model_results['garch_mc']['probability']
    elif require_garch:
        raise InsufficientDataError(
            f"{symbol}: GARCH Monte Carlo failed but require_garch=True"
        )
    else:
        recommended_prob = model_results['student_t']['probability']

    # Calculate P70 target
    p70_price = calculate_p70_limit(
        current_price, best_vol, horizon_days, direction
    )
    p70_distance = abs(p70_price - current_price) / current_price * 100

    results['recommendation'] = {
        'current_fill_probability': recommended_prob,
        'p70_limit_price': p70_price,
        'p70_distance_pct': p70_distance,
        'current_distance_pct': distance_pct,
        'adjustment_needed': p70_price - limit_price if direction == FillDirection.BUY else limit_price - p70_price,
        'status': 'GOOD' if recommended_prob >= 0.65 else 'ADJUST' if recommended_prob >= 0.40 else 'TOO_AGGRESSIVE'
    }

    return results


if __name__ == "__main__":
    # Example usage
    print("Limit Order Fill Probability Calculator")
    print("=" * 50)
    
    # Example: VOO at $633, limit at $614.82 (P70 target), 60-day horizon
    current = 633.0
    limit = 614.82
    horizon = 60
    vol = 0.1275  # 12.75% annualized
    
    calc = FillProbabilityCalculator(
        current_price=current,
        limit_price=limit,
        horizon_days=horizon,
        volatility=vol
    )
    
    print(f"\nExample: VOO")
    print(f"Current Price: ${current:.2f}")
    print(f"Limit Price: ${limit:.2f}")
    print(f"Distance: {(current - limit)/current * 100:.2f}%")
    print(f"Horizon: {horizon} days")
    print(f"Volatility: {vol*100:.2f}%")
    
    # GBM result
    gbm = calc.gbm_closed_form()
    print(f"\nGBM Closed-Form: {gbm.probability*100:.1f}%")
    
    # Student's t result
    t_result = calc.student_t_monte_carlo(nu=4.0, n_sims=50000)
    print(f"Student's t (ν=4): {t_result.probability*100:.1f}% "
          f"[{t_result.confidence_interval[0]*100:.1f}%, "
          f"{t_result.confidence_interval[1]*100:.1f}%]")
    
    print("\nNote: For full analysis with GARCH, provide historical returns.")
