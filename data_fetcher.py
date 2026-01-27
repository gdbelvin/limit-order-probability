"""
ETF Data Fetcher
================

Fetches historical OHLC data for ETFs from multiple sources:
1. Alpaca (recommended - free with paper account)
2. Tiingo (best value - $10/month for unlimited)
3. yfinance (fallback - free but unreliable)

Also includes utilities for fetching implied volatility data.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Union
from dataclasses import dataclass
import os
import warnings


@dataclass
class DataSource:
    """Configuration for a data source"""
    name: str
    requires_api_key: bool
    api_key_env_var: Optional[str] = None
    priority: int = 0


# Available data sources in priority order
DATA_SOURCES = {
    'alpaca': DataSource('Alpaca', True, 'ALPACA_API_KEY', priority=1),
    'tiingo': DataSource('Tiingo', True, 'TIINGO_API_KEY', priority=2),
    'yfinance': DataSource('yfinance', False, None, priority=3),
}


class DataFetcher:
    """
    Unified interface for fetching ETF data from multiple sources.
    
    Usage:
        fetcher = DataFetcher()
        df = fetcher.get_ohlc('VOO', days=252)
        returns = fetcher.get_returns('VOO', days=252)
    """
    
    def __init__(self, 
                 preferred_source: Optional[str] = None,
                 alpaca_api_key: Optional[str] = None,
                 alpaca_secret_key: Optional[str] = None,
                 tiingo_api_key: Optional[str] = None):
        """
        Initialize the data fetcher.
        
        Args:
            preferred_source: 'alpaca', 'tiingo', or 'yfinance'
            alpaca_api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            alpaca_secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            tiingo_api_key: Tiingo API key (or set TIINGO_API_KEY env var)
        """
        self.alpaca_api_key = alpaca_api_key or os.environ.get('ALPACA_API_KEY')
        self.alpaca_secret_key = alpaca_secret_key or os.environ.get('ALPACA_SECRET_KEY')
        self.tiingo_api_key = tiingo_api_key or os.environ.get('TIINGO_API_KEY')
        
        # Determine available sources
        self.available_sources = self._check_available_sources()
        
        # Set preferred source
        if preferred_source and preferred_source in self.available_sources:
            self.preferred_source = preferred_source
        elif self.available_sources:
            self.preferred_source = self.available_sources[0]
        else:
            raise RuntimeError(
                "No data sources available. Install yfinance (pip install yfinance) "
                "or configure Alpaca/Tiingo API keys."
            )
        
        print(f"Data source: {self.preferred_source}")
        print(f"Available sources: {', '.join(self.available_sources)}")
    
    def _check_available_sources(self) -> List[str]:
        """Check which data sources are available"""
        available = []
        
        # Check Alpaca
        if self.alpaca_api_key and self.alpaca_secret_key:
            try:
                from alpaca.data import StockHistoricalDataClient
                available.append('alpaca')
            except ImportError:
                pass
        
        # Check Tiingo
        if self.tiingo_api_key:
            try:
                from tiingo import TiingoClient
                available.append('tiingo')
            except ImportError:
                pass
        
        # Check yfinance (always available if installed)
        try:
            import yfinance
            available.append('yfinance')
        except ImportError:
            pass
        
        return available
    
    def get_ohlc(self, 
                 symbol: str, 
                 days: int = 252,
                 end_date: Optional[datetime] = None,
                 source: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch OHLC data for a symbol.
        
        Args:
            symbol: Ticker symbol (e.g., 'VOO')
            days: Number of trading days to fetch
            end_date: End date (default: today)
            source: Override preferred source
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
        """
        source = source or self.preferred_source
        
        if source == 'alpaca':
            return self._fetch_alpaca_ohlc(symbol, days, end_date)
        elif source == 'tiingo':
            return self._fetch_tiingo_ohlc(symbol, days, end_date)
        elif source == 'yfinance':
            return self._fetch_yfinance_ohlc(symbol, days, end_date)
        else:
            raise ValueError(f"Unknown source: {source}")
    
    def get_returns(self,
                    symbol: str,
                    days: int = 252,
                    return_type: str = 'log',
                    as_percentage: bool = False,
                    end_date: Optional[datetime] = None,
                    source: Optional[str] = None) -> pd.Series:
        """
        Calculate returns from price data.
        
        Args:
            symbol: Ticker symbol
            days: Number of trading days
            return_type: 'log' or 'simple'
            as_percentage: If True, multiply by 100
            end_date: End date
            source: Data source override
            
        Returns:
            Series of returns
        """
        # Fetch extra days to account for return calculation
        df = self.get_ohlc(symbol, days + 10, end_date, source)
        
        if return_type == 'log':
            returns = np.log(df['close'] / df['close'].shift(1))
        else:
            returns = df['close'].pct_change()
        
        returns = returns.dropna().tail(days)
        
        if as_percentage:
            returns = returns * 100
        
        returns.name = symbol
        return returns
    
    def get_multiple_symbols(self,
                             symbols: List[str],
                             days: int = 252,
                             end_date: Optional[datetime] = None,
                             source: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLC data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            days: Number of trading days
            end_date: End date
            source: Data source override
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        result = {}
        
        for symbol in symbols:
            try:
                result[symbol] = self.get_ohlc(symbol, days, end_date, source)
                print(f"  Fetched {symbol}: {len(result[symbol])} days")
            except Exception as e:
                print(f"  Error fetching {symbol}: {e}")
        
        return result
    
    def _fetch_alpaca_ohlc(self, 
                           symbol: str, 
                           days: int,
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """Fetch data from Alpaca"""
        from alpaca.data import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        
        client = StockHistoricalDataClient(
            self.alpaca_api_key,
            self.alpaca_secret_key
        )
        
        end = end_date or datetime.now()
        # Add buffer for non-trading days
        start = end - timedelta(days=int(days * 1.5))
        
        request = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=TimeFrame.Day,
            start=start,
            end=end
        )
        
        bars = client.get_stock_bars(request)
        df = bars.df
        
        if isinstance(df.index, pd.MultiIndex):
            df = df.loc[symbol]
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        df = df.rename(columns={
            'vwap': 'vwap',
            'trade_count': 'trades'
        })
        
        # Keep only needed columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        return df.tail(days)
    
    def _fetch_tiingo_ohlc(self,
                           symbol: str,
                           days: int,
                           end_date: Optional[datetime]) -> pd.DataFrame:
        """Fetch data from Tiingo"""
        from tiingo import TiingoClient
        
        config = {'api_key': self.tiingo_api_key, 'session': True}
        client = TiingoClient(config)
        
        end = end_date or datetime.now()
        start = end - timedelta(days=int(days * 1.5))
        
        df = client.get_dataframe(
            symbol,
            startDate=start.strftime('%Y-%m-%d'),
            endDate=end.strftime('%Y-%m-%d'),
            frequency='daily'
        )
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Tiingo uses 'adjClose', 'adjHigh', etc. - use adjusted values
        rename_map = {
            'adjopen': 'open',
            'adjhigh': 'high',
            'adjlow': 'low',
            'adjclose': 'close',
            'adjvolume': 'volume'
        }
        
        # First try adjusted columns
        for old, new in rename_map.items():
            if old in df.columns:
                df = df.rename(columns={old: new})
        
        # Keep only needed columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        return df.tail(days)
    
    def _fetch_yfinance_ohlc(self,
                              symbol: str,
                              days: int,
                              end_date: Optional[datetime]) -> pd.DataFrame:
        """Fetch data from yfinance (fallback)"""
        import yfinance as yf
        
        end = end_date or datetime.now()
        start = end - timedelta(days=int(days * 1.5))
        
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start, end=end)
        
        # Standardize column names
        df.columns = df.columns.str.lower()
        
        # Keep only needed columns
        cols = ['open', 'high', 'low', 'close', 'volume']
        df = df[[c for c in cols if c in df.columns]]
        
        return df.tail(days)


class ImpliedVolatilityFetcher:
    """
    Fetch implied volatility data from free sources.
    
    Sources:
    - Option Strategist (free weekly data)
    - CBOE VIX-related products
    """
    
    @staticmethod
    def get_iv_from_optionstrategist(symbol: str) -> Optional[Dict]:
        """
        Scrape IV data from Option Strategist's free weekly report.
        
        Note: This is weekly data, updated on Saturdays.
        
        Returns:
            Dictionary with hv20, hv50, hv100, current_iv, iv_percentile
        """
        import requests
        
        url = "https://www.optionstrategist.com/calculators/free-volatility-data"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            
            # Parse the text data (it's in a specific format)
            lines = response.text.split('\n')
            
            for line in lines:
                if symbol.upper() in line:
                    parts = line.split()
                    if len(parts) >= 8:
                        return {
                            'symbol': symbol,
                            'hv20': float(parts[1]) / 100,
                            'hv50': float(parts[2]) / 100,
                            'hv100': float(parts[3]) / 100,
                            'current_iv': float(parts[5]) / 100,
                            'iv_percentile': float(parts[6].rstrip('%ile')) / 100,
                            'source': 'optionstrategist'
                        }
            
            return None
            
        except Exception as e:
            warnings.warn(f"Could not fetch IV from Option Strategist: {e}")
            return None
    
    @staticmethod
    def get_vix() -> Optional[float]:
        """
        Fetch current VIX level as a proxy for S&P 500 implied volatility.
        
        Returns:
            VIX level as decimal (e.g., 0.15 for VIX at 15)
        """
        try:
            import yfinance as yf
            vix = yf.Ticker('^VIX')
            hist = vix.history(period='1d')
            if not hist.empty:
                return hist['Close'].iloc[-1] / 100
        except Exception:
            pass
        
        return None


def fetch_portfolio_data(symbols: List[str],
                         days: int = 252,
                         preferred_source: Optional[str] = None,
                         **kwargs) -> Dict:
    """
    Convenience function to fetch data for a portfolio of symbols.
    
    Args:
        symbols: List of ticker symbols
        days: Number of trading days
        preferred_source: 'alpaca', 'tiingo', or 'yfinance'
        **kwargs: Additional arguments for DataFetcher
        
    Returns:
        Dictionary with OHLC DataFrames and returns
    """
    fetcher = DataFetcher(preferred_source=preferred_source, **kwargs)
    
    result = {
        'ohlc': {},
        'returns_pct': {},
        'returns_decimal': {},
        'current_prices': {},
        'volatility': {}
    }
    
    print(f"Fetching data for {len(symbols)} symbols...")
    
    for symbol in symbols:
        try:
            # Fetch OHLC
            ohlc = fetcher.get_ohlc(symbol, days)
            result['ohlc'][symbol] = ohlc
            
            # Calculate returns
            result['returns_pct'][symbol] = fetcher.get_returns(
                symbol, days, as_percentage=True
            )
            result['returns_decimal'][symbol] = fetcher.get_returns(
                symbol, days, as_percentage=False
            )
            
            # Current price
            result['current_prices'][symbol] = ohlc['close'].iloc[-1]
            
            # Simple volatility estimate
            result['volatility'][symbol] = result['returns_decimal'][symbol].std() * np.sqrt(252)
            
            print(f"  {symbol}: ${result['current_prices'][symbol]:.2f}, "
                  f"vol={result['volatility'][symbol]*100:.1f}%")
            
        except Exception as e:
            print(f"  Error with {symbol}: {e}")
    
    return result


if __name__ == "__main__":
    # Example usage
    print("ETF Data Fetcher")
    print("=" * 50)
    
    # Example ETF list
    symbols = [
        'VOO', 'VBR', 'VSS', 'VXUS', 'VWO',  # Equities
        'VGIT', 'VGSH', 'VGLT', 'VTIP', 'BNDX',  # Bonds
        'DBMF', 'CAOS'  # Alternatives
    ]
    
    # Try yfinance as fallback (no API key needed)
    try:
        fetcher = DataFetcher(preferred_source='yfinance')
        
        # Fetch one symbol as example
        print("\nFetching VOO data...")
        df = fetcher.get_ohlc('VOO', days=60)
        print(f"Got {len(df)} days of data")
        print(f"Latest close: ${df['close'].iloc[-1]:.2f}")
        
        # Calculate returns
        returns = fetcher.get_returns('VOO', days=60, as_percentage=True)
        print(f"Returns (last 5): {returns.tail().values}")
        print(f"Annualized vol: {returns.std() / 100 * np.sqrt(252) * 100:.2f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        print("\nTo use this module, install at least one data source:")
        print("  pip install yfinance  # Free, no API key")
        print("  pip install alpaca-py  # Free with paper account")
        print("  pip install tiingo  # $10/month for unlimited")
