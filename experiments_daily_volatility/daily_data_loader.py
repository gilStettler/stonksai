"""
Daily Data Loader for Multivariate Stock Forecasting

Loads and aligns daily data from multiple sources:
- AlphaVantage daily stock data
- FRED macroeconomic indicators
- Cross-stock correlations
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import warnings

from fred_features import FRED_FEATURES, get_feature_config


class DailyDataLoader:
    """Loader for daily stock and macroeconomic data."""
    
    def __init__(self, base_dir: str = None):
        """
        Initialize data loader.
        
        Args:
            base_dir: Base directory containing data folders (default: parent of finetuning/)
        """
        if base_dir is None:
            # Assume we're in finetuning/ directory
            self.base_dir = Path(__file__).parent.parent
        else:
            self.base_dir = Path(base_dir)
        
        self.alphavantage_dir = self.base_dir / "alphavantage_data"
        self.fred_dir = self.base_dir / "data_fred"
        
        # Validate directories exist
        if not self.alphavantage_dir.exists():
            raise FileNotFoundError(f"AlphaVantage data directory not found: {self.alphavantage_dir}")
        if not self.fred_dir.exists():
            raise FileNotFoundError(f"FRED data directory not found: {self.fred_dir}")
    
    def load_alphavantage_returns(self, ticker: str) -> pd.Series:
        """
        Load daily returns from AlphaVantage CSV.
        
        Args:
            ticker: Stock ticker (e.g., 'NSRGY', 'UBS')
        
        Returns:
            Series of daily log returns with date index
        """
        # Find CSV file for ticker
        csv_files = list(self.alphavantage_dir.glob(f"{ticker}*_daily_data.csv"))
        
        if not csv_files:
            raise FileNotFoundError(f"No AlphaVantage data found for ticker: {ticker}")
        
        csv_file = csv_files[0]
        
        # Load data
        df = pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
        
        # Calculate log returns from close price
        df = df.sort_index()  # Ensure chronological order
        
        # Filter out rows with invalid/zero prices
        df = df[df['close'] > 0].copy()
        
        # Calculate log returns
        log_returns = np.log(df['close'] / df['close'].shift(1))
        
        # Drop NaN (first value)
        log_returns = log_returns.dropna()
        
        # Name the series
        log_returns.name = f"{ticker}_returns"
        
        return log_returns
    
    def load_vix(self) -> pd.Series:
        """
        Load VIX data from FRED.
        
        Returns:
            Series of VIX values with date index
        """
        return self.load_fred_feature('vix')
    
    def load_fred_feature(self, feature_name: str) -> pd.Series:
        """
        Load a FRED feature.
        
        Args:
            feature_name: Name of feature (from fred_features.py)
        
        Returns:
            Series of feature values with date index
        """
        config = get_feature_config(feature_name)
        
        csv_file = self.fred_dir / config['file']
        
        if not csv_file.exists():
            raise FileNotFoundError(f"FRED data file not found: {csv_file}")
        
        # Load data
        df = pd.read_csv(csv_file, parse_dates=['date'], index_col='date')
        
        # Get the data column
        column_name = config['column']
        
        if column_name not in df.columns:
            raise ValueError(f"Column {column_name} not found in {csv_file}")
        
        series = df[column_name]
        
        # Handle missing values (empty strings or '.')
        series = pd.to_numeric(series, errors='coerce')
        
        # Interpolate lower frequency data to daily (state-of-the-art method)
        if config['freq'] in ['monthly', 'weekly']:
            print(f"  ℹ️  Interpolating {feature_name} from {config['freq']} to daily (PCHIP)")
            
            # Resample to daily frequency
            series = series.resample('D').asfreq()
            
            # Use PCHIP (Piecewise Cubic Hermite Interpolating Polynomial)
            # Advantages over linear:
            # - Smoother transitions (C1 continuous)
            # - Monotone-preserving (no artificial oscillations)
            # - Widely used in econometrics for temporal disaggregation
            # Reference: Denton (1971), Chow-Lin (1971)
            series = series.interpolate(method='pchip')
            
            # For any remaining NaN at the end, use forward-fill as fallback
            series = series.ffill()
            
            # For any NaN at the beginning, use backward-fill
            series = series.bfill()
        
        series.name = feature_name
        
        return series
    
    def compute_correlations(self, tickers: List[str], 
                           window: int = 60) -> pd.DataFrame:
        """
        Compute rolling correlations between stock returns.
        
        Args:
            tickers: List of stock tickers
            window: Rolling window size in days
        
        Returns:
            DataFrame with correlation matrix for each date
        """
        # Load returns for all tickers
        returns_dict = {}
        
        for ticker in tickers:
            try:
                returns = self.load_alphavantage_returns(ticker)
                returns_dict[ticker] = returns
            except FileNotFoundError:
                warnings.warn(f"Skipping {ticker}: data not found")
                continue
        
        # Create DataFrame of returns
        returns_df = pd.DataFrame(returns_dict)
        
        # Calculate rolling correlation
        # For each ticker, we'll get top N correlated tickers
        correlations = {}
        
        for ticker in returns_df.columns:
            # Calculate correlation with all other tickers
            corr = returns_df.rolling(window=window).corr()[ticker].unstack()
            correlations[ticker] = corr
        
        return correlations
    
    def get_top_correlated(self, ticker: str, 
                          returns_df: pd.DataFrame,
                          n: int = 3,
                          window: int = 60) -> List[str]:
        """
        Get top N correlated tickers for a given ticker.
        
        Args:
            ticker: Target ticker
            returns_df: DataFrame with returns for all tickers
            n: Number of top correlated tickers to return
            window: Rolling window for correlation calculation
        
        Returns:
            List of top N correlated ticker names (excluding self)
        """
        # Calculate mean correlation over the window
        corr_matrix = returns_df.corr()
        
        # Get correlations for target ticker
        correlations = corr_matrix[ticker].drop(ticker)  # Exclude self
        
        # Sort by absolute correlation (we care about strength, not direction)
        top_corr = correlations.abs().nlargest(n)
        
        return top_corr.index.tolist()
    
    def align_data(self, series_list: List[pd.Series], 
                   method: str = 'intersection') -> pd.DataFrame:
        """
        Align multiple time series to common dates.
        
        Args:
            series_list: List of pandas Series with date index
            method: 'intersection' (only common dates) or 'union' (all dates with ffill)
        
        Returns:
            DataFrame with aligned data
        """
        # Create DataFrame from series
        df = pd.concat(series_list, axis=1)
        
        if method == 'intersection':
            # Drop rows with any NaN
            df = df.dropna()
        elif method == 'union':
            # Forward fill missing values
            df = df.ffill()
            # Drop rows that are still NaN (at the beginning)
            df = df.dropna()
        else:
            raise ValueError(f"Unknown alignment method: {method}")
        
        return df
    
    def create_training_batch(self, 
                            ticker: str,
                            covariate_features: List[str],
                            include_correlations: bool = False,
                            correlation_window: int = 60,
                            n_correlations: int = 3) -> Dict:
        """
        Create a training batch for Chronos 2 with specified covariates.
        
        Args:
            ticker: Target stock ticker
            covariate_features: List of FRED feature names to include
            include_correlations: Whether to include correlated stock returns
            correlation_window: Window for correlation calculation
            n_correlations: Number of correlated stocks to include
        
        Returns:
            Dict with 'target' and optionally 'past_covariates'
        """
        # Load target returns
        target_returns = self.load_alphavantage_returns(ticker)
        
        # Collect covariates
        covariates_dict = {}
        
        # Add FRED features
        for feature in covariate_features:
            try:
                covariate = self.load_fred_feature(feature)
                covariates_dict[feature] = covariate
            except (FileNotFoundError, ValueError) as e:
                warnings.warn(f"Failed to load feature {feature}: {e}")
                continue
        
        # Add correlated stocks if requested
        if include_correlations:
            # Find all available tickers in the directory
            all_files = list(self.alphavantage_dir.glob("*_daily_data.csv"))
            all_tickers = [f.name.split('_daily_data.csv')[0] for f in all_files]
            
            # Filter out target ticker
            other_tickers = [t for t in all_tickers if t != ticker]
            
            if not other_tickers:
                warnings.warn("No other tickers found for correlation analysis")
            else:
                # Load returns for all tickers to compute correlation
                # Optimization: In a production system, we would cache this
                returns_map = {}
                returns_map[ticker] = target_returns
                
                for t in other_tickers:
                    try:
                        r = self.load_alphavantage_returns(t)
                        # Align with target to ensure valid correlation calculation
                        # We only care about the overlapping period
                        common_idx = target_returns.index.intersection(r.index)
                        if len(common_idx) > window: # Need enough overlapping data
                            returns_map[t] = r
                    except Exception:
                        continue
                
                if len(returns_map) > 1:
                    # Create DataFrame
                    returns_df = pd.DataFrame(returns_map)
                    
                    # Get top correlated
                    top_corr_tickers = self.get_top_correlated(
                        ticker, returns_df, n=n_correlations, window=correlation_window
                    )
                    
                    print(f"  ℹ️  Top {n_correlations} correlated with {ticker}: {top_corr_tickers}")
                    
                    # Add them as covariates
                    for corr_ticker in top_corr_tickers:
                        cov_name = f"corr_{corr_ticker}"
                        covariates_dict[cov_name] = returns_map[corr_ticker]

        # Align all data
        if covariates_dict:
            all_series = [target_returns] + list(covariates_dict.values())
            # Use intersection to ensure we have data for all covariates
            aligned_df = self.align_data(all_series, method='intersection')
            
            # Create batch dict
            batch = {
                'target': aligned_df[target_returns.name].values.astype(np.float32),
                'past_covariates': {
                    name: aligned_df[name].values.astype(np.float32)
                    for name in covariates_dict.keys()
                }
            }
        else:
            # No covariates - just target
            batch = {
                'target': target_returns.values.astype(np.float32)
            }
        
        return batch
