"""
Test script for DailyDataLoader

Verifies data loading and alignment functionality.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_daily_volatility.daily_data_loader import DailyDataLoader
from experiments_daily_volatility.fred_features import get_features_by_tier, FRED_FEATURES

def test_alphavantage_loading():
    """Test loading AlphaVantage daily returns."""
    print("\n" + "="*70)
    print("TEST 1: AlphaVantage Returns Loading")
    print("="*70)
    
    loader = DailyDataLoader()
    
    test_tickers = ['NSRGY', 'UBS', 'ABBNY']
    
    for ticker in test_tickers:
        try:
            returns = loader.load_alphavantage_returns(ticker)
            print(f"\n✓ {ticker}:")
            print(f"  - Data points: {len(returns):,}")
            print(f"  - Date range: {returns.index.min()} to {returns.index.max()}")
            print(f"  - Mean return: {returns.mean():.6f}")
            print(f"  - Std return: {returns.std():.6f}")
        except Exception as e:
            print(f"\n✗ {ticker}: {e}")

def test_vix_loading():
    """Test loading VIX data."""
    print("\n" + "="*70)
    print("TEST 2: VIX Loading")
    print("="*70)
    
    loader = DailyDataLoader()
    
    try:
        vix = loader.load_vix()
        print(f"\n✓ VIX loaded:")
        print(f"  - Data points: {len(vix):,}")
        print(f"  - Date range: {vix.index.min()} to {vix.index.max()}")
        print(f"  - Mean: {vix.mean():.2f}")
        print(f"  - Current (last): {vix.iloc[-1]:.2f}")
    except Exception as e:
        print(f"\n✗ VIX loading failed: {e}")

def test_fred_features():
    """Test loading FRED features."""
    print("\n" + "="*70)
    print("TEST 3: FRED Features Loading (Tier 1 + 2)")
    print("="*70)
    
    loader = DailyDataLoader()
    tier1_2_features = get_features_by_tier(max_tier=2)
    
    for feature_name in tier1_2_features:
        try:
            data = loader.load_fred_feature(feature_name)
            config = FRED_FEATURES[feature_name]
            print(f"\n✓ {feature_name} ({config['freq']}):")
            print(f"  - {config['description']}")
            print(f"  - Data points: {len(data):,}")
            print(f"  - Date range: {data.index.min()} to {data.index.max()}")
            print(f"  - Missing values: {data.isna().sum()}")
        except Exception as e:
            print(f"\n✗ {feature_name}: {e}")

def test_data_alignment():
    """Test aligning multiple time series."""
    print("\n" + "="*70)
    print("TEST 4: Data Alignment")
    print("="*70)
    
    loader = DailyDataLoader()
    
    try:
        # Load some data
        returns = loader.load_alphavantage_returns('NSRGY')
        vix = loader.load_vix()
        t10y2y = loader.load_fred_feature('t10y2y')
        
        # Align
        aligned = loader.align_data([returns, vix, t10y2y], method='intersection')
        
        print(f"\n✓ Alignment successful:")
        print(f"  - Original returns: {len(returns):,} points")
        print(f"  - Original VIX: {len(vix):,} points")
        print(f"  - Original T10Y2Y: {len(t10y2y):,} points")
        print(f"  - Aligned data: {len(aligned):,} points")
        print(f"  - Date range: {aligned.index.min()} to {aligned.index.max()}")
        print(f"  - Columns: {list(aligned.columns)}")
        print(f"  - Missing values: {aligned.isna().sum().sum()}")
    except Exception as e:
        print(f"\n✗ Alignment failed: {e}")
        import traceback
        traceback.print_exc()

def test_training_batch():
    """Test creating a training batch."""
    print("\n" + "="*70)
    print("TEST 5: Training Batch Creation")
    print("="*70)
    
    loader = DailyDataLoader()
    
    # Test different covariate combinations
    test_configs = [
        {'name': 'Baseline (no covariates)', 'features': []},
        {'name': 'VIX only', 'features': ['vix']},
        {'name': 'VIX + T10Y2Y', 'features': ['vix', 't10y2y']},
        {'name': 'Tier 1 features', 'features': ['vix', 'fedfunds', 't10y2y']},
    ]
    
    ticker = 'NSRGY'
    
    for config in test_configs:
        try:
            batch = loader.create_training_batch(
                ticker=ticker,
                covariate_features=config['features']
            )
            
            print(f"\n✓ {config['name']}:")
            print(f"  - Target shape: {batch['target'].shape}")
            
            if 'past_covariates' in batch:
                print(f"  - Covariates: {list(batch['past_covariates'].keys())}")
                for cov_name, cov_data in batch['past_covariates'].items():
                    print(f"    • {cov_name}: shape {cov_data.shape}")
            else:
                print(f"  - No covariates")
                
        except Exception as e:
            print(f"\n✗ {config['name']}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("\n" + "="*70)
    print("DAILY DATA LOADER - COMPREHENSIVE TESTS")
    print("="*70)
    
    test_alphavantage_loading()
    test_vix_loading()
    test_fred_features()
    test_data_alignment()
    test_training_batch()
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70 + "\n")
