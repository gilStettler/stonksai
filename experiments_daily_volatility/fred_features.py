"""
FRED Feature Definitions and Metadata

Defines available FRED macroeconomic features for daily stock forecasting.
"""

from typing import Dict, Any

# FRED Feature Configuration
FRED_FEATURES: Dict[str, Dict[str, Any]] = {
    # Tier 1: High Impact Expected
    'vix': {
        'file': 'vixcls.csv',
        'column': 'VIXCLS',
        'freq': 'daily',
        'description': 'CBOE Volatility Index - Market volatility measure',
        'tier': 1,
    },
    'fedfunds': {
        'file': 'fedfunds.csv',
        'column': 'FEDFUNDS',
        'freq': 'monthly',
        'description': 'Federal Funds Effective Rate - Monetary policy signal',
        'tier': 1,
    },
    't10y2y': {
        'file': 't10y2y.csv',
        'column': 'T10Y2Y',
        'freq': 'daily',
        'description': '10-Year Treasury Constant Maturity Minus 2-Year - Yield curve spread',
        'tier': 1,
    },
    
    # Tier 2: Medium Impact Expected
    'dgs10': {
        'file': 'dgs10.csv',
        'column': 'DGS10',
        'freq': 'daily',
        'description': '10-Year Treasury Constant Maturity Rate - Risk-free rate proxy',
        'tier': 2,
    },
    'stlfsi4': {
        'file': 'stlfsi4.csv',
        'column': 'STLFSI4',
        'freq': 'weekly',
        'description': 'St. Louis Fed Financial Stress Index - Market stress indicator',
        'tier': 2,
    },
    
    # Tier 3: Optional/Experimental
    'unrate': {
        'file': 'unrate.csv',
        'column': 'UNRATE',
        'freq': 'monthly',
        'description': 'Unemployment Rate - Economic health indicator',
        'tier': 3,
    },
    'medcpim158sfrbcle': {
        'file': 'medcpim158sfrbcle.csv',
        'column': 'MEDCPIM158SFRBCLE',
        'freq': 'monthly',
        'description': 'Median Consumer Price Index - Inflation measure',
        'tier': 3,
    },
}


def get_features_by_tier(max_tier: int = 2) -> list:
    """
    Get FRED features up to specified tier.
    
    Args:
        max_tier: Maximum tier to include (1, 2, or 3)
    
    Returns:
        List of feature names
    """
    return [name for name, config in FRED_FEATURES.items() 
            if config['tier'] <= max_tier]


def get_daily_features() -> list:
    """Get only daily frequency FRED features."""
    return [name for name, config in FRED_FEATURES.items() 
            if config['freq'] == 'daily']


def get_feature_config(feature_name: str) -> Dict[str, Any]:
    """Get configuration for a specific feature."""
    if feature_name not in FRED_FEATURES:
        raise ValueError(f"Unknown feature: {feature_name}")
    return FRED_FEATURES[feature_name]
