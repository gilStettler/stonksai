"""
Daily Multivariate Fine-tuning with Ablation Study

Trains 4 model versions to measure covariate impact:
- V0_baseline: No covariates (just returns)
- V1_vix: Only VIX
- V2_vix_fred: VIX + FRED features (fedfunds, t10y2y)
- V3_full: VIX + FRED + cross-stock correlations (future)

This allows measuring the marginal impact of each covariate group.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_daily_volatility.daily_data_loader import DailyDataLoader
from experiments_daily_volatility.fred_features import get_features_by_tier

try:
    from chronos import Chronos2Pipeline
    CHRONOS2_AVAILABLE = True
except ImportError:
    CHRONOS2_AVAILABLE = False
    print("⚠️ Chronos 2 not available")


# Configuration
SMI_TICKERS = [
    'NSRGY',   # Nestlé
    'UBS',     # UBS
    'ABBNY',   # ABB
    '0QLR.LON', # Novartis
    '0QKI.LON', # Swisscom
    '0QP2.LON', # Zurich Insurance
    '0QKY.LON', # Holcim
    '0QNO.LON', # Lonza
    '0QPS.LON', # Givaudan
    '0A0D.LON', # Alcon
    '0Z4C.LON', # Sika
    '0QOQ.LON', # Partners Group
    '0QMG.LON', # Swiss Life
    '0QQ2.LON', # Geberit
    '0QMW.LON', # Kuehne + Nagel
    '0QK6.LON', # Logitech
]


def prepare_data_for_version(loader: DailyDataLoader,
                            tickers: list,
                            version_name: str,
                            covariate_features: list,
                            test_size: float = 0.2,
                            include_correlations: bool = False):
    """
    Prepare training and validation data for a model version.
    
    Args:
        loader: DailyDataLoader instance
        tickers: List of stock tickers
        version_name: Name of the model version
        covariate_features: List of FRED feature names
        test_size: Fraction for validation split
    
    Returns:
        Tuple of (train_inputs, val_inputs, metadata)
    """
    print(f"\n{'='*70}")
    print(f"PREPARING DATA: {version_name}")
    print(f"Covariates: {covariate_features if covariate_features else 'None (baseline)'}")
    print(f"{'='*70}")
    
    train_inputs = []
    val_inputs = []
    metadata = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker}...")
        
        try:
            # Create training batch with specified covariates
            batch = loader.create_training_batch(
                ticker=ticker,
                covariate_features=covariate_features,
                include_correlations=include_correlations
            )
            
            target = batch['target']
            
            if len(target) < 200:
                print(f"  ⚠️ Insufficient data: {len(target)} points - SKIPPING")
                continue
            
            # Split into train/val
            split_idx = int(len(target) * (1 - test_size))
            
            # Train batch
            train_item = {'target': target[:split_idx].astype(np.float32)}
            val_item = {'target': target[split_idx:].astype(np.float32)}
            
            # Add covariates if present
            if 'past_covariates' in batch:
                train_item['past_covariates'] = {}
                val_item['past_covariates'] = {}
                
                for cov_name, cov_data in batch['past_covariates'].items():
                    train_item['past_covariates'][cov_name] = cov_data[:split_idx].astype(np.float32)
                    val_item['past_covariates'][cov_name] = cov_data[split_idx:].astype(np.float32)
            
            train_inputs.append(train_item)
            val_inputs.append(val_item)
            
            print(f"  ✓ Train: {len(train_item['target']):,} points, Val: {len(val_item['target']):,} points")
            
            metadata.append({
                'ticker': ticker,
                'train_size': len(train_item['target']),
                'val_size': len(val_item['target']),
                'has_covariates': 'past_covariates' in train_item,
                'covariates': list(train_item.get('past_covariates', {}).keys()) if 'past_covariates' in train_item else []
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*70}")
    print(f"✓ Prepared {len(train_inputs)} series for {version_name}")
    print(f"  Total training points: {sum(m['train_size'] for m in metadata):,}")
    print(f"{'='*70}")
    
    return train_inputs, val_inputs, metadata


def train_model_version(version_name: str,
                       covariate_features: list,
                       train_inputs: list,
                       val_inputs: list,
                       prediction_length: int = 5,
                       num_steps: int = 3500,
                       learning_rate: float = 1e-6,
                       batch_size: int = 8):
    """
    Train a single model version.
    
    Args:
        version_name: Name for this version
        covariate_features: List of covariates used
        train_inputs: Training data
        val_inputs: Validation data
        prediction_length: Forecast horizon
        num_steps: Training steps
        learning_rate: Learning rate
        batch_size: Batch size
    
    Returns:
        Trained pipeline or None if failed
    """
    if not CHRONOS2_AVAILABLE:
        raise ImportError("Chronos 2 not available")
    
    print(f"\n{'='*70}")
    print(f"TRAINING: {version_name}")
    print(f"{'='*70}")
    
    output_dir = Path(f"./daily_results/models/{version_name}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nDevice: {device}")
    
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    print("\nLoading base model...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("✓ Model loaded")
    
    print(f"\n{'-'*70}")
    print("CONFIGURATION")
    print(f"{'-'*70}")
    print(f"  Version:              {version_name}")
    print(f"  Covariates:           {covariate_features if covariate_features else 'None'}")
    print(f"  Training series:      {len(train_inputs)}")
    print(f"  Validation series:    {len(val_inputs)}")
    print(f"  Steps:                {num_steps:,}")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Prediction length:    {prediction_length}")
    print(f"{'-'*70}")
    
    # Estimate time
    steps_per_sec = 2.0  # Conservative estimate
    estimated_mins = (num_steps / steps_per_sec) / 60
    print(f"\n⏱️  Estimated time: ~{estimated_mins:.0f} minutes")
    
    print(f"\n{'='*70}")
    print("STARTING TRAINING...")
    print(f"{'='*70}\n")
    
    try:
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            prediction_length=prediction_length,
            validation_inputs=val_inputs,
            num_steps=num_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            logging_steps=100,
        )
        
        print(f"\n{'='*70}")
        print("✅ TRAINING COMPLETE!")
        print(f"{'='*70}")
        
        # Save model
        print(f"\nSaving model to {output_dir}...")
        finetuned_pipeline.model.save_pretrained(output_dir)
        
        # Save configuration
        config = {
            'version_name': version_name,
            'covariate_features': covariate_features,
            'num_series': len(train_inputs),
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'prediction_length': prediction_length,
            'trained_at': datetime.now().isoformat(),
        }
        
        with open(output_dir / "../training_config_{}.json".format(version_name), 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model and config saved!")
        return finetuned_pipeline
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def run_ablation_study():
    """
    Run complete ablation study with 4 model versions.
    """
    print("\n" + "="*70)
    print("DAILY MULTIVARIATE FORECASTING - ABLATION STUDY")
    print("="*70)
    print("\nThis will train 4 model versions to measure covariate impact:")
    print("  1. V0_baseline: No covariates")
    print("  2. V1_vix: Only VIX")
    print("  3. V2_vix_fred: VIX + FRED (fedfunds, t10y2y)")
    print("  4. V3_full: VIX + FRED + Correlations (future)")
    print("\nEstimated total time: 4-6 hours")
    print("="*70)
    
    if not CHRONOS2_AVAILABLE:
        print("\n❌ Chronos 2 not available. Cannot proceed.")
        return
    
    # Initialize data loader
    loader = DailyDataLoader()
    
    # Define ablation versions
    # Define ablation versions
    versions = [
        # Already trained:
        # {'name': 'v0_baseline', 'features': []},
        # {'name': 'v1_vix', 'features': ['vix']},
        # {'name': 'v2_vix_fred', 'features': ['vix', 'fedfunds', 't10y2y']},
        
        # New version to train:
        {'name': 'v3_full', 'features': ['vix', 'fedfunds', 't10y2y'], 'include_correlations': True},
    ]
    
    # Training configuration
    training_config = {
        'prediction_length': 5,
        'num_steps': 3500,
        'learning_rate': 1e-6,
        'batch_size': 8,
        'test_size': 0.2,
    }
    
    # Train each version
    results = {}
    
    for version_config in versions:
        version_name = version_config['name']
        covariate_features = version_config['features']
        include_correlations = version_config.get('include_correlations', False)
        
        print(f"\n\n{'#'*70}")
        print(f"# VERSION: {version_name}")
        print(f"{'#'*70}")
        
        # Prepare data
        train_inputs, val_inputs, metadata = prepare_data_for_version(
            loader=loader,
            tickers=SMI_TICKERS,
            version_name=version_name,
            covariate_features=covariate_features,
            test_size=training_config['test_size'],
            include_correlations=include_correlations
        )
        
        if not train_inputs:
            print(f"\n❌ No training data for {version_name}. Skipping.")
            continue
        
        # Save metadata
        metadata_path = Path(f"./daily_results/models/{version_name}_metadata.json")
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        with open(metadata_path, 'w') as f:
            json.dump({
                'metadata': metadata,
                'num_series': len(train_inputs),
                'total_train_points': sum(m['train_size'] for m in metadata),
            }, f, indent=2)
        
        # Train model (extract test_size before passing kwargs)
        train_config = {k: v for k, v in training_config.items() if k != 'test_size'}
        model = train_model_version(
            version_name=version_name,
            covariate_features=covariate_features,
            train_inputs=train_inputs,
            val_inputs=val_inputs,
            **train_config
        )
        
        results[version_name] = {
            'success': model is not None,
            'num_series': len(train_inputs),
            'covariates': covariate_features,
        }
    
    # Summary
    print("\n\n" + "="*70)
    print("ABLATION STUDY COMPLETE")
    print("="*70)
    
    for version_name, result in results.items():
        status = "✓" if result['success'] else "✗"
        print(f"{status} {version_name}: {result['num_series']} series, "
              f"covariates={result['covariates']}")
    
    print(f"\nModels saved to: ./daily_results/models/")
    print("\nNext step: Run covariate_impact_analyzer.py to measure impact")
    print("="*70 + "\n")


if __name__ == "__main__":
    run_ablation_study()
