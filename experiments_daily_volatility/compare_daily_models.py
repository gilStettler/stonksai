"""
Compare Daily Models - Evaluation across all model versions

Evaluates baseline and fine-tuned models with different covariates
on daily stock return forecasting.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
from datetime import datetime
import warnings
import warnings
warnings.filterwarnings('ignore')
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_daily_volatility.daily_data_loader import DailyDataLoader

try:
    from chronos import Chronos2Pipeline
    CHRONOS2_AVAILABLE = True
except ImportError:
    CHRONOS2_AVAILABLE = False


SMI_TICKERS = [
    'NSRGY', 'UBS', 'ABB NY', '0QLR.LON', '0QKI.LON', '0QP2.LON',
    '0QKY.LON', '0QNO.LON', '0QPS.LON', '0A0D.LON', '0Z4C.LON',
    '0QOQ.LON', '0QMG.LON', '0QQ2.LON', '0QMW.LON', '0QK6.LON',
]


def evaluate_model(model, ticker: str, loader: DailyDataLoader, 
                  covariate_features: list, model_name: str,
                  test_size: float = 0.2,
                  include_correlations: bool = False) -> dict:
    """Evaluate a model on one ticker."""
    
    try:
        # Load data with specified covariates
        batch = loader.create_training_batch(
            ticker=ticker,
            covariate_features=covariate_features,
            include_correlations=include_correlations
        )
        
        target = batch['target']
        
        if len(target) < 200:
            return None
        
        # Split
        split_idx = int(len(target) * (1 - test_size))
        context = target[:split_idx]
        test = target[split_idx:]
        
        # Prepare input
        input_data = [{'target': context.astype(np.float32)}]
        
        # Add covariates if present
        if 'past_covariates' in batch:
            input_data[0]['past_covariates'] = {}
            for cov_name, cov_data in batch['past_covariates'].items():
                input_data[0]['past_covariates'][cov_name] = cov_data[:split_idx].astype(np.float32)
        
        # Predict
        predictions = model.predict(input_data, prediction_length=len(test))
        
        # Extract median
        pred_tensor = predictions[0]
        if hasattr(pred_tensor, 'cpu'):
            pred_tensor = pred_tensor.cpu().numpy()
        pred_median = pred_tensor[0, 10, :]  # Median quantile
        
        # Metrics
        actual_len = min(len(test), len(pred_median))
        mae = np.mean(np.abs(test[:actual_len] - pred_median[:actual_len]))
        rmse = np.sqrt(np.mean((test[:actual_len] - pred_median[:actual_len]) ** 2))
        
        # Volatility Metrics (Magnitude prediction)
        vol_actual = np.abs(test[:actual_len])
        vol_pred = np.abs(pred_median[:actual_len])
        vol_mae = np.mean(np.abs(vol_actual - vol_pred))
        
        # Direction Accuracy
        # Use sign of returns. 0 is considered same as positive for simplicity or handle separately
        dir_actual = np.sign(test[:actual_len])
        dir_pred = np.sign(pred_median[:actual_len])
        direction_acc = np.mean(dir_actual == dir_pred)
        
        # --- NEW METRICS ---
        
        # Adjusted R^2
        r2 = r2_score(test[:actual_len], pred_median[:actual_len])
        n = actual_len
        p = len(covariate_features) + (1 if include_correlations else 0) # approx number of features
        if n - p - 1 > 0:
            adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        else:
            adj_r2 = np.nan
            
        # Quantile Losses & CRPS
        # Need full forecast tensor: (batch, samples, len)
        # We assume batch=1
        if hasattr(predictions[0], 'cpu'):
            forecast_tensor = predictions[0].cpu().numpy()
        else:
            forecast_tensor = predictions[0].numpy()
            
        # Check if we have quantiles or samples
        # Chronos usually returns samples by default unless quantile_levels specified in predict
        # But pipeline.predict returns samples.
        # We can approximate quantiles from samples.
        
        quantile_losses = {}
        quantiles = [0.1, 0.5, 0.9]
        
        # If shape is (1, 21, len), it might be quantiles if we used predict(quantile_levels=...)
        # But here we used default predict. Let's assume samples and compute quantiles.
        # forecast_tensor shape: (1, num_samples, len)
        
        samples = forecast_tensor[0] # (num_samples, len)
        
        for q in quantiles:
            # Calculate q-th quantile for each time step
            yhat_q = np.quantile(samples, q, axis=0)[:actual_len]
            
            # Pinball loss
            diff = test[:actual_len] - yhat_q
            loss = np.mean(np.maximum(q * diff, (q - 1) * diff))
            quantile_losses[f"q{q}"] = float(loss)
            
        # CRPS Approximation (using trapezoid rule on quantile losses)
        # CRPS = 2 * integral(QuantileLoss(q) dq)
        # We have 3 points: 0.1, 0.5, 0.9. Rough approx.
        # Better: compute more quantiles for CRPS or use analytical if distribution known.
        # Notebook used trapezoid on available quantiles.
        qs = np.array(quantiles)
        Ls = np.array([quantile_losses[f"q{q}"] for q in quantiles])
        crps = 2 * np.trapz(Ls, qs)
        
        return {
            'ticker': ticker,
            'model': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'mae': float(mae),
            'rmse': float(rmse),
            'vol_mae': float(vol_mae),
            'direction_acc': float(direction_acc),
            'adj_r2': float(adj_r2),
            'crps': float(crps),
            'quantile_loss_q0.1': quantile_losses['q0.1'],
            'quantile_loss_q0.5': quantile_losses['q0.5'],
            'quantile_loss_q0.9': quantile_losses['q0.9'],
            'n_test': actual_len,
            'n_test': actual_len,
        }
        
    except Exception as e:
        print(f"    Error evaluating {ticker}: {e}")
        return None


def compare_daily_models():
    """Compare all daily model versions."""
    print("="*70)
    print("DAILY MODEL COMPARISON - ABLATION STUDY")
    print("="*70)
    
    if not CHRONOS2_AVAILABLE:
        print("❌ Chronos 2 not available")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DailyDataLoader()
    
    # Define model versions to compare
    versions = {
        'v0_baseline': {'path': './daily_results/models/v0_baseline', 'features': []},
        'v1_vix': {'path': './daily_results/models/v1_vix', 'features': ['vix']},
        'v2_vix_fred': {'path': './daily_results/models/v2_vix_fred', 'features': ['vix', 'fedfunds', 't10y2y']},
        'v3_full': {'path': './daily_results/models/v3_full', 'features': ['vix', 'fedfunds', 't10y2y'], 'include_correlations': True},
    }
    
    # Load all models
    models = {}
    print("\nLoading models...")
    
    for version_name, config in versions.items():
        model_path = Path(config['path'])
        
        if not model_path.exists():
            print(f"  ⚠️  {version_name}: Not found at {model_path}")
            continue
        
        try:
            model = Chronos2Pipeline.from_pretrained(
                str(model_path),
                device_map=device,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )
            models[version_name] = {
                'model': model, 
                'features': config['features'],
                'include_correlations': config.get('include_correlations', False)
            }
            print(f"  ✓ {version_name}")
        except Exception as e:
            print(f"  ❌ {version_name}: {e}")
    
    if not models:
        print("\n❌ No models found. Run finetune_daily_multivariate.py first!")
        return
    
    # Evaluate all models on all tickers
    results = {version: [] for version in models.keys()}
    
    for i, ticker in enumerate(SMI_TICKERS, 1):
        print(f"\n[{i}/{len(SMI_TICKERS)}] Evaluating {ticker}...")
        
        for version_name, model_config in models.items():
            model = model_config['model']
            features = model_config['features']
            
            result = evaluate_model(
                model=model,
                ticker=ticker,
                loader=loader,
                covariate_features=features,
                model_name=version_name,
                include_correlations=model_config['include_correlations']
            )
            
            if result:
                results[version_name].append(result)
                results[version_name].append(result)
                print(f"  {version_name}: MAE={result['mae']:.4f}, RMSE={result['rmse']:.4f}, CRPS={result['crps']:.4f}, R2_adj={result['adj_r2']:.4f}")
    
    # Summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    summary_data = []
    
    for version_name, version_results in results.items():
        if version_results:
            mae_values = [r['mae'] for r in version_results]
            rmse_values = [r['rmse'] for r in version_results]
            crps_values = [r['crps'] for r in version_results]
            r2_values = [r['adj_r2'] for r in version_results]
            q01_values = [r['quantile_loss_q0.1'] for r in version_results]
            q05_values = [r['quantile_loss_q0.5'] for r in version_results]
            q09_values = [r['quantile_loss_q0.9'] for r in version_results]
            
            summary_data.append({
                'Version': version_name,
                'Avg MAE': np.mean(mae_values),
                'Avg RMSE': np.mean(rmse_values),
                'Avg CRPS': np.mean(crps_values),
                'Avg R2': np.nanmean(r2_values), # Use nanmean as R2 can be nan
                'QL 0.1': np.mean(q01_values),
                'QL 0.5': np.mean(q05_values),
                'QL 0.9': np.mean(q09_values),
                'Stocks': len(version_results),
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n" + summary_df.to_string(index=False))
    
    # Calculate improvements
    if 'v0_baseline' in results and results['v0_baseline']:
        baseline_mae = summary_df[summary_df['Version'] == 'v0_baseline']['Avg MAE'].values[0]
        
        print("\n" + "-"*70)
        print("IMPROVEMENTS vs BASELINE")
        print("-"*70)
        
        for _, row in summary_df.iterrows():
            if row['Version'] != 'v0_baseline':
                improvement = (baseline_mae - row['Avg MAE']) / baseline_mae * 100
                symbol = "✅" if improvement > 0 else "❌"
                print(f"{symbol} {row['Version']:15s}: {improvement:+.1f}%")
    
    # Save results
    output_dir = Path("./daily_results/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "comparison.json"
    with open(output_file, 'w') as f:
        json.dump({
            **results,
            'summary': summary_data,
            'evaluated_at': datetime.now().isoformat(),
        }, f, indent=2)
    
    print(f"\n✓ Results saved to: {output_file}")
    print("\nNext step: Run covariate_impact_analyzer.py to analyze impact")
    print("="*70 + "\n")


if __name__ == "__main__":
    compare_daily_models()
