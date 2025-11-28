"""
Visualize Predictions

Generates plots comparing actual stock returns vs predictions from
v0 (Baseline), v1 (VIX), and v2 (VIX+FRED) models.
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments_daily_volatility.daily_data_loader import DailyDataLoader

try:
    from chronos import Chronos2Pipeline
    CHRONOS2_AVAILABLE = True
except ImportError:
    CHRONOS2_AVAILABLE = False


def visualize_forecasts():
    print("="*70)
    print("VISUALIZING PREDICTIONS")
    print("="*70)
    
    if not CHRONOS2_AVAILABLE:
        print("❌ Chronos 2 not available")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    loader = DailyDataLoader()
    
    # Tickers to visualize
    tickers = ['NSRGY', 'UBS', '0QLR.LON']
    
    # Models to compare
    versions = {
        'v0_baseline': {'path': './daily_results/models/v0_baseline', 'features': [], 'color': 'gray', 'style': '--'},
        'v1_vix': {'path': './daily_results/models/v1_vix', 'features': ['vix'], 'color': 'blue', 'style': '-.'},
        'v2_vix_fred': {'path': './daily_results/models/v2_vix_fred', 'features': ['vix', 'fedfunds', 't10y2y'], 'color': 'green', 'style': '-'},
    }
    
    # Load models
    models = {}
    print("\nLoading models...")
    for name, config in versions.items():
        path = Path(config['path'])
        if path.exists():
            try:
                models[name] = {
                    'pipeline': Chronos2Pipeline.from_pretrained(
                        str(path),
                        device_map=device,
                        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                    ),
                    'config': config
                }
                print(f"✓ Loaded {name}")
            except Exception as e:
                print(f"❌ Failed to load {name}: {e}")
    
    if not models:
        print("No models loaded!")
        return

    output_dir = Path("./daily_results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    for ticker in tickers:
        print(f"\nGenerating plot for {ticker}...")
        
        try:
            # We want to predict the *last* 5 days of available data to compare with actuals
            # So we load data, and split: Context = All except last 5 days, Truth = Last 5 days
            
            # Load data (using v2 features to ensure we have all data needed for all models)
            # We'll filter features later for v0/v1
            max_features = versions['v2_vix_fred']['features']
            batch = loader.create_training_batch(ticker, max_features)
            
            target = batch['target']
            prediction_length = 5
            
            if len(target) < 100:
                print(f"Not enough data for {ticker}")
                continue
                
            # Split
            context = target[:-prediction_length]
            ground_truth = target[-prediction_length:]
            
            # Prepare plot
            plt.figure(figsize=(12, 6))
            
            # Plot Ground Truth
            # We plot the last 20 days of context + 5 days of truth
            plot_context_len = 20
            x_context = np.arange(-plot_context_len, 0)
            x_pred = np.arange(0, prediction_length)
            
            plt.plot(x_context, context[-plot_context_len:], color='black', label='History', linewidth=1.5)
            plt.plot(x_pred, ground_truth, color='black', label='Actual (Truth)', linewidth=2.5, marker='o')
            
            # Generate predictions for each model
            for name, model_info in models.items():
                pipeline = model_info['pipeline']
                config = model_info['config']
                features_needed = config['features']
                
                # Prepare input
                input_data = [{'target': context.astype(np.float32)}]
                
                if features_needed and 'past_covariates' in batch:
                    input_data[0]['past_covariates'] = {}
                    for feat in features_needed:
                        # Slice covariates to match context length
                        input_data[0]['past_covariates'][feat] = batch['past_covariates'][feat][:-prediction_length].astype(np.float32)
                
                # Predict
                forecast = pipeline.predict(input_data, prediction_length=prediction_length)
                
                # Get median prediction
                # forecast shape: (batch, num_samples, prediction_length) -> we want median sample
                # Chronos predict returns QuantileForecast or SampleForecast. 
                # The pipeline.predict returns a generator or list of forecasts.
                # Let's inspect the return type. Chronos2Pipeline.predict returns torch tensor of shape (batch_size, num_samples, prediction_length)
                
                # forecast is likely a tuple/list where the first element is the tensor
                forecast_tensor = forecast[0] 
                if hasattr(forecast_tensor, 'cpu'):
                    forecast_tensor = forecast_tensor.cpu().numpy()
                
                # forecast_tensor shape: (batch_size, num_samples/quantiles, prediction_length)
                # We have batch_size=1
                
                # If shape is (1, 21, prediction_length), it's likely quantiles.
                # Median is at index 10.
                if len(forecast_tensor.shape) == 3 and forecast_tensor.shape[1] == 21:
                    median_pred = forecast_tensor[0, 10, :]
                    low_pred = forecast_tensor[0, 2, :]  # ~10% quantile
                    high_pred = forecast_tensor[0, 18, :] # ~90% quantile
                else:
                    # Fallback if shape is different (e.g. samples)
                    # Assuming shape (batch, samples, len) -> take median over samples
                    median_pred = np.median(forecast_tensor[0], axis=0)
                    low_pred = np.quantile(forecast_tensor[0], 0.1, axis=0)
                    high_pred = np.quantile(forecast_tensor[0], 0.9, axis=0)
                
                # Plot
                plt.plot(x_pred, median_pred, 
                        color=config['color'], 
                        linestyle=config['style'], 
                        label=f"{name} (Pred)",
                        linewidth=2)
                
                # Optional: Plot 80% prediction interval for v2 (best model)
                if name == 'v2_vix_fred':
                    plt.fill_between(x_pred, low_pred, high_pred, color=config['color'], alpha=0.1, label=f"{name} 80% CI")

            plt.title(f"Forecast Comparison: {ticker} (Last 5 Days)", fontsize=14, fontweight='bold')
            plt.xlabel("Days (0 = Forecast Start)", fontsize=12)
            plt.ylabel("Log Returns", fontsize=12)
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left', frameon=True)
            
            # Save
            save_path = output_dir / f"{ticker}_forecast_comparison.png"
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"✓ Saved plot to {save_path}")
            
        except Exception as e:
            print(f"❌ Error for {ticker}: {e}")
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("VISUALIZATION COMPLETE")
    print("="*70)

if __name__ == "__main__":
    visualize_forecasts()
