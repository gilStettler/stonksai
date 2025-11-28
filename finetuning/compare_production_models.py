"""
Compare Production Models: Baseline vs Production Fine-tuned

Runs comprehensive evaluation on all 16 SMI tickers.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from app import (
    fetch_intraday_cached,
    compute_logrv_auto,
    SMI_TICKERS,
    SMI_NAMES,
    TZ
)

try:
    from chronos import Chronos2Pipeline
    CHRONOS2_AVAILABLE = True
except ImportError:
    CHRONOS2_AVAILABLE = False


def evaluate_ticker(model, ticker, model_name, test_size=0.2):
    """Evaluate model on one ticker."""
    
    try:
        # Load data
        df = fetch_intraday_cached(ticker, interval="5m", period="60d", tz=TZ, use_cache=True)
        logrv, logvol, meta = compute_logrv_auto(df, horizons_minutes=60)
        
        if len(logrv) < 100:
            return None
        
        # Split
        split_idx = int(len(logrv) * (1 - test_size))
        context = logrv.iloc[:split_idx].values
        test = logrv.iloc[split_idx:].values
        
        # Predict
        input_data = [{"target": context.astype(np.float32)}]
        predictions = model.predict(input_data, prediction_length=len(test))
        
        # Extract median (quantile 0.5)
        pred_tensor = predictions[0]
        if hasattr(pred_tensor, 'cpu'):
            pred_tensor = pred_tensor.cpu().numpy()
        pred_median = pred_tensor[0, 10, :]  # Median quantile
        
        # Metrics
        actual_len = min(len(test), len(pred_median))
        mae = np.mean(np.abs(test[:actual_len] - pred_median[:actual_len]))
        rmse = np.sqrt(np.mean((test[:actual_len] - pred_median[:actual_len]) ** 2))
        
        # Additional metrics
        mape = np.mean(np.abs((test[:actual_len] - pred_median[:actual_len]) / (test[:actual_len] + 1e-8))) * 100
        
        return {
            'ticker': ticker,
            'name': SMI_NAMES.get(ticker, ticker),
            'model': model_name,
            'mae': float(mae),
            'rmse': float(rmse),
            'mape': float(mape),
            'n_test': actual_len,
        }
        
    except Exception as e:
        print(f"    Error evaluating {ticker}: {e}")
        return None


def compare_production_models():
    """
    Compare baseline vs production fine-tuned model on all 16 SMI tickers.
    """
    
    print("="*70)
    print("PRODUCTION MODEL COMPARISON - 16 SMI AKTIEN")
    print("="*70)
    
    if not CHRONOS2_AVAILABLE:
        print("❌ Chronos 2 not available")
        return
    
    # Load models
    print("\nLoading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    baseline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("  ✓ Baseline loaded")
    
    # Check for fine-tuned model
    finetuned_path = Path("./production_finetuning_fast/model")
    if not finetuned_path.exists():
        print(f"\n❌ Fine-tuned model not found at {finetuned_path}")
        print("   Run finetune_production.py first!")
        return
    
    finetuned = Chronos2Pipeline.from_pretrained(
        str(finetuned_path),
        device_map=device,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    print("  ✓ Fine-tuned loaded")
    
    # Evaluate all tickers
    results = {'baseline': [], 'finetuned': []}
    
    for i, ticker in enumerate(SMI_TICKERS, 1):
        print(f"\n[{i}/{len(SMI_TICKERS)}] {ticker} ({SMI_NAMES.get(ticker, ticker)})")
        
        # Baseline
        print("  Baseline...", end=" ", flush=True)
        base_res = evaluate_ticker(baseline, ticker, "baseline")
        if base_res:
            results['baseline'].append(base_res)
            print(f"MAE={base_res['mae']:.4f}")
        else:
            print("Failed")
        
        # Fine-tuned
        print("  Fine-tuned...", end=" ", flush=True)
        ft_res = evaluate_ticker(finetuned, ticker, "finetuned")
        if ft_res:
            results['finetuned'].append(ft_res)
            print(f"MAE={ft_res['mae']:.4f}")
            
            if base_res and ft_res:
                improvement = (base_res['mae'] - ft_res['mae']) / base_res['mae'] * 100
                symbol = "✅" if improvement > 0 else "❌"
                print(f"  {symbol} Change: {improvement:+.1f}%")
        else:
            print("Failed")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY - ALL 16 SMI TICKERS")
    print("="*70)
    
    if results['baseline'] and results['finetuned']:
        base_df = pd.DataFrame(results['baseline'])
        ft_df = pd.DataFrame(results['finetuned'])
        
        print(f"\n{'Ticker':<10} {'Baseline MAE':<15} {'Fine-tuned MAE':<15} {'Improvement':<12}")
        print("-"*70)
        
        for _, base_row in base_df.iterrows():
            ticker = base_row['ticker']
            ft_row = ft_df[ft_df['ticker'] == ticker]
            
            if not ft_row.empty:
                ft_mae = ft_row.iloc[0]['mae']
                improvement = (base_row['mae'] - ft_mae) / base_row['mae'] * 100
                symbol = "✅" if improvement > 0 else "❌"
                print(f"{ticker:<10} {base_row['mae']:<15.4f} {ft_mae:<15.4f} {symbol} {improvement:+.1f}%")
        
        print("-"*70)
        print(f"{'AVERAGE':<10} {base_df['mae'].mean():<15.4f} {ft_df['mae'].mean():<15.4f}", end=" ")
        
        avg_improvement = (base_df['mae'].mean() - ft_df['mae'].mean()) / base_df['mae'].mean() * 100
        symbol = "✅" if avg_improvement > 0 else "❌"
        print(f"{symbol} {avg_improvement:+.1f}%")
        
        # Count improvements
        improvements = []
        for _, base_row in base_df.iterrows():
            ticker = base_row['ticker']
            ft_row = ft_df[ft_df['ticker'] == ticker]
            if not ft_row.empty:
                imp = (base_row['mae'] - ft_row.iloc[0]['mae']) / base_row['mae'] * 100
                improvements.append(imp)
        
        positive_count = sum(1 for imp in improvements if imp > 0)
        
        print(f"\n📊 Performance Summary:")
        print(f"  Improved tickers:    {positive_count}/{len(improvements)}")
        print(f"  Average improvement: {avg_improvement:+.1f}%")
        print(f"  RMSE improvement:    {(base_df['rmse'].mean() - ft_df['rmse'].mean()) / base_df['rmse'].mean() * 100:+.1f}%")
        
        # Save results
        output_dir = Path("./production_evaluation")
        output_dir.mkdir(exist_ok=True)
        
        with open(output_dir / "comparison.json", 'w') as f:
            json.dump({
                'baseline': results['baseline'],
                'finetuned': results['finetuned'],
                'summary': {
                    'avg_mae_baseline': float(base_df['mae'].mean()),
                    'avg_mae_finetuned': float(ft_df['mae'].mean()),
                    'avg_improvement_pct': float(avg_improvement),
                    'tickers_improved': int(positive_count),
                    'total_tickers': len(improvements),
                    'evaluated_at': datetime.now().isoformat(),
                }
            }, f, indent=2)
        
        print(f"\n✓ Results saved to {output_dir / 'comparison.json'}")
        
        # Decision
        print("\n" + "="*70)
        print("DEPLOYMENT RECOMMENDATION")
        print("="*70)
        
        if avg_improvement > 10:
            print("✅ DEPLOY FINE-TUNED MODEL")
            print(f"   Improvement: {avg_improvement:.1f}% exceeds 10% threshold")
        elif avg_improvement > 5:
            print("⚠️  CONSIDER DEPLOYMENT")
            print(f"   Improvement: {avg_improvement:.1f}% is moderate")
        elif avg_improvement > 0:
            print("⏸️  USE BASELINE")
            print(f"   Improvement: {avg_improvement:.1f}% too small")
        else:
            print("❌ USE BASELINE")
            print(f"   Fine-tuned is worse: {avg_improvement:.1f}%")
    
    return results


if __name__ == "__main__":
    compare_production_models()
