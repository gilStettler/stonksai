"""
Production Fine-tuning - OPTIMIERT für schnelleres Training

Reduzierte Parameter für praktikables Training (1-2h statt 6h)
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
import json
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
    print("⚠️ Chronos 2 not available")


def prepare_production_data(tickers, test_size=0.2, max_history_days=60):
    """Prepare all SMI tickers for production fine-tuning."""
    print("="*70)
    print("PREPARING PRODUCTION DATA - ALL SMI AKTIEN")
    print("="*70)
    
    train_inputs = []
    val_inputs = []
    metadata = []
    
    for i, ticker in enumerate(tickers, 1):
        print(f"\n[{i}/{len(tickers)}] Processing {ticker} ({SMI_NAMES.get(ticker, ticker)})")
        
        try:
            df = fetch_intraday_cached(ticker, interval="5m", period=f"{max_history_days}d", tz=TZ, use_cache=True)
            logrv, logvol, meta = compute_logrv_auto(df, horizons_minutes=60)
            
            if len(logrv) < 100:
                print(f"  ⚠️ Insufficient data: {len(logrv)} points - SKIPPING")
                continue
            
            if logvol is not None and not logvol.empty:
                common_idx = logrv.index.intersection(logvol.index)
                logrv = logrv.loc[common_idx]
                logvol = logvol.loc[common_idx]
            
            split_idx = int(len(logrv) * (1 - test_size))
            train_logrv = logrv.iloc[:split_idx]
            train_logvol = logvol.iloc[:split_idx] if logvol is not None else None
            val_logrv = logrv.iloc[split_idx:]
            val_logvol = logvol.iloc[split_idx:] if logvol is not None else None
            
            train_item = {"target": train_logrv.values.astype(np.float32)}
            val_item = {"target": val_logrv.values.astype(np.float32)}
            
            if train_logvol is not None and not train_logvol.empty:
                train_item["past_covariates"] = {"log_volume": train_logvol.values.astype(np.float32)}
                val_item["past_covariates"] = {"log_volume": val_logvol.values.astype(np.float32)}
            
            train_inputs.append(train_item)
            val_inputs.append(val_item)
            
            print(f"  ✓ Train: {len(train_logrv):,} points, Val: {len(val_logrv):,} points")
            
            metadata.append({
                'ticker': ticker,
                'name': SMI_NAMES.get(ticker, ticker),
                'train_size': len(train_logrv),
                'val_size': len(val_logrv),
                'has_volume': train_logvol is not None and not train_logvol.empty,
            })
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            continue
    
    output_dir = Path("./production_finetuning_fast")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "data_metadata.json", 'w') as f:
        json.dump({
            'num_series': len(train_inputs),
            'total_train_points': sum(m['train_size'] for m in metadata),
            'metadata': metadata,
        }, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"✓ Prepared {len(train_inputs)} series")
    print(f"  Total training points: {sum(m['train_size'] for m in metadata):,}")
    print(f"{'='*70}")
    
    return train_inputs, val_inputs, metadata


def fast_finetune(train_inputs, val_inputs):
    """Schnelleres Fine-tuning mit reduzierten aber effektiven Parametern."""
    
    if not CHRONOS2_AVAILABLE:
        raise ImportError("Chronos 2 not available")
    
    print("\n" + "="*70)
    print("FAST PRODUCTION FINE-TUNING")
    print("="*70)
    
    output_dir = Path("./production_finetuning_fast/model")
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
    
    # OPTIMIERTE Parameter für schnelleres Training
    num_steps = 3500  # Balance zwischen Training-Zeit und Konvergenz
    learning_rate = 1e-6
    batch_size = 8  # Reduziert von 16 (schneller)
    prediction_length = 3  # Reduziert von 5 (schneller)
    
    print("\n" + "-"*70)
    print("CONFIGURATION")
    print("-"*70)
    print(f"  Training series:      {len(train_inputs)}")
    print(f"  Validation series:    {len(val_inputs)}")
    print(f"  Steps:                {num_steps:,}")
    print(f"  Learning rate:        {learning_rate}")
    print(f"  Batch size:           {batch_size}")
    print(f"  Prediction length:    {prediction_length}")
    print("-"*70)
    
    # Estimate
    steps_per_sec = 4.2 / 2  # Conservative (wegen mehr Daten)
    estimated_mins = (num_steps / steps_per_sec) / 60
    
    print(f"\n⏱️  Estimated time: ~{estimated_mins:.0f} minutes")
    
    print("\n" + "="*70)
    print("STARTING TRAINING...")
    print("="*70 + "\n")
    
    try:
        finetuned_pipeline = pipeline.fit(
            inputs=train_inputs,
            prediction_length=prediction_length,
            validation_inputs=val_inputs,
            num_steps=num_steps,
            learning_rate=learning_rate,
            batch_size=batch_size,
            logging_steps=50,
        )
        
        print("\n" + "="*70)
        print("✅ TRAINING COMPLETE!")
        print("="*70)
        
        print(f"\nSaving model to {output_dir}")
        finetuned_pipeline.model.save_pretrained(output_dir)
        
        config = {
            'num_series': len(train_inputs),
            'num_steps': num_steps,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'prediction_length': prediction_length,
            'finetuned_at': datetime.now().isoformat(),
        }
        
        with open(output_dir.parent / "training_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Model saved!")
        return finetuned_pipeline
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    print("\n🚀 FAST PRODUCTION FINE-TUNING")
    print("Optimiert für ~90-120 Min Training\n")
    
    if not CHRONOS2_AVAILABLE:
        print("❌ Chronos 2 not available")
    else:
        train_inputs, val_inputs, metadata = prepare_production_data(SMI_TICKERS, test_size=0.2)
        
        if train_inputs:
            fast_finetune(train_inputs, val_inputs)
        else:
            print("❌ No data prepared")
