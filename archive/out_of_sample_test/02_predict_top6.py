
"""
02_predict_top6.py

Purpose:
    Runs the prediction for Dec 1, 2025 using the "Top-6 Features" model.
    This model uses the 6 most important lagged features identified in the ablation study.
    
    Features:
    - ctc_vol_lag_1, ctc_vol_lag_2, ctc_vol_lag_3
    - abs_log_return_lag_1, abs_log_return_lag_2, abs_log_return_lag_3

Usage:
    python 02_predict_top6.py
"""

import transformers
try:
    from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration
except ImportError:
    pass

# Patch chronos.chronos2 to have T5ForConditionalGeneration
import chronos.chronos2
try:
    chronos.chronos2.T5ForConditionalGeneration = T5ForConditionalGeneration
except NameError:
    from transformers import T5ForConditionalGeneration
    chronos.chronos2.T5ForConditionalGeneration = T5ForConditionalGeneration

import os
import glob
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from tqdm import tqdm

# Configuration
# Relative path to training data
DATA_DIR = os.path.join("..", "processed_data")
MODEL_NAME = "amazon/chronos-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_COL = "ctc_vol"
DATE_COL = "timestamp"
ID_COL = "id"
FREQ = "D" 

# SELECTED FEATURES (Top-6)
SELECTED_FEATURES = [
    "ctc_vol_lag_1",
    "ctc_vol_lag_2",
    "ctc_vol_lag_3",
    "abs_log_return_lag_1",
    "abs_log_return_lag_2",
    "abs_log_return_lag_3"
]

print(f"Using device: {DEVICE}")
print(f"Using {len(SELECTED_FEATURES)} selected features: {SELECTED_FEATURES}")

def load_and_prep_data(filepath, stock_id):
    """Load and prep data exactly like run_clean_ablation.py"""
    try:
        df = pd.read_csv(filepath)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        
        if ID_COL not in df.columns:
            df[ID_COL] = stock_id
            
        # Daily reindexing
        df = df.set_index(DATE_COL)
        start_date = df.index.min()
        end_date = df.index.max()
        idx = pd.date_range(start=start_date, end=end_date, freq=FREQ)
        df = df.reindex(idx).ffill().reset_index()
        df.rename(columns={"index": DATE_COL}, inplace=True)
        df[ID_COL] = stock_id
        
        return df.dropna()
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None

def predict_all():
    # 1. Load Model
    print("Loading Chronos-2 model...")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    
    # 2. Find all files
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} stock files in {DATA_DIR}")
    
    results = []
    
    for filepath in tqdm(csv_files):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        
        # Load Data
        df = load_and_prep_data(filepath, stock_name)
        if df is None or len(df) < 50:
            continue

        # Verify features exist
        missing_feats = [f for f in SELECTED_FEATURES if f not in df.columns]
        if missing_feats:
            print(f"Skipping {stock_name}: Missing features {missing_feats}")
            continue
        
        # Create 10 future rows
        last_date = df[DATE_COL].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 11)]
        
        # Target Date: Monday, Dec 1st
        target_date = pd.Timestamp("2025-12-01")
        target_index = -1
        
        for i, d in enumerate(future_dates):
            if d.date() == target_date.date():
                target_index = i
                break
                
        if target_index == -1:
            print(f"Skipping {stock_name}: Target date {target_date.date()} out of range (Last data: {last_date.date()})")
            continue
        
        # Create future rows
        future_rows = pd.DataFrame({
            ID_COL: [stock_name] * 10,
            DATE_COL: future_dates
        })
        
        # Populate features
        last_row = df.iloc[-1]
        for feat in SELECTED_FEATURES:
            if "_lag_1" in feat:
                base_feat = feat.replace("_lag_1", "")
                if base_feat in df.columns:
                    val = last_row[base_feat]
                else:
                     val = last_row[feat] 
            else:
                val = last_row[feat]
            future_rows[feat] = val
            
        # Append and Process
        full_df = pd.concat([df, future_rows], ignore_index=True)
        
        # Ensure datatypes match
        for col in df.columns:
            if col in full_df.columns:
                full_df[col] = full_df[col].astype(df[col].dtype)

        full_df = full_df.set_index(DATE_COL).asfreq(FREQ).reset_index()
        full_df[ID_COL] = stock_name
        full_df = full_df.ffill()
        
        context_df = full_df.iloc[:-10].copy()
        future_df = full_df.iloc[-10:].copy()
        
        # Predict
        context_cols = [ID_COL, DATE_COL, TARGET_COL] + SELECTED_FEATURES
        future_cols = [ID_COL, DATE_COL] + SELECTED_FEATURES
        
        try:
            forecast = pipeline.predict_df(
                context_df[context_cols],
                future_df[future_cols],
                prediction_length=10,
                quantile_levels=[0.1, 0.5, 0.9],
                target=TARGET_COL,
                id_column=ID_COL,
                timestamp_column=DATE_COL
            )
            
            pred_vol = forecast['0.5'].values[target_index]
            low_vol = forecast['0.1'].values[target_index]
            high_vol = forecast['0.9'].values[target_index]
            
            results.append({
                "Stock": stock_name,
                "Date": target_date.date(),
                "Volatility": pred_vol,
                "Lower_80": low_vol,
                "Upper_80": high_vol,
                "Features": len(SELECTED_FEATURES)
            })
            
        except Exception as e:
            print(f"Error predicting {stock_name}: {e}")

    # Output Results
    print("\n" + "="*80)
    print(f"PREDICTIONS FOR {pd.Timestamp('2025-12-01').date()} (Top-6 Features)")
    print("="*80)
    print(f"{'Stock':<25} | {'Volatility':<12} | {'Lower 80%':<12} | {'Upper 80%':<12}")
    print("-" * 80)
    
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values("Volatility", ascending=False)
        for _, row in results_df.iterrows():
            print(f"{row['Stock']:<25} | {row['Volatility']:.6f}     | {row['Lower_80']:.6f}     | {row['Upper_80']:.6f}")
            
        # Save to CSV
        save_path = "predictions_2025_12_01_top6.csv"
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved predictions to {save_path}")
    else:
        print("No predictions generated.")

if __name__ == "__main__":
    predict_all()
