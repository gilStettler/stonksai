
"""
02b_predict_all_features.py

Purpose:
    Runs the prediction for Dec 1, 2025 using the "All Features" model.
    This model uses all 56 available features (noisy baseline).

Usage:
    python 02b_predict_all_features.py
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

print(f"Using device: {DEVICE}")

def get_all_features(df, exclude_cols):
    """Dynamically identify all numeric feature columns."""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    valid_features = []
    for c in numeric_cols:
        if c in exclude_cols: continue
        if "_lag_" in c or "_mean" in c:
            valid_features.append(c)
    return valid_features

def load_and_prep_data(filepath, stock_id):
    try:
        df = pd.read_csv(filepath)
        df[DATE_COL] = pd.to_datetime(df[DATE_COL])
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        
        if ID_COL not in df.columns:
            df[ID_COL] = stock_id
            
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
    print("Loading Chronos-2 model...")
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    print(f"Found {len(csv_files)} stock files in {DATA_DIR}")
    
    results = []
    
    for filepath in tqdm(csv_files):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        df = load_and_prep_data(filepath, stock_name)
        if df is None or len(df) < 50: continue

        # Identify features dynamically
        exclude = [ID_COL, DATE_COL, TARGET_COL, "close", "open", "high", "low", "volume"]
        features = get_all_features(df, exclude)
        
        # Create 10 future rows
        last_date = df[DATE_COL].max()
        future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, 11)]
        target_date = pd.Timestamp("2025-12-01")
        
        target_index = -1
        for i, d in enumerate(future_dates):
            if d.date() == target_date.date():
                target_index = i
                break
        if target_index == -1: continue
        
        future_rows = pd.DataFrame({ID_COL: [stock_name]*10, DATE_COL: future_dates})
        last_row = df.iloc[-1]
        for feat in features:
            future_rows[feat] = last_row[feat]
            
        full_df = pd.concat([df, future_rows], ignore_index=True)
        full_df = full_df.set_index(DATE_COL).asfreq(FREQ).reset_index().ffill()
        
        context_df = full_df.iloc[:-10].copy()
        future_df = full_df.iloc[-10:].copy()
        
        context_cols = [ID_COL, DATE_COL, TARGET_COL] + features
        future_cols = [ID_COL, DATE_COL] + features
        
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
            results.append({
                "Stock": stock_name,
                "Volatility": pred_vol,
                "Features": len(features)
            })
        except Exception as e:
            print(f"Error predicting {stock_name}: {e}")

    results_df = pd.DataFrame(results)
    if not results_df.empty:
        save_path = "predictions_2025_12_01_all_features.csv"
        results_df.to_csv(save_path, index=False)
        print(f"\nSaved predictions to {save_path}")

if __name__ == "__main__":
    predict_all()
