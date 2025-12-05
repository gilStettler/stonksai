
"""
03_finetune_experiment.py

Purpose:
    Runs the Fine-Tuning experiment to test if training Chronos-2 on SMI data
    improves performance.
    
    It:
    1. Loads training data (up to 2024).
    2. Fine-tunes Chronos-2 for 200 steps.
    3. Predicts Dec 1, 2025 with the fine-tuned model.
    4. Saves the results for comparison.

Usage:
    python 03_finetune_experiment.py
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
import random

# Configuration
# Relative path to training data
DATA_DIR = os.path.join("..", "processed_data")
MODEL_NAME = "amazon/chronos-2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_COL = "ctc_vol"
DATE_COL = "timestamp"
ID_COL = "id"
FREQ = "D"
OUTPUT_DIR = "finetuned_model"

# Use Top-6 Features for Fine-Tuning
SELECTED_FEATURES = [
    "ctc_vol_lag_1",
    "ctc_vol_lag_2",
    "ctc_vol_lag_3",
    "abs_log_return_lag_1",
    "abs_log_return_lag_2",
    "abs_log_return_lag_3"
]

print(f"Using device: {DEVICE}")

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

def prepare_training_data():
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    train_inputs = []
    
    print("Preparing training data...")
    for filepath in tqdm(csv_files):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        df = load_and_prep_data(filepath, stock_name)
        if df is None: continue
        
        # Split: Train until 2024-12-31
        train_df = df[df[DATE_COL] <= "2024-12-31"].copy()
        
        if len(train_df) < 50: continue
        
        # Format for Chronos
        # Target
        target = torch.tensor(train_df[TARGET_COL].values, dtype=torch.float32)
        
        # Features
        feat_dict = {}
        for feat in SELECTED_FEATURES:
            if feat in train_df.columns:
                feat_dict[feat] = torch.tensor(train_df[feat].values, dtype=torch.float32)
            else:
                # Fallback if missing (should not happen with correct data)
                feat_dict[feat] = torch.zeros_like(target)
                
        train_inputs.append({
            "target": target,
            "past_covariates": feat_dict
        })
        
    return train_inputs

def train():
    pipeline = Chronos2Pipeline.from_pretrained(
        MODEL_NAME,
        device_map=DEVICE,
        torch_dtype=torch.bfloat16,
    )
    
    train_inputs = prepare_training_data()
    print(f"Training on {len(train_inputs)} series.")
    
    # Fine-tune
    # We use a small number of steps to demonstrate the effect quickly
    # But enough to potentially overfit
    pipeline.fit(
        inputs=train_inputs,
        prediction_length=10,
        num_steps=200, # 200 steps is enough for small data
        learning_rate=1e-5,
        batch_size=4
    )
    
    print(f"Saving fine-tuned model to {OUTPUT_DIR}...")
    pipeline.save_pretrained(OUTPUT_DIR)
    return pipeline

def predict_with_finetuned(pipeline):
    print("Running prediction with fine-tuned model...")
    csv_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    results = []
    
    for filepath in tqdm(csv_files):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        df = load_and_prep_data(filepath, stock_name)
        if df is None: continue

        # Prepare context/future exactly like before
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
        for feat in SELECTED_FEATURES:
            future_rows[feat] = last_row[feat] if feat in df.columns else 0
            
        full_df = pd.concat([df, future_rows], ignore_index=True)
        full_df = full_df.set_index(DATE_COL).asfreq(FREQ).reset_index().ffill()
        
        context_df = full_df.iloc[:-10].copy()
        future_df = full_df.iloc[-10:].copy()
        
        context_cols = [ID_COL, DATE_COL, TARGET_COL] + SELECTED_FEATURES
        future_cols = [ID_COL, DATE_COL] + SELECTED_FEATURES
        
        try:
            forecast = pipeline.predict_df(
                context_df[context_cols],
                future_df[future_cols],
                prediction_length=10,
                target=TARGET_COL,
                id_column=ID_COL,
                timestamp_column=DATE_COL
            )
            pred_vol = forecast['0.5'].values[target_index]
            results.append({"Stock": stock_name, "Volatility": pred_vol})
        except Exception as e:
            print(f"Error: {e}")
            
    pd.DataFrame(results).to_csv("predictions_finetuned.csv", index=False)
    print("Saved predictions_finetuned.csv")

if __name__ == "__main__":
    pipeline = train()
    predict_with_finetuned(pipeline)
