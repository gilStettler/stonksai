
"""
04_compare_results.py

Purpose:
    Compares the predictions from different models against the actual data.
    Calculates metrics (R2, Adj R2, RMSE, MAE, Correlation) and saves comparison CSVs.
    
    It supports comparing:
    - Zero-Shot predictions
    - Top-6 Feature predictions
    - All Features predictions
    - Fine-Tuned predictions

Usage:
    python 04_compare_results.py
"""

import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import os
import glob

# Configuration
ACTUALS_DIR = "actuals_data"
TARGET_DATE = "2025-12-01"

# Files to compare
COMPARISONS = {
    "Zero-Shot": "predictions_2025_12_01_zeroshot.csv",
    "Top-6 Features": "predictions_2025_12_01_top6.csv",
    "All Features": "predictions_2025_12_01_all_features.csv",
    "Fine-Tuned": "predictions_finetuned.csv"
}

def compare_file(model_name, pred_file):
    print(f"\nEvaluating: {model_name} ({pred_file})")
    
    if not os.path.exists(pred_file):
        print(f"  File not found: {pred_file}")
        return
        
    preds_df = pd.read_csv(pred_file)
    actuals = []
    
    # Load Actuals
    csv_files = glob.glob(os.path.join(ACTUALS_DIR, "*.csv"))
    for filepath in csv_files:
        filename = os.path.basename(filepath)
        stock_name = filename.replace("data_", "").replace(".csv", "")
        
        try:
            df = pd.read_csv(filepath)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            row = df[df["timestamp"] == pd.Timestamp(TARGET_DATE)]
            
            if not row.empty:
                actual_vol = row["ctc_vol"].values[0]
                actuals.append({
                    "Stock": stock_name,
                    "Actual_Vol": actual_vol
                })
        except Exception as e:
            pass
            
    actuals_df = pd.DataFrame(actuals)
    if actuals_df.empty:
        print("  No actual data found.")
        return

    # Merge
    comparison = pd.merge(preds_df, actuals_df, on="Stock", how="inner")
    
    # Calculate Metrics
    y_true = comparison["Actual_Vol"]
    y_pred = comparison["Volatility"]
    n = len(y_true)
    
    # Determine p (number of predictors) for Adjusted R2
    if "zero" in model_name.lower():
        p = 0
    elif "top-6" in model_name.lower() or "fine" in model_name.lower():
        p = 6
    else:
        p = 56 # All features
        
    r2 = r2_score(y_true, y_pred)
    
    # Adjusted R2
    if n > p + 1:
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        adj_r2 = np.nan
        
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    
    print(f"  R² Score:     {r2:.4f}")
    print(f"  Adj. R²:      {adj_r2:.4f}")
    print(f"  RMSE:         {rmse:.6f}")
    print(f"  MAE:          {mae:.6f}")
    print(f"  Correlation:  {corr:.4f}")
    
    # Save comparison
    save_name = f"comparison_{model_name.lower().replace(' ', '_')}.csv"
    comparison.to_csv(save_name, index=False)
    print(f"  Saved details to {save_name}")

if __name__ == "__main__":
    print(f"Comparing predictions for {TARGET_DATE}...")
    for name, file in COMPARISONS.items():
        compare_file(name, file)
