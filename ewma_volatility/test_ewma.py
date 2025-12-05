"""
EWMA Volatility Test for Chronos-2
==================================

This script tests EWMA (Exponentially Weighted Moving Average) volatility
as target instead of simple close-to-close volatility.

EWMA weights recent observations higher using decay factor λ (lambda).
σ²_t = λ * σ²_(t-1) + (1-λ) * r²_t

Standard λ = 0.94 (RiskMetrics)
"""

import os
import glob
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ============================
# Configuration
# ============================
DATA_DIR = "../processed_data"
DATE_COL = "timestamp"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"

# EWMA Settings
LAMBDA = 0.94  # RiskMetrics standard decay factor

# Output directory
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device settings
DEVICE = "cuda"

# Rolling forecast settings
SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80


def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    """
    Calculate EWMA volatility.
    
    σ²_t = λ * σ²_(t-1) + (1-λ) * r²_t
    
    Recent days are weighted higher!
    """
    n = len(returns)
    ewma_var = np.zeros(n)
    
    # Initialize with first squared return
    ewma_var[0] = returns.iloc[0] ** 2
    
    # Calculate EWMA variance
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (returns.iloc[t] ** 2)
    
    # Return volatility (sqrt of variance)
    ewma_vol = np.sqrt(ewma_var)
    return pd.Series(ewma_vol, index=returns.index)


def prepare_ewma_data(filepath: str) -> pd.DataFrame:
    """Load data and calculate EWMA volatility with lags."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # Calculate log returns
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Calculate EWMA volatility
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"].fillna(0), LAMBDA)
    
    # Create lag features
    for lag in [1, 2, 3]:
        df[f"ewma_vol_lag_{lag}"] = df["ewma_vol"].shift(lag)
    
    # Also use abs_log_return lags if available
    if "abs_log_return" not in df.columns:
        df["abs_log_return"] = np.abs(df["log_return"])
    
    for lag in [1, 2, 3]:
        df[f"abs_log_return_lag_{lag}"] = df["abs_log_return"].shift(lag)
    
    # Business day reindexing
    pieces = []
    for sid, g in df.groupby(ID_COL):
        g = g.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
        g = g.set_index(DATE_COL)
        start_date = g.index.min()
        end_date = g.index.max()
        bdays_idx = pd.date_range(start=start_date, end=end_date, freq=FREQ)
        g = g.reindex(bdays_idx)
        g = g.ffill()
        g[ID_COL] = sid
        g.index.name = DATE_COL
        pieces.append(g.reset_index())
    
    df_processed = pd.concat(pieces, ignore_index=True)
    return df_processed.dropna()


def run_ewma_prediction(pipeline, df: pd.DataFrame, stock_name: str) -> dict:
    """Run prediction with EWMA volatility as target."""
    
    TARGET_COL = "ewma_vol"
    FEATURE_COLS = [
        "ewma_vol_lag_1", "ewma_vol_lag_2", "ewma_vol_lag_3",
        "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3"
    ]
    
    # Sort data
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
    
    # Filter from split start date
    df_split = df[df[DATE_COL] >= pd.Timestamp(SPLIT_START_DATE)]
    df_split = df_split.sort_values(DATE_COL).reset_index(drop=True)
    
    if len(df_split) < 50:
        return None
    
    # Compute split points
    n_total = len(df_split)
    n_train = int(np.floor(TRAIN_FRACTION * n_total))
    
    context_df = df_split.iloc[:n_train].copy()
    future_df = df_split.iloc[n_train:].copy()
    
    if len(future_df) == 0:
        return None
    
    pred_len = len(future_df)
    
    # Build column sets
    context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    future_cols = [ID_COL, DATE_COL] + FEATURE_COLS
    
    # Filter to available columns
    context_cols = [c for c in context_cols if c in context_df.columns]
    future_cols = [c for c in future_cols if c in future_df.columns]
    
    context_df_chronos = context_df[context_cols].reset_index(drop=True)
    future_df_chronos = future_df[future_cols].reset_index(drop=True)
    
    try:
        # Run Chronos inference
        pred_df = pipeline.predict_df(
            context_df_chronos,
            future_df=future_df_chronos,
            prediction_length=pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=ID_COL,
            timestamp_column=DATE_COL,
            target=TARGET_COL
        )
        
        # Merge predictions with actuals
        results = pred_df.merge(
            future_df[[ID_COL, DATE_COL, TARGET_COL]],
            on=[ID_COL, DATE_COL],
            how="left"
        )
        
        y_true = results[TARGET_COL].astype(float).values
        y_pred = results["0.5"].astype(float).values
        
        # Calculate metrics
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        n = len(y_true)
        p = len(FEATURE_COLS)
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        return {
            "stock": stock_name,
            "predictions": y_pred.tolist(),
            "actuals": y_true.tolist(),
            "dates": results[DATE_COL].tolist(),
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "adj_r2": adj_r2,
            "n_predictions": n
        }
        
    except Exception as e:
        print(f"Error predicting {stock_name}: {e}")
        return None


def main():
    print("=" * 60)
    print("EWMA Volatility Test - Chronos-2")
    print("=" * 60)
    print(f"Lambda (decay): {LAMBDA}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load model
    print("\nLoading Chronos-2 pipeline...")
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-2",
        device_map=DEVICE
    )
    print("Model loaded successfully!")
    
    # Find all stock files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"\nFound {len(csv_files)} stock files")
    
    if not csv_files:
        print("No data files found!")
        return
    
    # Process each stock
    results = []
    
    for filepath in tqdm(csv_files, desc="Processing stocks"):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        stock_name = stock_name.split("_")[0]
        
        try:
            df = prepare_ewma_data(filepath)
            result = run_ewma_prediction(pipeline, df, stock_name)
            
            if result:
                results.append(result)
                print(f"\n{stock_name}: MAE={result['mae']:.6f}, R²={result['r2']:.4f}")
                
        except Exception as e:
            print(f"\nError processing {stock_name}: {e}")
            continue
    
    if not results:
        print("No results!")
        return
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("EWMA VOLATILITY RESULTS")
    print("=" * 60)
    
    avg_mae = np.mean([r['mae'] for r in results])
    avg_rmse = np.mean([r['rmse'] for r in results])
    avg_r2 = np.mean([r['r2'] for r in results])
    avg_adj_r2 = np.mean([r['adj_r2'] for r in results])
    total_preds = sum([r['n_predictions'] for r in results])
    
    print(f"Stocks evaluated: {len(results)}")
    print(f"Total predictions: {total_preds}")
    print(f"Average MAE: {avg_mae:.6f}")
    print(f"Average RMSE: {avg_rmse:.6f}")
    print(f"Average R²: {avg_r2:.4f}")
    print(f"Average Adjusted R²: {avg_adj_r2:.4f}")
    
    # Save results
    metrics_df = pd.DataFrame([{
        "Stock": r["stock"],
        "MAE": r["mae"],
        "RMSE": r["rmse"],
        "R2": r["r2"],
        "Adj_R2": r["adj_r2"],
        "N_Predictions": r["n_predictions"]
    } for r in results])
    
    summary_row = pd.DataFrame([{
        "Stock": "AGGREGATE",
        "MAE": avg_mae,
        "RMSE": avg_rmse,
        "R2": avg_r2,
        "Adj_R2": avg_adj_r2,
        "N_Predictions": total_preds
    }])
    metrics_df = pd.concat([metrics_df, summary_row], ignore_index=True)
    
    output_file = os.path.join(OUTPUT_DIR, "ewma_results.csv")
    metrics_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Compare with CTC results
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"EWMA (λ={LAMBDA}):  MAE={avg_mae:.6f}, R²={avg_r2:.4f}")
    print(f"(Compare with CTC-5d: MAE≈0.00234, R²≈0.87)")


if __name__ == "__main__":
    main()
