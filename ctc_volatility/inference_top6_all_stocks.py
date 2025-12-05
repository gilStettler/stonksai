"""
Chronos-2 Inference Script - Top-6 Features for All SMI Stocks
==============================================================

This script performs rolling next-day volatility forecasting for all SMI stocks
using the optimal Top-6 feature configuration identified in the ablation study.

Uses the same API as inference_chronos2.ipynb (predict_df method).

Target: ctc_vol (close-to-close volatility, window_size=5)
Features: ctc_vol_lag_1, ctc_vol_lag_2, ctc_vol_lag_3,
          abs_log_return_lag_1, abs_log_return_lag_2, abs_log_return_lag_3

Usage:
    python inference_top6_all_stocks.py
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
TARGET_COL = "ctc_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"

# Top-6 Features (optimal from ablation study)
FEATURE_COLS = [
    "ctc_vol_lag_1", "ctc_vol_lag_2", "ctc_vol_lag_3",
    "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3"
]

# Rolling forecast settings
SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80  # 80% context, 20% test

# Output directory
OUTPUT_DIR = "."
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device settings
DEVICE = "cuda"  # Change to "cpu" if no GPU available


def load_and_prep_data(filepath: str) -> pd.DataFrame:
    """Load and prepare stock data from CSV with business day reindexing."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # Keep only needed columns
    keep_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    available_cols = [c for c in keep_cols if c in df.columns]
    df = df[available_cols].copy()
    
    # Business day reindexing
    pieces = []
    for sid, g in df.groupby(ID_COL):
        g = g.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
        g = g.set_index(DATE_COL)
        start_date = g.index.min()
        end_date = g.index.max()
        bdays_idx = pd.date_range(start=start_date, end=end_date, freq=FREQ)
        g = g.reindex(bdays_idx)
        
        # Interpolate numeric columns
        value_cols = [TARGET_COL] + [c for c in FEATURE_COLS if c in g.columns]
        g[value_cols] = g[value_cols].interpolate()
        
        g[ID_COL] = sid
        g.index.name = DATE_COL
        pieces.append(g.reset_index())
    
    df_processed = pd.concat(pieces, ignore_index=True)
    return df_processed.dropna()


def run_stock_prediction(pipeline, df: pd.DataFrame, stock_name: str) -> dict:
    """Run prediction for a single stock using predict_df API."""
    
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
            "n_predictions": n,
            "context_df": context_df,
            "future_df": future_df
        }
        
    except Exception as e:
        print(f"Error predicting {stock_name}: {e}")
        return None


def plot_rolling_backtest(result: dict, context_df: pd.DataFrame, future_df: pd.DataFrame, output_dir: str):
    """Generate rolling backtest plot for a single stock."""
    stock_name = result["stock"]
    predictions = result["predictions"]
    actuals = result["actuals"]
    dates = result["dates"]
    
    pred_len = len(predictions)
    if pred_len == 0:
        return
    
    # Prepare data
    backtest_df = pd.DataFrame({
        DATE_COL: dates,
        "y_pred": predictions,
        "y_true": actuals
    })
    backtest_df[DATE_COL] = pd.to_datetime(backtest_df[DATE_COL])
    backtest_df = backtest_df.sort_values(DATE_COL)
    
    # Get historical data (same length as prediction)
    hist_part = context_df[[DATE_COL, TARGET_COL]].copy()
    hist_part = hist_part.sort_values(DATE_COL).tail(pred_len)
    
    # Get true future data
    true_future = future_df[[DATE_COL, TARGET_COL]].copy()
    true_future = true_future.sort_values(DATE_COL).head(pred_len)
    
    # Create plot
    plt.figure(figsize=(14, 7))
    
    # Plot historical values
    plt.plot(
        hist_part[DATE_COL],
        hist_part[TARGET_COL].astype(float),
        label=f"Historical {TARGET_COL} (last {pred_len} days)",
        linewidth=1.5,
        color='blue',
        alpha=0.7
    )
    
    # Plot true future values
    plt.plot(
        true_future[DATE_COL],
        true_future[TARGET_COL].astype(float),
        label=f"True {TARGET_COL} (Test Period)",
        linewidth=1.5,
        color='green',
        marker="o",
        markersize=3
    )
    
    # Plot predictions
    plt.plot(
        backtest_df[DATE_COL],
        backtest_df["y_pred"].astype(float),
        label="Chronos-2 Prediction (median)",
        linestyle="--",
        linewidth=2,
        color='red',
        marker="x",
        markersize=4
    )
    
    # Vertical line at forecast start
    split_ts = true_future[DATE_COL].min()
    plt.axvline(split_ts, color='gray', linestyle="--", linewidth=1, label="Forecast Start")
    
    # Add metrics to title
    mae = result["mae"]
    r2 = result["r2"]
    
    plt.title(f"Chronos-2 Rolling Backtest: {stock_name}\nMAE: {mae:.6f} | R²: {r2:.4f}")
    plt.xlabel("Date")
    plt.ylabel(TARGET_COL)
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Tight layout
    plt.tight_layout()
    
    # Save plot
    plot_file = os.path.join(output_dir, f"backtest_{stock_name}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()


def plot_results(stock_metrics: list, output_dir: str):
    """Generate plots for all stocks."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stocks = [m["Stock"] for m in stock_metrics]
    maes = [m["MAE"] for m in stock_metrics]
    rmses = [m["RMSE"] for m in stock_metrics]
    r2s = [m["R2"] for m in stock_metrics]
    
    # MAE
    ax = axes[0, 0]
    ax.bar(range(len(stocks)), maes, color='steelblue')
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error by Stock')
    ax.axhline(np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.6f}')
    ax.legend()
    
    # RMSE
    ax = axes[0, 1]
    ax.bar(range(len(stocks)), rmses, color='darkorange')
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error by Stock')
    ax.axhline(np.mean(rmses), color='red', linestyle='--', label=f'Mean: {np.mean(rmses):.6f}')
    ax.legend()
    
    # R²
    ax = axes[1, 0]
    colors = ['green' if r > 0 else 'red' for r in r2s]
    ax.bar(range(len(stocks)), r2s, color=colors)
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('R² Score by Stock')
    ax.axhline(np.mean(r2s), color='blue', linestyle='--', label=f'Mean: {np.mean(r2s):.4f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.legend()
    
    # Predictions count
    ax = axes[1, 1]
    n_preds = [m["N_Predictions"] for m in stock_metrics]
    ax.bar(range(len(stocks)), n_preds, color='purple')
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Predictions per Stock')
    
    plt.tight_layout()
    plot_file = os.path.join(output_dir, "top6_all_stocks_metrics.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file}")


def main():
    print("=" * 60)
    print("Chronos-2 Top-6 Features - All Stocks Inference")
    print("=" * 60)
    print(f"Target: {TARGET_COL}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Device: {DEVICE}")
    print(f"Split Start: {SPLIT_START_DATE}")
    print(f"Train Fraction: {TRAIN_FRACTION}")
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
    stock_metrics = []
    
    for filepath in tqdm(csv_files, desc="Processing stocks"):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        stock_name = stock_name.split("_")[0]  # Get ticker only
        
        try:
            df = load_and_prep_data(filepath)
            result = run_stock_prediction(pipeline, df, stock_name)
            
            if result:
                results.append(result)
                stock_metrics.append({
                    "Stock": result["stock"],
                    "MAE": result["mae"],
                    "RMSE": result["rmse"],
                    "R2": result["r2"],
                    "Adj_R2": result["adj_r2"],
                    "N_Predictions": result["n_predictions"]
                })
                # Generate backtest plot for this stock
                plot_rolling_backtest(result, result["context_df"], result["future_df"], OUTPUT_DIR)
                print(f"\n{stock_name}: MAE={result['mae']:.6f}, RMSE={result['rmse']:.6f}, R²={result['r2']:.4f}")
                
        except Exception as e:
            print(f"\nError processing {stock_name}: {e}")
            continue
    
    if not results:
        print("No results to report!")
        return
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("AGGREGATE RESULTS")
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
    
    # Save metrics to CSV
    metrics_df = pd.DataFrame(stock_metrics)
    summary_row = pd.DataFrame([{
        "Stock": "AGGREGATE",
        "MAE": avg_mae,
        "RMSE": avg_rmse,
        "R2": avg_r2,
        "Adj_R2": avg_adj_r2,
        "N_Predictions": total_preds
    }])
    metrics_df = pd.concat([metrics_df, summary_row], ignore_index=True)
    
    metrics_file = os.path.join(OUTPUT_DIR, "top6_all_stocks_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved to {metrics_file}")
    
    # Generate plot
    plot_results(stock_metrics, OUTPUT_DIR)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
