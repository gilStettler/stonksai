"""
EWMA + VIX Volatility Inference - All Stocks
=============================================

Tests all SMI stocks with the optimal EWMA + VIX configuration (K=2).
Generates individual plots and aggregate metrics.

Optimal Features (from ablation study):
- ewma_vol_lag_1
- vix_lag_1

This configuration achieves:
- MAE: ~0.00130 (11.3% better than EWMA K=1 without VIX)
- R²: ~94.3%
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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "processed_data")
VIX_PATH = os.path.join(SCRIPT_DIR, "..", "data_fred", "vixcls.csv")
OUTPUT_DIR = SCRIPT_DIR
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

DATE_COL = "timestamp"
TARGET_COL = "ewma_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"

# EWMA Settings
LAMBDA = 0.94  # RiskMetrics standard

# OPTIMAL K=2 Features (from ablation study with VIX)
FEATURE_COLS = ["ewma_vol_lag_1", "vix_lag_1"]

# Rolling forecast settings
SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80

# Device
DEVICE = "cuda"


def calculate_ewma_volatility(returns, lambda_=0.94):
    """Calculate EWMA volatility."""
    n = len(returns)
    ewma_var = np.zeros(n)
    returns_filled = returns.fillna(0).values
    ewma_var[0] = returns_filled[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (returns_filled[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


def load_vix():
    """Load and prepare VIX data."""
    vix = pd.read_csv(VIX_PATH)
    vix.columns = ['date', 'vix']
    vix['date'] = pd.to_datetime(vix['date'])
    vix['vix'] = pd.to_numeric(vix['vix'], errors='coerce')
    vix = vix.dropna()
    vix['vix'] = vix['vix'] / 100  # Normalize
    vix['vix_lag_1'] = vix['vix'].shift(1)
    return vix


def prepare_data(filepath, vix_df):
    """Load and prepare stock data with EWMA and VIX."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # Calculate log returns
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Calculate EWMA
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df["ewma_vol"].shift(1)
    
    # Merge VIX
    df = df.merge(vix_df[['date', 'vix_lag_1']], left_on=DATE_COL, right_on='date', how='left')
    df = df.drop(columns=['date'], errors='ignore')
    
    # Business day reindexing
    pieces = []
    for sid, g in df.groupby(ID_COL):
        g = g.sort_values(DATE_COL).drop_duplicates(subset=[DATE_COL])
        g = g.set_index(DATE_COL)
        bdays_idx = pd.date_range(start=g.index.min(), end=g.index.max(), freq=FREQ)
        g = g.reindex(bdays_idx).ffill()
        g[ID_COL] = sid
        g.index.name = DATE_COL
        pieces.append(g.reset_index())
    
    return pd.concat(pieces, ignore_index=True).dropna()


def run_prediction(pipeline, df, stock_name):
    """Run prediction for a single stock."""
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
    df_split = df[df[DATE_COL] >= pd.Timestamp(SPLIT_START_DATE)]
    df_split = df_split.sort_values(DATE_COL).reset_index(drop=True)
    
    if len(df_split) < 50:
        return None
    
    n_train = int(np.floor(TRAIN_FRACTION * len(df_split)))
    context_df = df_split.iloc[:n_train].copy()
    future_df = df_split.iloc[n_train:].copy()
    
    if len(future_df) == 0:
        return None
    
    pred_len = len(future_df)
    
    context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    future_cols = [ID_COL, DATE_COL] + FEATURE_COLS
    
    context_cols = [c for c in context_cols if c in context_df.columns]
    future_cols = [c for c in future_cols if c in future_df.columns]
    
    try:
        pred_df = pipeline.predict_df(
            context_df[context_cols].reset_index(drop=True),
            future_df=future_df[future_cols].reset_index(drop=True),
            prediction_length=pred_len,
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=ID_COL,
            timestamp_column=DATE_COL,
            target=TARGET_COL
        )
        
        results = pred_df.merge(
            future_df[[ID_COL, DATE_COL, TARGET_COL]],
            on=[ID_COL, DATE_COL],
            how="left"
        )
        
        y_true = results[TARGET_COL].astype(float).values
        y_pred = results["0.5"].astype(float).values
        
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
            "q10": results["0.1"].astype(float).tolist(),
            "q90": results["0.9"].astype(float).tolist(),
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


def plot_stock_forecast(result, output_dir):
    """Generate forecast plot for a single stock."""
    stock = result["stock"]
    dates = pd.to_datetime(result["dates"])
    y_pred = result["predictions"]
    y_true = result["actuals"]
    q10 = result["q10"]
    q90 = result["q90"]
    context_df = result["context_df"]
    
    pred_len = len(y_pred)
    
    plt.figure(figsize=(14, 6))
    
    # Historical
    hist = context_df[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).tail(pred_len)
    plt.plot(hist[DATE_COL], hist[TARGET_COL], label="Historical", color='blue', alpha=0.7, linewidth=1.5)
    
    # True
    plt.plot(dates, y_true, label="True", color='green', marker='o', markersize=2, linewidth=1.5)
    
    # Prediction
    plt.plot(dates, y_pred, label="Prediction", color='red', linestyle='--', marker='x', markersize=3, linewidth=2)
    
    # Confidence interval
    plt.fill_between(dates, q10, q90, alpha=0.2, color='red', label="10-90%")
    
    # Forecast start
    plt.axvline(dates.min(), color='gray', linestyle='--', linewidth=1, label="Forecast start")
    
    mae = result["mae"]
    r2 = result["r2"]
    
    plt.title(f"EWMA + VIX Forecast: {stock}\nMAE: {mae:.6f} | R²: {r2:.4f}")
    plt.xlabel("Date")
    plt.ylabel("EWMA Volatility")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_file = os.path.join(PLOTS_DIR, f"ewma_vix_{stock}.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()


def plot_aggregate_metrics(stock_metrics, output_dir):
    """Generate aggregate metrics plot."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    stocks = [m["Stock"] for m in stock_metrics]
    maes = [m["MAE"] for m in stock_metrics]
    rmses = [m["RMSE"] for m in stock_metrics]
    r2s = [m["R2"] for m in stock_metrics]
    adj_r2s = [m["Adj_R2"] for m in stock_metrics]
    
    # MAE
    ax = axes[0, 0]
    ax.bar(range(len(stocks)), maes, color='steelblue')
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('MAE')
    ax.set_title('MAE by Stock')
    ax.axhline(np.mean(maes), color='red', linestyle='--', label=f'Mean: {np.mean(maes):.6f}')
    ax.legend()
    
    # RMSE
    ax = axes[0, 1]
    ax.bar(range(len(stocks)), rmses, color='darkorange')
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('RMSE')
    ax.set_title('RMSE by Stock')
    ax.axhline(np.mean(rmses), color='red', linestyle='--', label=f'Mean: {np.mean(rmses):.6f}')
    ax.legend()
    
    # R²
    ax = axes[1, 0]
    colors = ['green' if r > 0.9 else 'orange' if r > 0.8 else 'red' for r in r2s]
    ax.bar(range(len(stocks)), r2s, color=colors)
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('R²')
    ax.set_title('R² by Stock')
    ax.axhline(np.mean(r2s), color='blue', linestyle='--', label=f'Mean: {np.mean(r2s):.4f}')
    ax.legend()
    
    # Adjusted R² (Replaces N_Predictions)
    ax = axes[1, 1]
    colors_adj = ['green' if r > 0.9 else 'orange' if r > 0.8 else 'red' for r in adj_r2s]
    ax.bar(range(len(stocks)), adj_r2s, color=colors_adj)
    ax.set_xticks(range(len(stocks)))
    ax.set_xticklabels(stocks, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Adj R²')
    ax.set_title('Adjusted R² by Stock')
    ax.axhline(np.mean(adj_r2s), color='blue', linestyle='--', label=f'Mean: {np.mean(adj_r2s):.4f}')
    ax.legend()
    
    plt.suptitle("EWMA + VIX Volatility Forecast Metrics (K=2)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "ewma_vix_aggregate_metrics.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Aggregate plot saved: {plot_file}")


def main():
    print("=" * 60)
    print("EWMA + VIX Volatility Inference - All Stocks")
    print("=" * 60)
    print(f"Target: {TARGET_COL}")
    print(f"Features: {FEATURE_COLS}")
    print(f"Lambda: {LAMBDA}")
    print(f"Device: {DEVICE}")
    print("=" * 60)
    
    # Load model
    print("\nLoading Chronos-2 pipeline...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    print("Model loaded!")
    
    # Load VIX
    print("\nLoading VIX data...")
    vix_df = load_vix()
    print(f"VIX data: {len(vix_df)} rows")
    
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
        stock_name = stock_name.split("_")[0]
        
        try:
            df = prepare_data(filepath, vix_df)
            result = run_prediction(pipeline, df, stock_name)
            
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
                
                # Generate plot for this stock
                plot_stock_forecast(result, OUTPUT_DIR)
                print(f"\n{stock_name}: MAE={result['mae']:.6f}, R²={result['r2']:.4f}")
                
        except Exception as e:
            print(f"\nError processing {stock_name}: {e}")
            continue
    
    if not results:
        print("No results!")
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
    
    # Save metrics
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
    
    metrics_file = os.path.join(OUTPUT_DIR, "ewma_vix_all_stocks_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print(f"\nMetrics saved: {metrics_file}")
    
    # Generate aggregate plot
    plot_aggregate_metrics(stock_metrics, OUTPUT_DIR)
    
    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)


if __name__ == "__main__":
    main()
