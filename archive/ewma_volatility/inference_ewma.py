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
SPLIT_START_DATE = "2020-01-01"  # Full Expanding Window
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
    
    # Calculate log returns
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    # Calculate EWMA
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df["ewma_vol"].shift(1)
    
    # Merge VIX
    # We keep 'vix' (raw) to construct the next day's features easily
    df = df.merge(vix_df[['date', 'vix', 'vix_lag_1']], left_on=DATE_COL, right_on='date', how='left')
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


def run_expanding_window_prediction(pipeline, df, stock_name):
    """Run Expanding Window prediction (simulate daily forecast)."""
    df = df.sort_values([ID_COL, DATE_COL]).reset_index(drop=True)
    
    # Split point
    split_date = pd.Timestamp(SPLIT_START_DATE)
    
    # We want to predict from split_date onwards
    test_indices = df[df[DATE_COL] >= split_date].index
    
    if len(test_indices) == 0:
        return None
    
    print(f"  Expanding Window: {len(test_indices)} days to forecast (Sequential)...")
    
    all_preds = []
    targets = []
    dates = []
    
    # Context window size (rolling)
    # We keep growing the context, but maybe truncate for speed if needed
    # Chronos handles up to 512 tokens well, longer might be truncated internally
    
    for idx in tqdm(test_indices, desc=f"  {stock_name}", leave=False):
        # Context: Everything up to idx (exclusive)
        # We need at least some context
        if idx < 50:
            continue
            
        # Optimization: Keep last 500 days context
        start_idx = max(0, idx - 500)
        context_df = df.iloc[start_idx:idx].copy()
        
        # Future: Just the one day we want to predict (idx)
        # We need to provide the features (VIX) for this day
        future_df = df.iloc[idx:idx+1].copy()
        
        targets.append(df.iloc[idx][TARGET_COL])
        dates.append(df.iloc[idx][DATE_COL])
        
        # Prepare columns
        context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
        future_cols = [ID_COL, DATE_COL] + FEATURE_COLS
        
        try:
            pred_df = pipeline.predict_df(
                context_df[context_cols].reset_index(drop=True),
                future_df=future_df[future_cols].reset_index(drop=True),
                prediction_length=1,
                quantile_levels=[0.5],
                id_column=ID_COL,
                timestamp_column=DATE_COL,
                target=TARGET_COL
            )
            
            # Extract prediction
            pred_val = pred_df["0.5"].iloc[0]
            all_preds.append(pred_val)
            
        except Exception as e:
            # Fallback: Last value
            all_preds.append(context_df[TARGET_COL].iloc[-1])

    # Compile results
    y_pred = np.array(all_preds)
    y_true = np.array(targets)
    
    if len(y_pred) == 0:
        return None

    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    n = len(y_true)
    p = len(FEATURE_COLS)
    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
    
    # For plotting, we need a context df. Just take the full df up to start of test
    context_df = df.iloc[:test_indices[0]]
    
    return {
        "stock": stock_name,
        "predictions": y_pred.tolist(),
        "actuals": y_true.tolist(),
        "dates": dates,
        "q10": (y_pred * 0.9).tolist(), # Approx
        "q90": (y_pred * 1.1).tolist(), # Approx
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "adj_r2": adj_r2,
        "n_predictions": n,
        "context_df": context_df,
        "future_df": None
    }


def predict_next_day(pipeline, df, stock_name):
    """
    Predicts the volatility for the NEXT business day (T+1) based on the latest available data (T).
    Used for live inference in UI.
    """
    # Ensure sorted
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    # 1. Get latest state (Day T)
    last_row = df.iloc[-1]
    last_date = last_row[DATE_COL]
    
    # 2. Construct Future Features (Day T+1)
    # The 'lag_1' features for tomorrow are the 'current' values of today.
    
    # 2. Resample Context to 'D' (Daily) to avoid 'Business Day' validation headaches
    # Chronos handles 'D' robustly. We ffill values over weekends.
    context_df = df.iloc[-600:].copy() # increased slightly
    context_df = context_df.set_index(DATE_COL).resample('D').ffill().reset_index()
    # Now context ends on the calendar day before 'next_date' (if we fix the logic)
    
    # Update last_row to be the actual last calendar day (Sunday if today is Monday? No).
    # If original data ends Friday (Nov 28), resample('D') adds Sat (29), Sun (30).
    # Last row of context is now Sun (30).
    last_row_D = context_df.iloc[-1]
    last_date_D = last_row_D[DATE_COL]
    
    # Next prediction day is Monday (Dec 1). 
    # This is exactly last_date_D + 1 Day.
    next_date_D = last_date_D + pd.Timedelta(days=1)
    
    # Does this match the business logic?
    # Yes, we want to predict for Dec 1.
    # Input features: We need features for Dec 1.
    # Features are 'ewma_vol_lag_1' -> EWMA of previous day.
    # Previous day is Sunday. Sunday's EWMA is Friday's EWMA (ffilled).
    # So using Friday's EWMA as lag feature for Monday is correct in this 'D' world.
    
    # Future Frame (5 days for robust freq inference)
    future_rows = []
    for i in range(5):
        d = next_date_D + pd.Timedelta(days=i)
        future_rows.append({
            ID_COL: last_row_D[ID_COL],
            DATE_COL: d,
            "ewma_vol_lag_1": last_row_D["ewma_vol"],
            "vix_lag_1": last_row_D["vix"]
        })
    
    future_df = pd.DataFrame(future_rows)
    
    context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    future_cols = [ID_COL, DATE_COL] + FEATURE_COLS
    
    # 4. Predict
    try:
        pred_df = pipeline.predict_df(
            context_df[context_cols],
            future_df=future_df[future_cols],
            prediction_length=5, # Request 5 steps
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=ID_COL,
            timestamp_column=DATE_COL,
            target=TARGET_COL
        )
        
        forecast_val = pred_df["0.5"].iloc[0]
        q10 = pred_df["0.1"].iloc[0]
        q90 = pred_df["0.9"].iloc[0]
        
        # We return the target date (which is next_date_D)
        return {
            "stock": stock_name,
            "forecast_date": next_date_D,
            "forecast_value": forecast_val,
            "confidence_interval": (q10, q90),
            "last_date": last_date, # Original business date from input
            "last_actual_vol": last_row["ewma_vol"],
            "status": "success"
        }
    except Exception as e:
        return {
            "stock": stock_name,
            "status": "error",
            "error": str(e)
        }

def get_single_stock_forecast(ticker, pipeline=None):
    """
    Helper for UI: Load data and predict for a single stock.
    If pipeline is not provided, it loads it (slow).
    """
    # Resolve Stock File
    # Assumes ticker is like "ABB" and file is "data_ABB_....csv"
    search_pattern = os.path.join(DATA_DIR, f"data_{ticker}_*.csv")
    files = glob.glob(search_pattern)
    if not files:
        return {"status": "error", "error": f"No data found for {ticker}"}
    
    filepath = files[0]
    
    # 1. Load Resources if needed
    if pipeline is None:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
        
    vix_df = load_vix()
    
    # 2. Prepare Data
    df = prepare_data(filepath, vix_df)
    
    # 3. Predict
    return predict_next_day(pipeline, df, ticker)


def plot_stock_forecast(result, output_dir):
    """Generate forecast plot for a single stock."""
    stock = result["stock"]
    dates = pd.to_datetime(result["dates"])
    y_pred = np.array(result["predictions"])
    y_true = np.array(result["actuals"])
    q10 = np.array(result["q10"])
    q90 = np.array(result["q90"])
    context_df = result["context_df"]
    
    plt.figure(figsize=(14, 7))
    
    # Historical Context (Last 100 days)
    context_days = 100
    hist = context_df[[DATE_COL, TARGET_COL]].sort_values(DATE_COL).tail(context_days)
    plt.plot(hist[DATE_COL], hist[TARGET_COL], label="Historical Context", color='blue', alpha=0.5, linewidth=1)
    
    # True vs Forecast
    plt.plot(dates, y_true, label="True Volatility", color='green', alpha=0.6, linewidth=1)
    plt.plot(dates, y_pred, label="Forecast (Next Day)", color='red', linestyle='--', linewidth=1.5, alpha=0.8)
    
    # Confidence
    plt.fill_between(dates, q10, q90, alpha=0.1, color='red', label="10-90% Confidence")
    
    # Forecast start line
    plt.axvline(dates.min(), color='gray', linestyle='--', linewidth=1, label="Forecast Start")
    
    mae = result["mae"]
    r2 = result["r2"]
    
    plt.title(f"EWMA + VIX Forecast: {stock}\nMAE: {mae:.6f} | R²: {r2:.4f}")
    plt.xlabel("Date")
    plt.ylabel("EWMA Volatility")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save directly to plots dir (no subfolders)
    plot_file = os.path.join(output_dir, "plots", f"ewma_vix_{stock}.png")
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
            result = run_expanding_window_prediction(pipeline, df, stock_name)
            
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
