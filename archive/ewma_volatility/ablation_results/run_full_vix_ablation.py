"""
Complete EWMA Ablation Study with VIX (K=1 to K=20)
====================================================

Tests all feature combinations including VIX.
VIX is placed early in the feature list since it showed promise.
"""

import os
import glob
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Configuration
DATA_DIR = "processed_data"
VIX_PATH = "data_fred/vixcls.csv"
OUTPUT_DIR = "vix_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "timestamp"
TARGET_COL = "ewma_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"
LAMBDA = 0.94

SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80
DEVICE = "cuda"

# Extended feature list with VIX (ordered by expected importance)
ALL_FEATURES = [
    # Core EWMA lags
    "ewma_vol_lag_1",
    "vix_lag_1",           # VIX early - showed promise!
    "ewma_vol_lag_2", 
    "ewma_vol_lag_3",
    # Abs log return lags
    "abs_log_return_lag_1",
    "abs_log_return_lag_2",
    "abs_log_return_lag_3",
    # Extended EWMA lags
    "ewma_vol_lag_4",
    "ewma_vol_lag_5",
    # Extended abs log return lags
    "abs_log_return_lag_4",
    "abs_log_return_lag_5",
    # More EWMA lags
    "ewma_vol_lag_6",
    "ewma_vol_lag_7",
    "ewma_vol_lag_8",
    "ewma_vol_lag_9",
    "ewma_vol_lag_10",
    # More abs log return lags
    "abs_log_return_lag_6",
    "abs_log_return_lag_7",
    "abs_log_return_lag_8",
]


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


def calculate_ewma_volatility(returns, lambda_=0.94):
    """Calculate EWMA volatility."""
    n = len(returns)
    ewma_var = np.zeros(n)
    returns_filled = returns.fillna(0).values
    ewma_var[0] = returns_filled[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (returns_filled[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


def prepare_data(filepath, vix_df):
    """Load and prepare data with all features including VIX."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # Calculate log returns
    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))
    
    if "abs_log_return" not in df.columns:
        df["abs_log_return"] = np.abs(df["log_return"])
    
    # Calculate EWMA volatility
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    
    # Create all lag features (up to 10 lags)
    for lag in range(1, 11):
        df[f"ewma_vol_lag_{lag}"] = df["ewma_vol"].shift(lag)
        df[f"abs_log_return_lag_{lag}"] = df["abs_log_return"].shift(lag)
    
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


def run_prediction(pipeline, df, feature_cols):
    """Run prediction with given features."""
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
    
    context_cols = [ID_COL, DATE_COL, TARGET_COL] + feature_cols
    future_cols = [ID_COL, DATE_COL] + feature_cols
    
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
        
        return {"mae": mae, "rmse": rmse, "r2": r2}
        
    except Exception as e:
        print(f"Error: {e}")
        return None


def create_plots(results_df, output_dir):
    """Create comparison plots."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    k_values = results_df["K"].values
    
    # MAE Plot
    ax = axes[0]
    ax.plot(k_values, results_df["MAE"], marker='o', linewidth=2, markersize=6, color='steelblue')
    best_k_mae = results_df.loc[results_df["MAE"].idxmin(), "K"]
    ax.axvline(best_k_mae, color='red', linestyle='--', label=f'Best K={int(best_k_mae)}')
    ax.set_xlabel("K (Number of Features)")
    ax.set_ylabel("MAE")
    ax.set_title("MAE vs Number of Features (with VIX)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE Plot
    ax = axes[1]
    ax.plot(k_values, results_df["RMSE"], marker='s', linewidth=2, markersize=6, color='darkorange')
    best_k_rmse = results_df.loc[results_df["RMSE"].idxmin(), "K"]
    ax.axvline(best_k_rmse, color='red', linestyle='--', label=f'Best K={int(best_k_rmse)}')
    ax.set_xlabel("K (Number of Features)")
    ax.set_ylabel("RMSE")
    ax.set_title("RMSE vs Number of Features (with VIX)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # R² Plot
    ax = axes[2]
    ax.plot(k_values, results_df["R2"], marker='^', linewidth=2, markersize=6, color='green')
    best_k_r2 = results_df.loc[results_df["R2"].idxmax(), "K"]
    ax.axvline(best_k_r2, color='red', linestyle='--', label=f'Best K={int(best_k_r2)}')
    ax.set_xlabel("K (Number of Features)")
    ax.set_ylabel("R²")
    ax.set_title("R² vs Number of Features (with VIX)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.suptitle("EWMA + VIX Ablation Study", fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_file = os.path.join(output_dir, "ewma_vix_ablation_metrics.png")
    plt.savefig(plot_file, dpi=150)
    plt.close()
    print(f"Plot saved to {plot_file}")


def main():
    print("=" * 60)
    print("EWMA + VIX Ablation Study (K=1 to 20)")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    print("Model loaded!")
    
    # Load VIX
    print("\nLoading VIX data...")
    vix_df = load_vix()
    print(f"VIX data: {len(vix_df)} rows")
    
    # Load all stock files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"\nFound {len(csv_files)} stock files")
    
    # Prepare all data
    print("\nPreparing data with VIX...")
    all_data = {}
    for filepath in tqdm(csv_files, desc="Loading"):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "").split("_")[0]
        df = prepare_data(filepath, vix_df)
        if len(df) > 100:
            all_data[stock_name] = df
    
    print(f"Loaded {len(all_data)} stocks")
    
    # Test K=1 to K=20
    results = []
    max_k = min(20, len(ALL_FEATURES))
    
    for k in range(1, max_k + 1):
        print(f"\nTesting K={k}...")
        feature_subset = ALL_FEATURES[:k]
        has_vix = "vix_lag_1" in feature_subset
        
        k_results = []
        for stock_name, df in tqdm(all_data.items(), desc=f"K={k}", leave=False):
            result = run_prediction(pipeline, df, feature_subset)
            if result:
                k_results.append(result)
        
        if k_results:
            avg_mae = np.mean([r["mae"] for r in k_results])
            avg_rmse = np.mean([r["rmse"] for r in k_results])
            avg_r2 = np.mean([r["r2"] for r in k_results])
            
            results.append({
                "K": k,
                "Has_VIX": has_vix,
                "Features": ", ".join(feature_subset[:3]) + ("..." if k > 3 else ""),
                "MAE": avg_mae,
                "RMSE": avg_rmse,
                "R2": avg_r2,
                "N_Stocks": len(k_results)
            })
            
            print(f"K={k} (VIX={has_vix}): MAE={avg_mae:.6f}, R²={avg_r2:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, "ewma_vix_full_ablation.csv")
    results_df.to_csv(output_file, index=False)
    print(f"\nResults saved to {output_file}")
    
    # Create plots
    create_plots(results_df, OUTPUT_DIR)
    
    # Print summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df[["K", "Has_VIX", "MAE", "R2"]].to_string(index=False))
    
    best_mae_k = results_df.loc[results_df["MAE"].idxmin(), "K"]
    best_r2_k = results_df.loc[results_df["R2"].idxmax(), "K"]
    
    print(f"\nBest K by MAE: {int(best_mae_k)}")
    print(f"Best K by R²:  {int(best_r2_k)}")
    print(f"\nOptimal Features: {ALL_FEATURES[:int(best_mae_k)]}")


if __name__ == "__main__":
    main()
