"""
Complete CTC Ablation Study with VIX (K=1 to K=20)
===================================================

Tests all CTC feature combinations including VIX.
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
TARGET_COL = "ctc_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"

SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80
DEVICE = "cuda"

# CTC Feature list with VIX
ALL_FEATURES = [
    "ctc_vol_lag_1",
    "vix_lag_1",           # VIX early
    "ctc_vol_lag_2", 
    "ctc_vol_lag_3",
    "abs_log_return_lag_1",
    "abs_log_return_lag_2",
    "abs_log_return_lag_3",
    "ctc_vol_lag_4",
    "ctc_vol_lag_5",
    "abs_log_return_lag_4",
    "abs_log_return_lag_5",
    "ctc_vol_lag_6",
    "ctc_vol_lag_7",
    "ctc_vol_lag_8",
    "ctc_vol_lag_9",
    "ctc_vol_lag_10",
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
    vix['vix'] = vix['vix'] / 100
    vix['vix_lag_1'] = vix['vix'].shift(1)
    return vix


def prepare_data(filepath, vix_df):
    """Load and prepare CTC data with VIX."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # CTC features should already exist in processed_data
    # Create additional lags if needed
    for lag in range(1, 11):
        if f"ctc_vol_lag_{lag}" not in df.columns and "ctc_vol" in df.columns:
            df[f"ctc_vol_lag_{lag}"] = df["ctc_vol"].shift(lag)
        if f"abs_log_return_lag_{lag}" not in df.columns and "abs_log_return" in df.columns:
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
    """Run prediction."""
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


def main():
    print("=" * 60)
    print("CTC + VIX Ablation Study (K=1 to 20)")
    print("=" * 60)
    
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    print("Model loaded!")
    
    vix_df = load_vix()
    print(f"VIX data: {len(vix_df)} rows")
    
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"Found {len(csv_files)} stock files")
    
    # Prepare all data
    all_data = {}
    for filepath in tqdm(csv_files, desc="Loading"):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "").split("_")[0]
        df = prepare_data(filepath, vix_df)
        if len(df) > 100:
            all_data[stock_name] = df
    
    print(f"Loaded {len(all_data)} stocks")
    
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
                "MAE": avg_mae,
                "RMSE": avg_rmse,
                "R2": avg_r2,
                "N_Stocks": len(k_results)
            })
            
            print(f"K={k} (VIX={has_vix}): MAE={avg_mae:.6f}, R²={avg_r2:.4f}")
    
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, "ctc_vix_full_ablation.csv")
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("CTC + VIX RESULTS")
    print("=" * 60)
    print(results_df[["K", "Has_VIX", "MAE", "R2"]].to_string(index=False))
    
    best_mae_k = results_df.loc[results_df["MAE"].idxmin(), "K"]
    print(f"\nBest K by MAE: {int(best_mae_k)}")
    print(f"Optimal Features: {ALL_FEATURES[:int(best_mae_k)]}")


if __name__ == "__main__":
    main()
