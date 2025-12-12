"""
Ablation Study with VIX Feature
================================

Tests the impact of adding VIX as an external feature to:
1. CTC volatility (Top-6 + VIX)
2. EWMA volatility (K=1 + VIX)

VIX is the "Fear Index" and directly measures market volatility expectations.
"""

import os
import glob
import pandas as pd
import numpy as np
from chronos import Chronos2Pipeline
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configuration
DATA_DIR = "processed_data"
VIX_PATH = "data_fred/vixcls.csv"
OUTPUT_DIR = "vix_ablation"
os.makedirs(OUTPUT_DIR, exist_ok=True)

DATE_COL = "timestamp"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"
LAMBDA = 0.94

SPLIT_START_DATE = "2020-01-01"
TRAIN_FRACTION = 0.80
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
    # Normalize VIX to same scale as volatility (divide by 100 to get percentage)
    vix['vix'] = vix['vix'] / 100
    vix['vix_lag_1'] = vix['vix'].shift(1)
    return vix


def prepare_data_with_vix(filepath, vix_df, target_type='ctc'):
    """Load stock data and merge with VIX."""
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
    
    # Calculate target based on type
    if target_type == 'ewma':
        df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
        for lag in [1, 2, 3]:
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


def run_prediction(pipeline, df, target_col, feature_cols):
    """Run prediction with given configuration."""
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
    
    context_cols = [ID_COL, DATE_COL, target_col] + feature_cols
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
            target=target_col
        )
        
        results = pred_df.merge(
            future_df[[ID_COL, DATE_COL, target_col]],
            on=[ID_COL, DATE_COL],
            how="left"
        )
        
        y_true = results[target_col].astype(float).values
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
    print("VIX Feature Ablation Study")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    print("Model loaded!")
    
    # Load VIX
    print("\nLoading VIX data...")
    vix_df = load_vix()
    print(f"VIX data: {len(vix_df)} rows")
    
    # Find stock files
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"\nFound {len(csv_files)} stock files")
    
    # Define test configurations
    configs = [
        # CTC configurations
        {
            "name": "CTC_Top6",
            "target": "ctc_vol",
            "features": ["ctc_vol_lag_1", "ctc_vol_lag_2", "ctc_vol_lag_3",
                        "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3"],
            "type": "ctc"
        },
        {
            "name": "CTC_Top6_VIX",
            "target": "ctc_vol",
            "features": ["ctc_vol_lag_1", "ctc_vol_lag_2", "ctc_vol_lag_3",
                        "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3",
                        "vix_lag_1"],
            "type": "ctc"
        },
        # EWMA configurations
        {
            "name": "EWMA_K1",
            "target": "ewma_vol",
            "features": ["ewma_vol_lag_1"],
            "type": "ewma"
        },
        {
            "name": "EWMA_K1_VIX",
            "target": "ewma_vol",
            "features": ["ewma_vol_lag_1", "vix_lag_1"],
            "type": "ewma"
        },
        {
            "name": "EWMA_K6",
            "target": "ewma_vol",
            "features": ["ewma_vol_lag_1", "ewma_vol_lag_2", "ewma_vol_lag_3",
                        "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3"],
            "type": "ewma"
        },
        {
            "name": "EWMA_K6_VIX",
            "target": "ewma_vol",
            "features": ["ewma_vol_lag_1", "ewma_vol_lag_2", "ewma_vol_lag_3",
                        "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3",
                        "vix_lag_1"],
            "type": "ewma"
        },
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['name']}")
        print(f"Target: {config['target']}")
        print(f"Features: {config['features']}")
        print(f"{'='*60}")
        
        config_results = []
        
        for filepath in tqdm(csv_files, desc=config['name']):
            stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "").split("_")[0]
            
            try:
                df = prepare_data_with_vix(filepath, vix_df, config['type'])
                result = run_prediction(pipeline, df, config['target'], config['features'])
                
                if result:
                    config_results.append(result)
                    
            except Exception as e:
                print(f"Error for {stock_name}: {e}")
                continue
        
        if config_results:
            avg_mae = np.mean([r["mae"] for r in config_results])
            avg_rmse = np.mean([r["rmse"] for r in config_results])
            avg_r2 = np.mean([r["r2"] for r in config_results])
            
            results.append({
                "Config": config["name"],
                "Target": config["target"],
                "N_Features": len(config["features"]),
                "Has_VIX": "vix_lag_1" in config["features"],
                "MAE": avg_mae,
                "RMSE": avg_rmse,
                "R2": avg_r2,
                "N_Stocks": len(config_results)
            })
            
            print(f"\n{config['name']}: MAE={avg_mae:.6f}, R²={avg_r2:.4f}")
    
    # Save results
    results_df = pd.DataFrame(results)
    output_file = os.path.join(OUTPUT_DIR, "vix_ablation_results.csv")
    results_df.to_csv(output_file, index=False)
    
    print("\n" + "=" * 60)
    print("VIX ABLATION RESULTS")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print(f"\nSaved to {output_file}")
    
    # Compare VIX impact
    print("\n" + "=" * 60)
    print("VIX IMPACT ANALYSIS")
    print("=" * 60)
    
    for base in ["CTC_Top6", "EWMA_K1", "EWMA_K6"]:
        base_result = results_df[results_df["Config"] == base].iloc[0]
        vix_result = results_df[results_df["Config"] == f"{base}_VIX"].iloc[0]
        
        mae_change = (vix_result["MAE"] - base_result["MAE"]) / base_result["MAE"] * 100
        r2_change = (vix_result["R2"] - base_result["R2"]) * 100
        
        print(f"\n{base}:")
        print(f"  Without VIX: MAE={base_result['MAE']:.6f}, R²={base_result['R2']:.4f}")
        print(f"  With VIX:    MAE={vix_result['MAE']:.6f}, R²={vix_result['R2']:.4f}")
        print(f"  MAE change:  {mae_change:+.2f}%")
        print(f"  R² change:   {r2_change:+.2f}%")


if __name__ == "__main__":
    main()
