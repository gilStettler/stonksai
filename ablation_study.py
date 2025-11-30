import os
import pandas as pd
import numpy as np
import torch
from chronos import BaseChronosPipeline, Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def ablation_study():
    # Configuration
    PREDICTION_LENGTH = 5  # Forecast 5 days ahead (production setting)
    CONTEXT_LENGTH = 60  # Lookback 60 days
    TARGET_COL = "ctc_vol"
    ID_COL = "id"
    DATE_COL = "timestamp"
    
    # Feature Groups for ablation study
    FEATURE_GROUPS = {
        "Baseline (Target Only)": [],
        "Technicals": ["volume", "return", "RSI", "ATR", "SMA", "EMA", "BBANDS"],
        "Macro": ["SP500", "EUROSTOXX_50", "VIX", "CHFUSD", "INFLATION", "FEDERAL_FUNDS_RATE"],
        "Peers": ["Peer_"]  # Will match any column starting with Peer_
    }
    
    # Load Model (Zero-Shot)
    print("Loading Chronos-2 model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline: Chronos2Pipeline = BaseChronosPipeline.from_pretrained(
        "amazon/chronos-2", 
        device_map=device
    )
    
    data_dir = "processed_data"
    stock_files = [f for f in os.listdir(data_dir) if f.startswith("data_") and f.endswith(".csv")]
    
    print(f"Found {len(stock_files)} stocks to test.")
    results = []
    
    for stock_file in stock_files:
        stock_name = stock_file.replace("data_", "").replace(".csv", "")
        print(f"\nProcessing {stock_name}...")
        
        file_path = os.path.join(data_dir, stock_file)
        try:
            df = pd.read_csv(file_path, parse_dates=['timestamp'])
        except Exception as e:
            print(f"  Error loading {stock_file}: {e}")
            continue
        
        # Convert timestamp to datetime if not already
        if 'timestamp' not in df.columns:
            # Assume first column is timestamp if index
            df = df.reset_index()
            df.rename(columns={df.columns[0]: 'timestamp'}, inplace=True)
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.sort_values('timestamp', inplace=True)
        
        # CRITICAL: Reindex to Business Days to ensure regular frequency
        # This is how the chronos-2-test notebook handles it
        all_bdays = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='B')
        df = df.set_index('timestamp').reindex(all_bdays)
        df.index.name = 'timestamp'
        df = df.reset_index()
        
        # Forward fill missing values (from weekends/holidays)
        # Only for numeric columns to avoid issues
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill()
            
        # Ensure we have enough data
        if len(df) < CONTEXT_LENGTH + PREDICTION_LENGTH:
            print(f"  Not enough data for {stock_name}, skipping.")
            continue
        
        # Check if target column exists
        if TARGET_COL not in df.columns:
            print(f"  Target column {TARGET_COL} not found, skipping.")
            continue
            
        # Split point
        split_idx = len(df) - PREDICTION_LENGTH
        
        # Context Data (History) and Future Data
        context_df = df.iloc[:split_idx].copy()
        future_df = df.iloc[split_idx:].copy()
        
        # Ground truth
        y_true = future_df[TARGET_COL].values
        
        # Test each feature group
        for group_name, features in FEATURE_GROUPS.items():
            print(f"  Testing Feature Group: {group_name}")
            
            # Identify available features for this group
            if group_name == "Peers":
                available_features = [c for c in df.columns if c.startswith("Peer_")]
            else:
                available_features = [f for f in features if f in df.columns]
            
            # Build column sets (similar to notebook approach)
            context_cols = [TARGET_COL] + available_features
            future_cols = available_features  # future_df does NOT include target
            
            # Prepare DataFrames for Chronos
            # Add ID and reset index to make timestamp a column
            context_df_chronos = context_df[context_cols].copy()
            context_df_chronos[ID_COL] = stock_name
            context_df_chronos = context_df_chronos.reset_index()
            context_df_chronos.rename(columns={context_df_chronos.columns[0]: DATE_COL}, inplace=True)
            
            # Reorder columns: [id, timestamp, target, ...features]
            context_df_chronos = context_df_chronos[[ID_COL, DATE_COL, TARGET_COL] + available_features]
            
            try:
                # Predict
                if not available_features:
                    # Baseline: univariate prediction without future_df
                    pred_df = pipeline.predict_df(
                        context_df_chronos,
                        prediction_length=PREDICTION_LENGTH,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column=ID_COL,
                        timestamp_column=DATE_COL,
                        target=TARGET_COL
                    )
                else:
                    # With covariates: use future_df
                    future_df_chronos = future_df[future_cols].copy()
                    future_df_chronos[ID_COL] = stock_name
                    future_df_chronos = future_df_chronos.reset_index()
                    future_df_chronos.rename(columns={future_df_chronos.columns[0]: DATE_COL}, inplace=True)
                    
                    # Reorder columns: [id, timestamp, ...features]
                    future_df_chronos = future_df_chronos[[ID_COL, DATE_COL] + available_features]
                    
                    pred_df = pipeline.predict_df(
                        context_df_chronos,
                        future_df=future_df_chronos,
                        prediction_length=PREDICTION_LENGTH,
                        quantile_levels=[0.1, 0.5, 0.9],
                        id_column=ID_COL,
                        timestamp_column=DATE_COL,
                        target=TARGET_COL
                    )
                
                # Extract predictions
                # pred_df contains: item_id, timestamp, target_name, predictions (mean), 0.1, 0.5, 0.9 quantiles
                y_pred_median = pred_df["0.5"].values  # Median forecast
                y_pred_q10 = pred_df["0.1"].values     # 10th percentile
                y_pred_q90 = pred_df["0.9"].values     # 90th percentile
                
                # --- Deterministic Metrics (Point Forecast) ---
                mae = mean_absolute_error(y_true, y_pred_median)
                mse = mean_squared_error(y_true, y_pred_median)
                rmse = np.sqrt(mse)
                
                # --- Adjusted R² ---
                # Measures proportion of variance explained, adjusted for number of features
                from sklearn.metrics import r2_score
                r2 = r2_score(y_true, y_pred_median)
                n = len(y_true)
                p = len(available_features)  # Number of predictors
                if n - p - 1 > 0:
                    adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
                else:
                    adj_r2 = np.nan
                
                # --- Quantile Losses ---
                # Pinball loss for each quantile
                quantile_losses = {}
                quantiles = [0.1, 0.5, 0.9]
                quantile_preds = {"0.1": y_pred_q10, "0.5": y_pred_median, "0.9": y_pred_q90}
                
                for q in quantiles:
                    y_pred_q = quantile_preds[f"{q}"]
                    diff = y_true - y_pred_q
                    # Pinball loss: max(q * diff, (q-1) * diff)
                    loss = np.mean(np.maximum(q * diff, (q - 1) * diff))
                    quantile_losses[f"QL_{q}"] = loss
                
                # --- CRPS (Continuous Ranked Probability Score) ---
                # Approximation using quantile losses via trapezoidal rule
                # CRPS = 2 * integral of QuantileLoss(q) from 0 to 1
                # We approximate with 3 quantiles: 0.1, 0.5, 0.9
                qs = np.array(quantiles)
                Ls = np.array([quantile_losses[f"QL_{q}"] for q in quantiles])
                crps = 2 * np.trapz(Ls, qs)
                
                results.append({
                    "Stock": stock_name,
                    "Feature Group": group_name,
                    "MAE": mae,
                    "MSE": mse,
                    "RMSE": rmse,
                    "Adjusted_R2": adj_r2,
                    "CRPS": crps,
                    "QL_0.1": quantile_losses["QL_0.1"],
                    "QL_0.5": quantile_losses["QL_0.5"],
                    "QL_0.9": quantile_losses["QL_0.9"],
                    "Num Features": len(available_features)
                })
                print(f"    MAE: {mae:.4f}, RMSE: {rmse:.4f}, CRPS: {crps:.4f}, Adj.R²: {adj_r2:.3f}")
                
            except Exception as e:
                print(f"    Prediction failed: {e}")
                results.append({
                    "Stock": stock_name,
                    "Feature Group": group_name,
                    "MAE": np.nan,
                    "MSE": np.nan,
                    "RMSE": np.nan,
                    "Num Features": len(available_features),
                    "Error": str(e)
                })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False)
    print("\n\nAblation Study Completed. Results saved to ablation_results.csv")
    
    # Display summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS BY FEATURE GROUP")
    print("="*80)
    
    summary = results_df.groupby("Feature Group").agg({
        "MAE": ["mean", "std", "min", "max"],
        "RMSE": ["mean", "std", "min", "max"],
        "CRPS": ["mean", "std", "min", "max"],
        "Adjusted_R2": ["mean", "std"],
        "QL_0.1": ["mean"],
        "QL_0.5": ["mean"],
        "QL_0.9": ["mean"],
        "Num Features": "first"
    }).round(6)
    
    print(summary)
    
    # Calculate improvement vs baseline
    print("\n" + "="*80)
    print("IMPROVEMENT vs BASELINE (Lower is better for CRPS, QL; Higher for R²)")
    print("="*80)
    
    baseline_df = results_df[results_df["Feature Group"] == "Baseline (Target Only)"]
    
    if not baseline_df.empty:
        for group_name in FEATURE_GROUPS.keys():
            if group_name == "Baseline (Target Only)":
                continue
            
            group_df = results_df[results_df["Feature Group"] == group_name]
            
            # Match by stock
            merged = baseline_df.merge(group_df, on="Stock", suffixes=("_baseline", "_group"))
            
            if not merged.empty:
                # Calculate improvements (positive = better for MAE/RMSE/CRPS, negative for R²)
                mae_improvement = ((merged["MAE_baseline"] - merged["MAE_group"]) / merged["MAE_baseline"] * 100).mean()
                rmse_improvement = ((merged["RMSE_baseline"] - merged["RMSE_group"]) / merged["RMSE_baseline"] * 100).mean()
                crps_improvement = ((merged["CRPS_baseline"] - merged["CRPS_group"]) / merged["CRPS_baseline"] * 100).mean()
                
                # For R², higher is better, so we compare differently
                r2_baseline = merged["Adjusted_R2_baseline"].mean()
                r2_group = merged["Adjusted_R2_group"].mean()
                r2_change = r2_group - r2_baseline
                
                print(f"\n{group_name}:")
                print(f"  MAE Improvement:  {mae_improvement:+.2f}%")
                print(f"  RMSE Improvement: {rmse_improvement:+.2f}%")
                print(f"  CRPS Improvement: {crps_improvement:+.2f}% ⭐ (Most Important)")
                print(f"  Adj. R² Change:   {r2_change:+.4f}")
    
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    ablation_study()
