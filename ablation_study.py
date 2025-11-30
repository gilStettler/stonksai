import os
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def ablation_study():
    # Configuration
    PREDICTION_LENGTH = 20 # Forecast 20 days
    CONTEXT_LENGTH = 60 # Lookback 60 days
    
    # Feature Groups
    FEATURE_GROUPS = {
        "Baseline (Target Only)": [],
        "Technicals": ["volume", "return", "RSI", "ATR", "SMA", "EMA", "BBANDS"], # Add specific columns if they exist
        "Macro": ["SP500", "EUROSTOXX_50", "VIX", "CHFUSD", "INFLATION", "FEDERAL_FUNDS_RATE"],
        "Peers": ["Peer_"] # Will match any column starting with Peer_
    }
    
    # Load Model (Zero-Shot)
    print("Loading Chronos-2 model...")
    # Using tiny for speed in testing, user can swap to small/base
    # Note: Chronos 2 models are usually named like 'amazon/chronos-t5-tiny' but accessed via Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-t5-tiny", 
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
import os
import pandas as pd
import numpy as np
import torch
from chronos import Chronos2Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error

def ablation_study():
    # Configuration
    PREDICTION_LENGTH = 20 # Forecast 20 days
    CONTEXT_LENGTH = 60 # Lookback 60 days
    TARGET_COL = "ctc_vol"
    
    # Feature Groups
    FEATURE_GROUPS = {
        "Baseline (Target Only)": [],
        "Technicals": ["volume", "return", "RSI", "ATR", "SMA", "EMA", "BBANDS"], # Add specific columns if they exist
        "Macro": ["SP500", "EUROSTOXX_50", "VIX", "CHFUSD", "INFLATION", "FEDERAL_FUNDS_RATE"],
        "Peers": ["Peer_"] # Will match any column starting with Peer_
    }
    
    # Load Model (Zero-Shot)
    print("Loading Chronos-2 model...")
    # Using tiny for speed in testing, user can swap to small/base
    # Note: Chronos 2 models are usually named like 'amazon/chronos-t5-tiny' but accessed via Chronos2Pipeline
    pipeline = Chronos2Pipeline.from_pretrained(
        "amazon/chronos-t5-tiny", 
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    
    data_dir = "processed_data"
    # Find all stock files
    stock_files = [f for f in os.listdir(data_dir) if f.startswith("data_") and f.endswith(".csv")]
    
    print(f"Found {len(stock_files)} stocks to test.")
    # Prediction Loop
    results = []
    
    for stock_file in stock_files:
        stock_name = stock_file.replace("data_", "").replace(".csv", "")
        print(f"\nProcessing {stock_name}...")
        
        file_path = os.path.join(data_dir, stock_file)
        try:
            df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        except Exception as e:
            print(f"  Error loading {stock_file}: {e}")
            continue
            
        # Ensure we have enough data
        if len(df) < CONTEXT_LENGTH + PREDICTION_LENGTH:
            print(f"  Not enough data for {stock_name}, skipping.")
            continue
            
        # Prepare Data
        # We'll take the last available window for testing
        # Context: [-CONTEXT_LENGTH-PREDICTION_LENGTH : -PREDICTION_LENGTH]
        # Future (Ground Truth): [-PREDICTION_LENGTH : ]
        
        # Split point
        split_idx = len(df) - PREDICTION_LENGTH
        
        # Context Data (History)
        context_df_full = df.iloc[:split_idx].copy()
        # We only need the last CONTEXT_LENGTH rows for the context input to Chronos
        # BUT for covariates, we might want to provide the full history if Chronos uses it.
        # Chronos 2 usually takes a dataframe. Let's provide the relevant context window.
        # To be safe and consistent with typical usage, let's provide a generous context if possible, 
        # or just the fixed CONTEXT_LENGTH. The prompt implies fixed context length.
        context_df = context_df_full.iloc[-CONTEXT_LENGTH:].copy()
        
        # Future Data (Ground Truth & Future Covariates)
        future_df_full = df.iloc[split_idx:].copy()
        
        # Target Series (Ground Truth)
        target_series = future_df_full[TARGET_COL]
        
        for group_name, features in FEATURE_GROUPS.items():
            print(f"  Testing Feature Group: {group_name}")
            
            # Identify available features for this group
            available_features = []
            if group_name == "Peers":
                available_features = [c for c in df.columns if c.startswith("Peer_")]
            else:
                available_features = [f for f in features if f in df.columns]
            
            # Prepare Input DataFrames for Chronos
            # Chronos 2 predict_df expects:
            # 1. df: Context dataframe (history)
            # 2. future_df: Dataframe with future covariates (optional)
            
            # Construct Context DF
            # Must contain: timestamp, target, and past covariates
            # We need to reshape to long format or just ensure columns are present if using wide format support?
            # Chronos 2 predict_df usually expects a long format or a specific structure. 
            # However, the quickstart shows it handling standard wide DFs if we specify target and id.
            # Let's add a dummy ID if not present.
            
            current_context = context_df.copy()
            current_context["id"] = stock_name
            current_context = current_context.reset_index() # Ensure timestamp is a column
            # Rename index to 'timestamp' if it's not named
            if "timestamp" not in current_context.columns:
                 current_context.rename(columns={"index": "timestamp", "Date": "timestamp"}, inplace=True)
            
            # Construct Future DF (for covariates)
            # Must contain: timestamp, and future values of covariates
            current_future = future_df_full.copy()
            current_future["id"] = stock_name
            current_future = current_future.reset_index()
            if "timestamp" not in current_future.columns:
                 current_future.rename(columns={"index": "timestamp", "Date": "timestamp"}, inplace=True)
            
            # Drop target from future_df to avoid leakage (though Chronos ignores it if passed as future_df, it's safer)
            if TARGET_COL in current_future.columns:
                current_future = current_future.drop(columns=[TARGET_COL])
            
            # Select only relevant columns + mandatory ones
            cols_to_keep = ["id", "timestamp"] + ([TARGET_COL] if TARGET_COL in current_context.columns else []) + available_features
            
            # Filter Context
            # Note: context must have target
            current_context_filtered = current_context[cols_to_keep].copy()
            
            # Filter Future
            # Future df should NOT have target, but MUST have covariates
            cols_to_keep_future = ["id", "timestamp"] + available_features
            current_future_filtered = current_future[cols_to_keep_future].copy()
            
            # Predict
            try:
                # If no features, we don't pass future_df (univariate forecast)
                if not available_features:
                    forecast_df = pipeline.predict_df(
                        current_context_filtered,
                        prediction_length=PREDICTION_LENGTH,
                        id_column="id",
                        timestamp_column="timestamp",
                        target=TARGET_COL
                    )
                else:
                    forecast_df = pipeline.predict_df(
                        current_context_filtered,
                        future_df=current_future_filtered,
                        prediction_length=PREDICTION_LENGTH,
                        id_column="id",
                        timestamp_column="timestamp",
                        target=TARGET_COL
                    )
                
                # Extract Predictions
                # forecast_df contains 'predictions' (mean) and quantiles. We use 'predictions' (mean) for metrics.
                # We need to align it with target_series.
                # forecast_df should be sorted by timestamp.
                
                predictions = forecast_df["predictions"].values
                
                # Calculate Metrics
                mae = mean_absolute_error(target_series, predictions)
                mse = mean_squared_error(target_series, predictions)
                
                results.append({
                    "Stock": stock_name,
                    "Feature Group": group_name,
                    "MAE": mae,
                    "MSE": mse
                })
                
            except Exception as e:
                print(f"    Prediction failed: {e}")
                # Fallback or record error
                results.append({
                    "Stock": stock_name,
                    "Feature Group": group_name,
                    "MAE": np.nan,
                    "MSE": np.nan
                })

    # Save Results
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False)
    print("\nAblation Study Completed. Results saved to ablation_results.csv")
    print(results_df)

if __name__ == "__main__":
    ablation_study()
