import os
import pandas as pd
import numpy as np

def inject_peer_features():
    data_dir = "processed_data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "data_" in f and "_daily_data" not in f]
    
    # Filter for the "Company_Ticker" format we established
    # e.g. data_Swisscom_0QKI.LON.csv
    # We want to ignore the old "data_Ticker.csv" if a better one exists, 
    # but for simplicity, let's just process all "data_*.csv" that look like our target files.
    # Actually, let's rely on the same logic as analyze_correlations to get the best file per ticker.
    
    print("Loading data for correlation analysis...")
    
    unique_stocks = {} # ticker -> {path, name, df}
    
    for f in files:
        parts = f.replace(".csv", "").split("_")
        if len(parts) >= 3: # data, Name, Ticker
            ticker = parts[-1]
            name = parts[1]
            unique_stocks[ticker] = {"path": f, "name": name}
    
    # Load all DFs
    for ticker, info in unique_stocks.items():
        path = os.path.join(data_dir, info["path"])
        try:
            df = pd.read_csv(path)
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
                info["df"] = df
            else:
                print(f"Warning: No timestamp in {path}")
        except Exception as e:
            print(f"Error loading {path}: {e}")

    # Calculate Correlations (Post-2020)
    price_data = {t: info["df"]["close"] for t, info in unique_stocks.items() if "df" in info}
    
    if "AMRZ" in price_data: del price_data["AMRZ"] # Exclude Amrize
    
    df_prices = pd.DataFrame(price_data)
    df_prices_2020 = df_prices[df_prices.index >= "2020-01-01"]
    
    if df_prices_2020.empty:
        print("Not enough data since 2020 for correlation.")
        return

    corr_matrix = df_prices_2020.pct_change().corr()
    
    print("\nInjecting Features...")
    
    for target_ticker, target_info in unique_stocks.items():
        if "df" not in target_info: continue
        if target_ticker not in corr_matrix.columns: continue
        
        print(f"Processing {target_info['name']} ({target_ticker})...")
        
        # Find Top 2 Peers (Absolute Correlation)
        # Get correlations for this stock
        stock_corrs = corr_matrix[target_ticker].drop(target_ticker) # Remove self
        
        # Sort by absolute value
        top_peers = stock_corrs.abs().sort_values(ascending=False).head(2)
        
        target_df = target_info["df"]
        
        # Clean up existing Peer columns to avoid merge conflicts (suffixes)
        cols_to_drop = [c for c in target_df.columns if c.startswith("Peer_")]
        if cols_to_drop:
            print(f"  Dropping {len(cols_to_drop)} existing peer columns.")
            target_df = target_df.drop(columns=cols_to_drop)
        
        for peer_ticker, abs_score in top_peers.items():
            real_score = stock_corrs[peer_ticker]
            peer_name = unique_stocks[peer_ticker]["name"]
            print(f"  -> Adding Peer: {peer_name} (Corr: {real_score:.2f})")
            
            # Get Peer Data
            peer_df = unique_stocks[peer_ticker]["df"]
            
            # Align Data (Merge)
            # We want to add peer_close and peer_return
            # We use left join to keep target's index
            
            # Create temp DF for peer features
            # We want to add peer_close, peer_return, AND the new volatility features
            features_to_inject = [
                "close", "return",
                "ctc_vol_lag_1", "ctc_vol_lag_2", "ctc_vol_lag_3",
                "abs_log_return_lag_1", "abs_log_return_lag_2", "abs_log_return_lag_3"
            ]
            
            # Ensure these columns exist in peer_df
            valid_features = [f for f in features_to_inject if f in peer_df.columns]
            
            peer_features = peer_df[valid_features].copy()
            peer_features.columns = [f"Peer_{peer_name}_{col}" for col in valid_features]
            
            # Merge
            target_df = target_df.merge(peer_features, left_index=True, right_index=True, how="left")
            
            # Forward fill missing peer data (if peer has gaps but target trades)
            # Limit ffill to avoid stale data? Maybe 5 days.
            for col in peer_features.columns:
                target_df[col] = target_df[col].ffill(limit=5)
                # Fill remaining NaNs with 0 only for return-like features, not prices/vol?
                # Actually, 0 for vol is risky, but better than NaN for training.
                # Let's fill 0 for now as per previous logic for returns.
                target_df[col] = target_df[col].fillna(0)
            
        # Save updated file
        # We overwrite the file or create a new one? 
        # Let's overwrite to make it the "standard" dataset for training.
        output_path = os.path.join(data_dir, target_info["path"])
        target_df.to_csv(output_path)
        print(f"  Saved updated {target_info['path']}")

if __name__ == "__main__":
    inject_peer_features()
