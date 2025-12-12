import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_correlations():
    data_dir = "processed_data"
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv") and "data_" in f]
    
    # Filter for the new named files if possible, or just use all unique tickers
    # Let's use the ones with Company Names if available, otherwise fallback
    # Actually, let's just grab 'close' price from each file
    
    price_data = {}
    
    print("Loading data...")
    
    # 1. Identify valid files and map to clean names
    # Priority: data_Company_Ticker.csv > data_Ticker.csv
    # We want to avoid duplicates (e.g. data_UBS.csv AND data_UBS_UBS.csv)
    
    unique_stocks = {} # Map ticker -> filepath
    
    for f in files:
        # Check if it matches new pattern: data_Name_Ticker.csv
        # We can try to split by underscore.
        parts = f.replace(".csv", "").split("_")
        
        if len(parts) >= 3: # data, Name, Ticker (e.g. data, Swisscom, 0QKI.LON)
            ticker = parts[-1]
            name = parts[1] # Company Name
            unique_stocks[ticker] = {"path": f, "name": name}
        elif len(parts) == 2: # data, Ticker (old format)
            ticker = parts[1]
            # Only add if we don't have a better version yet
            if ticker not in unique_stocks:
                unique_stocks[ticker] = {"path": f, "name": ticker} # Fallback name is ticker

    print(f"Found {len(unique_stocks)} unique stocks.")

    for ticker, info in unique_stocks.items():
        f = info["path"]
        name = info["name"]
        
        try:
            df = pd.read_csv(os.path.join(data_dir, f))
            if "timestamp" in df.columns and "close" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
                df = df.set_index("timestamp").sort_index()
                # Resample to ensure alignment
                price_data[name] = df["close"]
        except Exception as e:
            print(f"Skipping {f}: {e}")
            
    if not price_data:
        print("No data found.")
        return

    # Filter out Amrize explicitly
    if "Amrize" in price_data:
        del price_data["Amrize"]

    # Create DataFrame
    df_prices = pd.DataFrame(price_data)
    
    # Function to run analysis for a specific timeframe
    def run_analysis(df, label, filename_suffix):
        print(f"\n{'='*40}")
        print(f"Analysis: {label}")
        print(f"Timeframe: {df.index.min().date()} to {df.index.max().date()}")
        print(f"{'='*40}")
        
        # Calculate Correlation on Daily Returns
        df_returns = df.pct_change()
        corr_matrix = df_returns.corr()
        
        print(f"\nTop Correlations ({label}):")
        
        # Find top 3 correlated stocks for each stock
        for stock in corr_matrix.columns:
            print(f"\n{stock}:")
            top_corr = corr_matrix[stock].sort_values(ascending=False)
            top_corr = top_corr[top_corr < 0.999] # Remove self
            
            for other_stock, score in top_corr.head(3).items():
                print(f"  - {other_stock}: {score:.2f}")

        # Save Heatmap
        plt.figure(figsize=(14, 12))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1)
        plt.title(f"Stock Returns Correlation Matrix ({label})")
        plt.tight_layout()
        output_file = f"correlation_heatmap_{filename_suffix}.png"
        plt.savefig(output_file)
        print(f"\nSaved heatmap to {output_file}")

    # 1. All Time Analysis
    run_analysis(df_prices, "All Available Data", "all")

    # 2. Since 2020 Analysis
    df_2020 = df_prices[df_prices.index >= "2020-01-01"]
    if not df_2020.empty:
        run_analysis(df_2020, "Since 2020", "2020")
    else:
        print("\nNo data found since 2020.")

if __name__ == "__main__":
    analyze_correlations()
