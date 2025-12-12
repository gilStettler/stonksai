
"""
00_fetch_actuals.py

Purpose:
    Fetches fresh market data for the 18 target stocks from Alpha Vantage.
    This data is used as the "Ground Truth" to validate the predictions.
    
    It saves the data to a separate directory (actuals_data) to ensure
    strict separation from the training data.

Usage:
    python 00_fetch_actuals.py
"""

import os
import time
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
from dotenv import load_dotenv

# Load API Key
load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")

# Configuration
# Relative path to store fresh data
OUTPUT_DIR = "actuals_data"

# List of 18 stocks used in the validation
STOCKS = [
    "NSRGY", "UBS", "ABBNY", "0QLR.LON", "0QKI.LON", "0QP2.LON", 
    "0QKY.LON", "0QNO.LON", "0QPS.LON", "0A0D.LON", "0Z4C.LON", 
    "0QOQ.LON", "0QMG.LON", "0QQ2.LON", "0QMW.LON", "0QK6.LON", 
    "RHO6.FRK", "AMRZ"
]

def fetch_data():
    if not API_KEY:
        print("Error: ALPHAVANTAGE_API_KEY not found in .env")
        return

    ts = TimeSeries(key=API_KEY, output_format='pandas')
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Fetching fresh data for {len(STOCKS)} stocks...")
    print(f"Saving to: {OUTPUT_DIR}")
    
    for ticker in STOCKS:
        try:
            print(f"Fetching {ticker}...")
            # Use compact to save API calls and get recent data
            data, meta = ts.get_daily(symbol=ticker, outputsize='compact')
            
            # Calculate Volatility (Close-to-Close, 5-day window)
            # Same logic as in training data generation
            data['close_return'] = data['4. close'].pct_change()
            data['ctc_vol'] = data['close_return'].rolling(window=5).std()
            
            # Save
            # Format filename to match training data convention: data_Name_Ticker.csv
            # We use a simplified naming here for the test
            safe_ticker = ticker.replace(".", "_")
            filename = f"data_{safe_ticker}.csv"
            save_path = os.path.join(OUTPUT_DIR, filename)
            
            # Reset index to make timestamp a column
            data = data.reset_index()
            data.rename(columns={'date': 'timestamp'}, inplace=True)
            
            data.to_csv(save_path, index=False)
            print(f"  Saved {filename}")
            
            # Rate limit (Alpha Vantage free tier is 5 calls/min)
            time.sleep(15) 
            
        except Exception as e:
            print(f"  Error fetching {ticker}: {e}")

if __name__ == "__main__":
    fetch_data()
