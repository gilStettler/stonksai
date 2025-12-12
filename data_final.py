"""
Simple script to:
- Download daily OHLCV data for a list of stock tickers from Alpha Vantage
- Compute daily returns (today's close vs yesterday's close) for each stock
- Download daily VIX data from the FRED API
- Merge OHLCV, Return and VIX per day
- Save one CSV file per stock

Requirements:
- pip install requests pandas
- Get free API keys from:
  * Alpha Vantage: https://www.alphavantage.co
  * FRED: https://fred.stlouisfed.org/
"""

import requests
import pandas as pd
import os

# --------------------------------------------------------------------
# CONFIGURATION
# --------------------------------------------------------------------

# Replace these with your real API keys
ALPHAVANTAGE_API_KEY = "BFMDNSKEEDKULH7S"
FRED_API_KEY = "6e4c3c9e3f698a2828f2dd1f9079bcff"

# List of stock tickers you want to download
TICKERS = ["0QKI.LON", "0QLR.LON", "NSRGY", "RHO6.FRK", "ABBNY", "UBS", "0QP2.LON", "0QKY.LON", "0QNO.LON", "0QPS.LON", "0A0D.LON", "0Z4C.LON", "0QOQ.LON", "0QMG.LON", "0QQ2.LON", "0QMW.LON", "0QK6.LON"]  # You can change this list

# Alpha Vantage endpoint for daily adjusted prices
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

# FRED endpoint for VIX (series_id = VIXCLS)
FRED_URL = "https://api.stlouisfed.org/fred/series/observations"


# --------------------------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------------------------

def get_daily_ohlcv_from_alpha_vantage(symbol: str) -> pd.DataFrame:
    """
    Download daily OHLCV data for one symbol from Alpha Vantage.

    Returns a DataFrame with:
    - index: Date (datetime)
    - columns: Open, High, Low, Close, Volume
    """
    # Prepare the query parameters for Alpha Vantage
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",  # daily prices
        "symbol": symbol,
        "outputsize": "full",                      # full history
        "apikey": ALPHAVANTAGE_API_KEY
    }

    # Send HTTP GET request
    response = requests.get(ALPHAVANTAGE_URL, params=params)
    data = response.json()

    # The time series is stored under this key
    time_series_key = "Time Series (Daily)"

    if time_series_key not in data:
        raise ValueError(f"Alpha Vantage response for {symbol} does not contain '{time_series_key}'.")

    # Create a DataFrame from the nested dictionary
    # data[time_series_key] is a dict: { "YYYY-MM-DD": { "1. open": "...", ... } }
    df = pd.DataFrame.from_dict(data[time_series_key], orient="index")

    # Rename columns to simpler names
    df = df.rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume"
    })

    # Convert index to datetime and sort by date (oldest first)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    # Keep only the columns we need (OHLCV)
    df = df[["Open", "High", "Low", "Close", "Volume"]]

    # Convert string values to float / integer
    df["Open"] = df["Open"].astype(float)
    df["High"] = df["High"].astype(float)
    df["Low"] = df["Low"].astype(float)
    df["Close"] = df["Close"].astype(float)
    df["Volume"] = df["Volume"].astype(float)

    return df



import yfinance as yf

def get_vix_from_fred() -> pd.DataFrame:
    """
    Download daily VIX data from Yahoo Finance (^VIX).
    Renamed keeping function signature compatible, but using yfinance internally.
    """
    print("Fetching VIX from Yahoo Finance...")
    vix = yf.Ticker("^VIX")
    # Get max history to be safe
    hist = vix.history(period="max")
    
    # yfinance returns index as Datetime with timezone. We need timezone-naive.
    hist.index = hist.index.tz_localize(None)
    
    vix_df = hist[["Close"]].rename(columns={"Close": "VIX"})
    vix_df.index.name = "date"
    vix_df = vix_df.sort_index()
    
    # Create 'value' column as alias if needed by downstream, or just VIX
    # Returns DataFrame with index 'date' and column 'VIX'
    return vix_df


# --------------------------------------------------------------------
# MAIN LOGIC
# --------------------------------------------------------------------

def main():
    # ----------------------------------------------------------------
    # 1) Download VIX data once (it is the same for all stocks)
    # ----------------------------------------------------------------
    print("Downloading VIX data from FRED...")
    vix_df = get_vix_from_fred()

    # Create output directory
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(SCRIPT_DIR, "data")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Saving data to: {OUTPUT_DIR}")

    # ----------------------------------------------------------------
    # 2) Loop over all tickers, download OHLCV, compute returns,
    #    merge with VIX and save to CSV
    # ----------------------------------------------------------------
    for symbol in TICKERS:
        print(f"Processing {symbol}...")

        # --------------------------------------------
        # 2a) Download OHLCV data for this stock
        # --------------------------------------------
        stock_df = get_daily_ohlcv_from_alpha_vantage(symbol)

        # --------------------------------------------
        # 2b) Compute daily return (Close vs previous Close)
        #     Return_t = Close_t / Close_{t-1} - 1
        # --------------------------------------------
        stock_df["Return"] = stock_df["Close"].pct_change()

        # --------------------------------------------
        # 2c) Merge stock data with VIX data by date
        #     We join on the index (date). Left join means:
        #     - keep all stock dates
        #     - add VIX where available
        # --------------------------------------------
        merged_df = stock_df.join(vix_df, how="left")

        # --------------------------------------------
        # 2d) Save to CSV with one file per symbol
        #     Columns: OHLCV, Return, VIX per day
        # --------------------------------------------
        # Sanitize filename (remove special chars if any)
        safe_symbol = symbol.replace("/", "_").replace("\\", "_")
        output_filename = os.path.join(OUTPUT_DIR, f"data_{safe_symbol}.csv")
        merged_df.to_csv(output_filename, index_label="timestamp") # using 'timestamp' as header for date match

        print(f"Saved {output_filename}")

        print(f"Saved {output_filename}")

    print("Done.")


# Only run main() if this file is executed directly
if __name__ == "__main__":
    main()
