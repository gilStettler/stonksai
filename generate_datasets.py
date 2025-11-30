import os
import io
import time
import requests
import pandas as pd
import numpy as np
import yfinance as yf
from fredapi import Fred
from dotenv import load_dotenv
import quantreo.features_engineering as fe

# Load environment variables
load_dotenv(override=True)

# Configuration
ALPHAVANTAGE_API_KEY = os.getenv('ALPHAVANTAGE_API_KEY')
FRED_API_KEY = os.getenv('FRED_API_KEY')
BASE_URL = "https://www.alphavantage.co/query?"

if not FRED_API_KEY:
    print("Warning: FRED_API_KEY not found in environment variables.")

def get_alpha_vantage_data(function, symbol=None, interval=None, outputsize="full"):
    """Helper to fetch data from Alpha Vantage."""
    url = f"{BASE_URL}function={function}&apikey={ALPHAVANTAGE_API_KEY}&datatype=csv"
    if symbol:
        url += f"&symbol={symbol}"
    if interval:
        url += f"&interval={interval}"
    if outputsize:
        url += f"&outputsize={outputsize}"
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        # Check if response is valid CSV or error message
        if "Error Message" in response.text or "Information" in response.text:
            print(f"API Error for {function} {symbol}: {response.text[:100]}")
            return None
        return pd.read_csv(io.StringIO(response.text))
    except Exception as e:
        print(f"Failed to fetch {function} for {symbol}: {e}")
        return None

def get_macro_data():
    """Fetches and processes all macroeconomic data."""
    print("Fetching Macro Data...")
    
    # 1. Alpha Vantage Macro Data
    macro_functions = {
        "FEDERAL_FUNDS_RATE": "daily",
        "INFLATION": None, # annual
        "REAL_GDP": "quarterly",
        "REAL_GDP_PER_CAPITA": "quarterly",
        "TREASURY_YIELD": "daily",
        "CPI": "monthly"
    }
    
    macro_dfs = []
    for func, interval in macro_functions.items():
        print(f"  Fetching {func}...")
        df = get_alpha_vantage_data(func, interval=interval)
        if df is not None:
            df = df.rename(columns={"timestamp": "timestamp", "time": "timestamp", "value": func})
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.set_index("timestamp")
            macro_dfs.append(df)
            time.sleep(0.5) # Rate limit safety

    # 2. Yahoo Finance Indices
    print("  Fetching Market Indices (Yahoo)...")
    indices = ["^FTSE", "^STOXX50E", "^GSPC"]
    df_indices = yf.download(tickers=indices, period="max", interval="1d", auto_adjust=True, progress=False)["Close"]
    df_indices.rename(columns={"^FTSE": "FTSE_100", "^STOXX50E": "EUROSTOXX_50", "^GSPC": "SP500"}, inplace=True)
    df_indices.index = pd.to_datetime(df_indices.index)
    df_indices.index.name = "timestamp"
    macro_dfs.append(df_indices)

    # 3. FRED Data
    if FRED_API_KEY:
        print("  Fetching FRED Data...")
        fred = Fred(api_key=FRED_API_KEY)
        series_map = {
            "VIX": "VIXCLS", "CHFUSD": "DEXSZUS", "EURUSD": "DEXUSEU",
            "US1Y": "DGS1", "US2Y": "DGS2", "US5Y": "DGS5", "US10Y": "DGS10", "US30Y": "DGS30",
            "YC_Slope": "T10Y2Y", "Stress Index": "STLFSI4", "FEDFUNDS": "FEDFUNDS",
            "UNRATE": "UNRATE", "Median CPI": "MEDCPIM158SFRBCLE", "UMCSENT": "UMCSENT",
            "USEPUINDXD": "USEPUINDXD"
        }
        fred_dfs = []
        for name, sid in series_map.items():
            try:
                s = fred.get_series(sid)
                s.index = pd.to_datetime(s.index)
                s.name = name
                fred_dfs.append(s)
                time.sleep(0.2)
            except Exception as e:
                print(f"    Failed to fetch {name} from FRED: {e}")
        
        if fred_dfs:
            df_fred = pd.concat(fred_dfs, axis=1)
            df_fred.index.name = "timestamp"
            macro_dfs.append(df_fred)

    # Merge all macro data
    print("  Merging Macro Data...")
    if not macro_dfs:
        return pd.DataFrame()
        
    df_macro = macro_dfs[0]
    for df in macro_dfs[1:]:
        df_macro = df_macro.merge(df, on="timestamp", how="outer")
    
    # Handle Publication Lags
    pub_lags = {
        "FEDERAL_FUNDS_RATE": 1, "INFLATION": 14, "REAL_GDP": 30, "REAL_GDP_PER_CAPITA": 30,
        "TREASURY_YIELD": 0, "CPI": 14, "FTSE_100": 0, "EUROSTOXX_50": 0, "SP500": 0,
        "VIX": 0, "CHFUSD": 1, "EURUSD": 1, "US1Y": 1, "US2Y": 1, "US5Y": 1,
        "US10Y": 1, "US30Y": 1, "YC_Slope": 0, "Stress Index": 6, "FEDFUNDS": 1,
        "UNRATE": 7, "Median CPI": 14, "UMCSENT": 15, "USEPUINDXD": 30
    }
    
    df_macro = df_macro.sort_index()
    for col in df_macro.columns:
        if col in pub_lags:
            df_macro[col] = df_macro[col].shift(pub_lags[col])
            
    # Resample to daily and forward fill
    full_index = pd.date_range(start=df_macro.index.min(), end=df_macro.index.max(), freq="D")
    df_macro = df_macro.reindex(full_index)
    df_macro.index.name = "timestamp"
    df_macro = df_macro.ffill()
    
    return df_macro

    # 2. Find all stocks in alphavantage_data
    data_dir = "alphavantage_data"
    files = [f for f in os.listdir(data_dir) if f.endswith("_daily_data.csv")]
    
    print(f"Found {len(files)} stock files to process.")
    
    # Ticker to Company Name Mapping
    TICKER_MAP = {
        "0QKI.LON": "Swisscom",
        "0QLR.LON": "Novartis",
        "NSRGY": "Nestle",
        "RHO6.FRK": "Roche",
        "ABBNY": "ABB",
        "UBS": "UBS",
        "0QP2.LON": "Zurich_Insurance",
        "0QKY.LON": "Holcim",
        "0QNO.LON": "Lonza",
        "0QPS.LON": "Givaudan",
        "0A0D.LON": "Alcon",
        "0Z4C.LON": "Sika",
        "0QOQ.LON": "Partners_Group",
        "0QMG.LON": "Swiss_Life",
        "0QQ2.LON": "Geberit",
        "0QMW.LON": "Kuehne_Nagel",
        "0QK6.LON": "Logitech",
        "AMRZ": "Amrize"
    }

    for f in files:
        # Extract symbol from filename "SYMBOL_daily_data.csv"
        symbol = f.replace("_daily_data.csv", "")
        
        # Determine Company Name
        company_name = TICKER_MAP.get(symbol, "Unknown")
        
        # Process
        try:
            # Pass company_name to process_stock if needed, or just handle saving here?
            # Let's modify process_stock to accept an output filename or handle it inside.
            # Actually, better to refactor process_stock to take the output path as argument.
            # But for minimal changes, I'll just change how process_stock saves.
            # Wait, process_stock is defined above. I need to modify process_stock signature or logic.
            # Let's modify process_stock to accept output_filename.
            pass 
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        
        # Pause to be nice to API
        time.sleep(1)

# Redefining process_stock to accept output_name
def process_stock(symbol, df_macro, output_name):
    print(f"\nProcessing {symbol} ({output_name})...")
    
    # 1. Fetch OHLCV
    df_ohlcv = get_alpha_vantage_data("TIME_SERIES_DAILY", symbol=symbol, outputsize="full")
    if df_ohlcv is None:
        return
    
    df_ohlcv["timestamp"] = pd.to_datetime(df_ohlcv["timestamp"])
    df_ohlcv = df_ohlcv.set_index("timestamp").sort_index()
    
    # 2. Technical Indicators
    indicators = ["SMA", "EMA", "RSI", "ATR", "BBANDS"]
    time_periods = [14] # Default
    data = df_ohlcv.merge(df_indicators, left_index=True, right_index=True, how="left")
    
    # 3. Feature Engineering
    # Target Volatility
    data["ctc_vol"] = fe.volatility.close_to_close_volatility(df=data, close_col="close", window_size=5)
    data["parkinson_vol"] = fe.volatility.parkinson_volatility(df=data, high_col="high", low_col="low", window_size=5)
    
    # Returns
    data['return'] = data['close'].pct_change()
    data["log_return"] = np.log(data["close"] / data["close"].shift(1))
    data["abs_log_return"] = data["log_return"].abs()
    
    # Candle Features
    data["range_t"] = np.log(data["high"] / data["low"])
    data["body"] = (data["close"] - data["open"]).abs()
    data["upper_wick"] = data["high"] - data[["open", "close"]].max(axis=1)
    data["lower_wick"] = data[["open", "close"]].min(axis=1) - data["low"]
    
    # Lags
    features_to_lag = data.columns.tolist()
    for col in features_to_lag:
        for lag in [1, 2, 3]:
            data[f"{col}_lag_{lag}"] = data[col].shift(lag)
            
    # Rolling Features
    data["vol_5_mean"] = data["volume"].rolling(5).mean().shift(1)
    data["vol_20_mean"] = data["volume"].rolling(20).mean().shift(1)
    
    # 4. Merge with Macro
    data = data.merge(df_macro, left_index=True, right_index=True, how="left")
    
    # Drop NaNs created by lags and macro merge
    data = data.dropna()
    
    # Save
    output_file = f"processed_data/{output_name}"
    os.makedirs("processed_data", exist_ok=True)
    data.to_csv(output_file)
    print(f"  Saved to {output_file} ({len(data)} rows)")

def main():
    # 1. Get Macro Data (once for all stocks)
    df_macro = get_macro_data()
    print(f"Macro Data Ready: {len(df_macro)} rows")
    
    # 2. Find all stocks in alphavantage_data
    data_dir = "alphavantage_data"
    files = [f for f in os.listdir(data_dir) if f.endswith("_daily_data.csv")]
    
    print(f"Found {len(files)} stock files to process.")
    
    # Ticker to Company Name Mapping
    TICKER_MAP = {
        "0QKI.LON": "Swisscom",
        "0QLR.LON": "Novartis",
        "NSRGY": "Nestle",
        "RHO6.FRK": "Roche",
        "ABBNY": "ABB",
        "UBS": "UBS",
        "0QP2.LON": "Zurich_Insurance",
        "0QKY.LON": "Holcim",
        "0QNO.LON": "Lonza",
        "0QPS.LON": "Givaudan",
        "0A0D.LON": "Alcon",
        "0Z4C.LON": "Sika",
        "0QOQ.LON": "Partners_Group",
        "0QMG.LON": "Swiss_Life",
        "0QQ2.LON": "Geberit",
        "0QMW.LON": "Kuehne_Nagel",
        "0QK6.LON": "Logitech",
        "AMRZ": "Amrize"
    }
    
    for f in files:
        # Extract symbol from filename "SYMBOL_daily_data.csv"
        symbol = f.replace("_daily_data.csv", "")
        
        # Determine Output Filename
        company_name = TICKER_MAP.get(symbol, "Unknown")
        output_name = f"data_{company_name}_{symbol}.csv"
        
        # Process
        try:
            process_stock(symbol, df_macro, output_name)
        except Exception as e:
            print(f"Error processing {symbol}: {e}")
        
        # Pause to be nice to API
        time.sleep(1)

if __name__ == "__main__":
    main()
