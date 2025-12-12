
import os
import json
import glob
import pandas as pd
import numpy as np
from datetime import datetime
from tqdm import tqdm
from chronos import Chronos2Pipeline

# ==========================================
# CONFIGURATION
# ==========================================
# If you move this script, update ROOT_DIR to point to the 'stonksai' folder
# Currently assumes script is in stonksai/ewma_volatility or similar
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Try to find the project root intelligently
# We look for "processed_data" to verify we found the right place
POSSIBLE_ROOTS = [
    os.path.join(SCRIPT_DIR, ".."),      # If in ewma_volatility/
    os.path.join(SCRIPT_DIR, "stonksai"), # If in parent of stonksai
    SCRIPT_DIR                           # If in root
]

ROOT_DIR = None
for path in POSSIBLE_ROOTS:
    if os.path.exists(os.path.join(path, "processed_data")):
        ROOT_DIR = os.path.abspath(path)
        break

if ROOT_DIR is None:
    # Fallback/Hardcoded: User might need to set this if moving script far away
    # Please update this path if the script fails to find data!
    ROOT_DIR = SCRIPT_DIR
    print(f"WARNING: Could not auto-detect project root. Using relative path: {ROOT_DIR}")

DATA_DIR = os.path.join(ROOT_DIR, "data")
# Adjust VIX path based on your folder structure. 
# Assuming data_fred is in the project root.
VIX_PATH = os.path.join(ROOT_DIR, "data_fred", "vixcls.csv") 
HISTORY_FILE = os.path.join(SCRIPT_DIR, "forecast_history.json")

DEVICE = "cuda" # or "cpu"
DATE_COL = "timestamp"
TARGET_COL = "ewma_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"
LAMBDA = 0.94
FEATURE_COLS = ["ewma_vol_lag_1", "vix_lag_1"]

# ==========================================
# HELPER FUNCTIONS (Copied from inference_ewma.py)
# ==========================================

def calculate_ewma_volatility(returns, lambda_=0.94):
    """Calculate EWMA volatility."""
    n = len(returns)
    ewma_var = np.zeros(n)
    returns_filled = returns.fillna(0).values
    ewma_var[0] = returns_filled[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t-1] + (1 - lambda_) * (returns_filled[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


import yfinance as yf

def load_vix():
    """Load and prepare VIX data using yfinance (Real-time/Delayed)."""
    print("Fetching VIX from Yahoo Finance...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="2y") # Get enough history for lag features
    hist.index = hist.index.tz_localize(None)
    
    # Standardize column names
    vix_df = hist[["Close"]].rename(columns={"Close": "vix"})
    vix_df["date"] = vix_df.index
    vix_df = vix_df.reset_index(drop=True)
    
    # Normalize (raw value is percentage, e.g. 15.0. Model expects 0.15?)
    # CHECK: inference_ewma.py did "vix['vix'] / 100".
    # Let's verify what data_final.py does.
    # FRED data passed as is? 
    # In generate_daily_forecasts: "vix['vix'] = vix['vix'] / 100"
    
    # Yes, divide by 100.
    vix_df['vix'] = vix_df['vix'] / 100 
    
    vix_df['vix_lag_1'] = vix_df['vix'].shift(1)
    return vix_df

def prepare_data(filepath, vix_df):
    """Load and prepare stock data with EWMA and VIX."""
    df = pd.read_csv(filepath)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID
    
    # Calculate log returns
    if "log_return" not in df.columns:
        # Check for 'Close' or 'close' (case insensitive fallback)
        close_col = "Close" if "Close" in df.columns else "close"
        df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))
    
    # Calculate EWMA
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df["ewma_vol"].shift(1)
    
    # Merge VIX
    # We keep 'vix' (raw) to construct the next day's features easily
    df = df.merge(vix_df[['date', 'vix', 'vix_lag_1']], left_on=DATE_COL, right_on='date', how='left')
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

def predict_next_day(pipeline, df, stock_name):
    """
    Predicts the volatility for the NEXT business day (T+1) based on the latest available data (T).
    """
    # Ensure sorted
    df = df.sort_values(DATE_COL).reset_index(drop=True)
    
    # 1. Get latest state (Day T)
    last_row = df.iloc[-1]
    last_date = last_row[DATE_COL]
    
    # 2. Resample Context to 'D' (Daily) to avoid 'Business Day' validation headaches
    context_df = df.iloc[-600:].copy()
    context_df = context_df.set_index(DATE_COL).resample('D').ffill().reset_index()
    
    last_row_D = context_df.iloc[-1]
    last_date_D = last_row_D[DATE_COL]
    
    # Next prediction day
    next_date_D = last_date_D + pd.Timedelta(days=1)
    
    # Future Frame (5 days for robust freq inference)
    future_rows = []
    for i in range(5):
        d = next_date_D + pd.Timedelta(days=i)
        future_rows.append({
            ID_COL: last_row_D[ID_COL],
            DATE_COL: d,
            "ewma_vol_lag_1": last_row_D["ewma_vol"],
            "vix_lag_1": last_row_D["vix"]
        })
    
    future_df = pd.DataFrame(future_rows)
    
    context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    future_cols = [ID_COL, DATE_COL] + FEATURE_COLS
    
    # 4. Predict
    try:
        pred_df = pipeline.predict_df(
            context_df[context_cols],
            future_df=future_df[future_cols],
            prediction_length=5, # Request 5 steps
            quantile_levels=[0.1, 0.5, 0.9],
            id_column=ID_COL,
            timestamp_column=DATE_COL,
            target=TARGET_COL
        )
        
        forecast_val = pred_df["0.5"].iloc[0]
        q10 = pred_df["0.1"].iloc[0]
        q90 = pred_df["0.9"].iloc[0]
        
        return {
            "stock": stock_name,
            "forecast_date": next_date_D,
            "forecast_value": forecast_val,
            "confidence_interval": (q10, q90),
            "last_date": last_date,
            "last_actual_vol": last_row["ewma_vol"],
            "status": "success"
        }
    except Exception as e:
        return {
            "stock": stock_name,
            "status": "error",
            "error": str(e)
        }

# ==========================================
# MAIN EXECUTION
# ==========================================

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not read existing history ({e}). Starting fresh.")
    return {"data": {}}

def save_history(history):
    # Add metadata
    history["last_updated"] = datetime.now().isoformat()
    
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2, default=str)
    print(f"History saved to {HISTORY_FILE}")

def main():
    print("="*60)
    print("DAILY VOLATILITY FORECAST GENERATOR (STANDALONE)")
    print("="*60)
    print(f"Root Dir: {ROOT_DIR}")
    print(f"Data Dir: {DATA_DIR}")
    
    # 1. Load Model
    print("Loading Model...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)
    
    # 2. Load VIX
    print("Loading VIX...")
    try:
        vix_df = load_vix()
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return
    
    # 3. Load History
    history = load_history()
    if "data" not in history:
        history["data"] = {}
        
    # 4. Find Files
    if not os.path.exists(DATA_DIR):
        print(f"CRITICAL ERROR: Data directory not found at {DATA_DIR}")
        return
        
    csv_files = sorted(glob.glob(os.path.join(DATA_DIR, "data_*.csv")))
    print(f"Found {len(csv_files)} stocks.")
    
    # 5. Loop
    # How many past days to re-forecast for context?
    DAYS_BACK = 3
    
    for filepath in tqdm(csv_files, desc="Generating Forecasts"):
        stock_name = os.path.basename(filepath).replace("data_", "").replace(".csv", "")
        stock_name = stock_name.split("_")[0]
        
        try:
            df = prepare_data(filepath, vix_df)
            
            # Helper to run prediction at a specific index (simulating "as of that day")
            # We want to predict for T+1, T, T-1, T-2...
            # The 'predict_next_day' function takes the whole DF and uses the *last row* as the anchor.
            # So to predict for a past day, we just slice the DF to end at that day.
            
            # Example: 
            # If DF has 100 rows.
            # Slice [:100] -> Predicts for Day 101 (The "Next Day" / Tomorrow)
            # Slice [:99]  -> Predicts for Day 100 (Today) based on Day 99 information.
            # Slice [:98]  -> Predicts for Day 99 (Yesterday) based on Day 98 information.
            
            # We want: 
            # 1. Forecast for Tomorrow (using all data)
            # 2. Forecast for Today (using data until yesterday) + Compare with Actual Today
            # 3. Forecast for Yesterday (using data until day before) + Compare with Actual Yesterday
            # ...
            
            new_records = []
            
            # Loop from 0 (Tomorrow) down to DAYS_BACK
            for i in range(DAYS_BACK + 1):
                # i=0: Full data -> Forecast Tomorrow
                # i=1: Drop last row -> Forecast Today
                # i=2: Drop last 2 rows -> Forecast Yesterday
                
                if i == 0:
                    current_df = df # Full data
                else:
                    current_df = df.iloc[:-i] # Slice back
                
                # Check if we have enough data left
                if len(current_df) < 50:
                    continue
                
                # Run prediction
                result = predict_next_day(pipeline, current_df, stock_name)
                
                if result is None:
                    continue
                    
                if result.get("status") == "success":
                    forecast_date_str = str(result["forecast_date"])
                    
                    # Try to find ACTUAL value for comparison
                    # The forecast is for 'forecast_date'. Do we have that date in our original full 'df'?
                    actual_value = None
                    actual_row = df[df[DATE_COL] == result["forecast_date"]]
                    if not actual_row.empty:
                        actual_value = float(actual_row.iloc[0][TARGET_COL])
                    else:
                        # Future date (tomorrow), no actual value yet
                        actual_value = None

                    record = {
                        "target_date": conversation_date_standard(forecast_date_str),
                        "forecast_value": float(result["forecast_value"]),
                        "confidence_low": float(result["confidence_interval"][0]),
                        "confidence_high": float(result["confidence_interval"][1]),
                        "actual_value": actual_value, # New field: Truth
                        "last_known_date": str(result["last_date"]),
                        "last_known_vol": float(result["last_actual_vol"]),
                        "generated_at": datetime.now().isoformat(),
                         # Tag to identify if this is a "Re-Forecast" of past or a fresh "Next Day"
                        "type": "forecast_next_day" if i == 0 else "historical_reforecast"
                    }
                    new_records.append(record)
                    
                    if i == 0:
                        print(f"  [OK] {stock_name} (Next): {record['forecast_value']:.4f}")
                else:
                    print(f"  [ERR] {stock_name}: {result.get('error')}")

            # Update History
            if new_records:
                if stock_name not in history["data"]:
                    history["data"][stock_name] = []
                
                existing_list = history["data"][stock_name]
                
                # Merge logic:
                # For each new record, replace existing record with same target_date, or append.
                # Use a dict for easy lookup by date
                history_dict = {x["target_date"]: x for x in existing_list}
                
                for rec in new_records:
                    # Update or Add
                    history_dict[rec["target_date"]] = rec
                
                # Convert back to list and sort
                final_list = list(history_dict.values())
                final_list.sort(key=lambda x: x["target_date"])
                
                # Optional: Limit history length? Keep last 100 entries?
                # final_list = final_list[-100:] 
                
                history["data"][stock_name] = final_list
                
        except Exception as e:
            print(f"Critical error on {stock_name}: {e}")
            import traceback
            traceback.print_exc()
            
    # 6. Save
    save_history(history)
    print("\nAll done!")

def conversation_date_standard(date_str):
    # Just ensures string consistency if needed
    return str(pd.to_datetime(date_str))

if __name__ == "__main__":
    main()
