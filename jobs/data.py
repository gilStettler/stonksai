"""
Data Ingestion Job
==================

- Downloads daily OHLCV data from Alpha Vantage
- Computes daily returns
- Fetches VIX data from Yahoo Finance
- Merges stock data with VIX
- Writes:
  - snapshots -> jobs/data_csv/snapshots/YYYY-MM-DD/   (yesterday)
  - latest    -> jobs/data_csv/latest/

Triggered:
- manually: python data.py
- via API:  POST /v1/admin/jobs/ingest
"""

import os
import time
import shutil
import argparse
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

from dotenv import load_dotenv

# =========================================================
# Auth helper (optional)
# =========================================================

def require_job_auth(passed_token: str | None):
    """
    If JOB_TRIGGER_TOKEN is set, require matching token.
    """
    expected = os.getenv("JOB_TRIGGER_TOKEN", "").strip()
    if not expected:
        return

    got = (passed_token or os.getenv("JOB_RUN_TOKEN", "")).strip()
    if not got or got != expected:
        raise SystemExit(
            "[FATAL] Invalid or missing job token. "
            "Provide --token or set JOB_RUN_TOKEN."
        )

# =========================================================
# Repo root & .env
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # stonksai/jobs
REPO_ROOT = SCRIPT_DIR.parent                         # stonksai/
ENV_PATH = REPO_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    print(f"[WARN] .env not found at {ENV_PATH}")

# =========================================================
# Paths (ENV-driven, repo-root relative)
# =========================================================

DATA_CSV_BASE_DIR = Path(
    os.getenv("STOCK_CSV_DIR", "jobs/data_csv")
)
DATA_CSV_BASE_DIR = (
    DATA_CSV_BASE_DIR
    if DATA_CSV_BASE_DIR.is_absolute()
    else (REPO_ROOT / DATA_CSV_BASE_DIR).resolve()
)

LATEST_DIR = DATA_CSV_BASE_DIR / "latest"
SNAPSHOT_DIR = DATA_CSV_BASE_DIR / "snapshots"

LATEST_DIR.mkdir(parents=True, exist_ok=True)
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)

# =========================================================
# Config
# =========================================================

ALPHAVANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "").strip()
ALPHAVANTAGE_URL = "https://www.alphavantage.co/query"

# AlphaVantage free tier ≈ 5 requests / minute
SLEEP_SECONDS = float(os.getenv("ALPHAVANTAGE_SLEEP_SECONDS", "15"))

VIX_PERIOD = os.getenv("VIX_PERIOD", "max")  # "2y" or "max"

DEFAULT_TICKERS = [
    "0QKI.LON", "0QLR.LON", "NSRGY", "RHO6.FRK", "ABBNY", "UBS",
    "0QP2.LON", "0QKY.LON", "0QNO.LON", "0QPS.LON", "0A0D.LON",
    "0Z4C.LON", "0QOQ.LON", "0QMG.LON", "0QQ2.LON", "0QMW.LON",
    "0QK6.LON",
]

TICKERS_ENV = os.getenv("TICKERS", "").strip()
if TICKERS_ENV:
    TICKERS = [t.strip() for t in TICKERS_ENV.split(",") if t.strip()]
else:
    TICKERS = DEFAULT_TICKERS

# =========================================================
# Helpers
# =========================================================

def safe_filename(symbol: str) -> str:
    return symbol.replace("/", "_").replace("\\", "_")


def atomic_copy_to_latest(src_path: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    tmp = dst_dir / (src_path.name + ".tmp")
    final = dst_dir / src_path.name

    shutil.copyfile(src_path, tmp)
    os.replace(tmp, final)
    return final


def get_snapshot_day_dir(base_snapshot_dir: Path, day: str) -> Path:
    """
    Returns snapshots/<YYYY-MM-DD>/ and ensures it exists.
    """
    day_dir = base_snapshot_dir / day
    day_dir.mkdir(parents=True, exist_ok=True)
    return day_dir


def get_vix_from_yfinance(period: str = "max") -> pd.DataFrame:
    print("[vix] Fetching VIX from Yahoo Finance (^VIX)...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(period=period)

    if hist.empty:
        raise RuntimeError("VIX history returned empty.")

    hist.index = hist.index.tz_localize(None)
    df = hist[["Close"]].rename(columns={"Close": "VIX"})
    df.index.name = "timestamp"
    return df.sort_index()


def get_daily_ohlcv_from_alpha_vantage(symbol: str) -> pd.DataFrame:
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "outputsize": "full",
        "apikey": ALPHAVANTAGE_API_KEY,
    }

    r = requests.get(ALPHAVANTAGE_URL, params=params, timeout=30)
    data = r.json()

    if "Error Message" in data:
        raise ValueError(data["Error Message"])
    if "Note" in data:
        raise RuntimeError(data["Note"])
    if "Information" in data:
        raise RuntimeError(data["Information"])

    key = "Time Series (Daily)"
    if key not in data:
        raise ValueError(f"Missing '{key}' in AlphaVantage response.")

    df = pd.DataFrame.from_dict(data[key], orient="index").rename(columns={
        "1. open": "Open",
        "2. high": "High",
        "3. low": "Low",
        "4. close": "Close",
        "6. volume": "Volume",
    })

    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df[["Open", "High", "Low", "Close", "Volume"]].astype(float)

    return df

# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="Job trigger token")
    args = parser.parse_args()

    require_job_auth(args.token)

    if not ALPHAVANTAGE_API_KEY:
        raise SystemExit(
            "[FATAL] Missing ALPHAVANTAGE_API_KEY (set it in .env)"
        )

    # ✅ snapshots should be stored under yesterday's date
    run_day = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    day_snapshot_dir = get_snapshot_day_dir(SNAPSHOT_DIR, run_day)

    print("=" * 60)
    print("DATA INGESTION JOB")
    print("=" * 60)
    print(f"[out] BASE_DIR:         {DATA_CSV_BASE_DIR}")
    print(f"[out] LATEST_DIR:       {LATEST_DIR}")
    print(f"[out] SNAPSHOT_DIR:     {SNAPSHOT_DIR}")
    print(f"[out] DAY_SNAPSHOT_DIR: {day_snapshot_dir}")
    print(f"[tickers] {len(TICKERS)}")
    print(f"[rate] sleep {SLEEP_SECONDS}s")

    vix_df = get_vix_from_yfinance(period=VIX_PERIOD)

    for i, symbol in enumerate(TICKERS, start=1):
        print(f"\n[{i}/{len(TICKERS)}] {symbol}")

        try:
            stock_df = get_daily_ohlcv_from_alpha_vantage(symbol)
            stock_df["Return"] = stock_df["Close"].pct_change()

            merged = stock_df.join(vix_df, how="left")

            fname = f"data_{safe_filename(symbol)}.csv"

            # ✅ Snapshot in yesterday folder
            snapshot_path = day_snapshot_dir / fname
            merged.to_csv(snapshot_path, index_label="timestamp")
            print(f"[ok] snapshot {snapshot_path}")

            # ✅ Latest as before
            latest_path = atomic_copy_to_latest(snapshot_path, LATEST_DIR)
            print(f"[ok] latest   {latest_path}")

        except Exception as e:
            print(f"[err] {symbol}: {e}")

        if i < len(TICKERS):
            time.sleep(SLEEP_SECONDS)

    print("\nDONE.")

# =========================================================

if __name__ == "__main__":
    main()
