#!/usr/bin/env python3
"""
generate_daily_forecasts.py

Creates:
- ONE live prediction for NEXT TRADING DAY (T+1 business day)
- PLUS last 5 historical next-trading-day predictions (backtests)
- Writes a daily snapshot forecast_history_YYYY-MM-DD.json
- Copies snapshot into forecast_history.json (latest)

Reads paths from .env in repo root:
  FORECAST_STORE_DIR=jobs/data_out
  STOCK_CSV_DIR=jobs/data_csv  (may contain subfolders like latest/)
"""

import os
import json
import glob
import time
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay
from tqdm import tqdm
import yfinance as yf

import torch
from dotenv import load_dotenv
from chronos import Chronos2Pipeline


# =========================================================
# PATHS / ENV
# =========================================================
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))          # .../root/jobs
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))       # .../root

ENV_PATH = os.path.join(ROOT_DIR, ".env")
load_dotenv(dotenv_path=ENV_PATH)

FORECAST_STORE_DIR = os.getenv("FORECAST_STORE_DIR", "jobs/data_out")
STOCK_CSV_DIR = os.getenv("STOCK_CSV_DIR", "jobs/data_csv")

FORECAST_DIR = os.path.join(ROOT_DIR, FORECAST_STORE_DIR)
CSV_DIR = os.path.join(ROOT_DIR, STOCK_CSV_DIR)

os.makedirs(FORECAST_DIR, exist_ok=True)
SNAPSHOT_DIR = os.path.join(FORECAST_DIR, "snapshots")
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

LATEST_FILE = os.path.join(FORECAST_DIR, "forecast_history.json")
LOCK_FILE = os.path.join(FORECAST_DIR, ".forecast_job.lock")

print(f"✅ ROOT_DIR: {ROOT_DIR}")
print(f"✅ .env: {ENV_PATH} (exists={os.path.exists(ENV_PATH)})")
print(f"📂 CSV INPUT DIR: {CSV_DIR} (exists={os.path.exists(CSV_DIR)})")
print(f"📁 FORECAST OUTPUT DIR: {FORECAST_DIR}")
print(f"📝 HISTORY FILE: {LATEST_FILE}")
print("=" * 60)


# =========================================================
# CONFIG
# =========================================================
DATE_COL = "timestamp"
TARGET_COL = "ewma_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"

LAMBDA = 0.94
FEATURE_COLS = ["ewma_vol_lag_1", "vix_lag_1"]

# Desired behavior:
DAYS_BACK = 5

# IMPORTANT:
# Chronos expects timestamps consistent with inferred freq.
# We resample context to daily ("D"), so future_df MUST also be daily.
# We then pick the prediction step matching NEXT BUSINESS DAY.
INTERNAL_PRED_LEN_FOR_FREQ = 7  # must cover weekends safely (Fri -> Mon needs 3 steps)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE_PREF = pick_device()


# =========================================================
# HELPERS
# =========================================================
def to_datestr(x) -> str | None:
    if x is None:
        return None
    return pd.to_datetime(x).strftime("%Y-%m-%d")


def atomic_write_json(path: str, data: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=os.path.dirname(path))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def acquire_lock(lock_path: str, stale_after_seconds: int = 3600) -> bool:
    try:
        if os.path.exists(lock_path):
            age = time.time() - os.path.getmtime(lock_path)
            if age > stale_after_seconds:
                try:
                    os.remove(lock_path)
                except Exception:
                    pass

        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(json.dumps({"pid": os.getpid(), "created_at": time.time()}))
        return True
    except FileExistsError:
        return False


def release_lock(lock_path: str) -> None:
    try:
        if os.path.exists(lock_path):
            os.remove(lock_path)
    except Exception:
        pass


def is_weekend_datestr(yyyy_mm_dd: str) -> bool:
    try:
        d = pd.to_datetime(yyyy_mm_dd)
        return d.weekday() >= 5
    except Exception:
        return False


# =========================================================
# DATA PREP
# =========================================================
def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    n = len(returns)
    ewma_var = np.zeros(n, dtype=float)
    r = returns.fillna(0.0).values
    ewma_var[0] = r[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t - 1] + (1 - lambda_) * (r[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


def load_vix() -> pd.DataFrame:
    print("Fetching VIX from Yahoo Finance...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="2y")
    hist.index = hist.index.tz_localize(None)

    vix_df = hist[["Close"]].rename(columns={"Close": "vix"}).copy()
    vix_df["date"] = pd.to_datetime(vix_df.index).normalize()
    vix_df = vix_df.reset_index(drop=True)

    vix_df["vix"] = vix_df["vix"] / 100.0
    vix_df = vix_df.sort_values("date").reset_index(drop=True)
    vix_df["vix_lag_1"] = vix_df["vix"].shift(1)
    return vix_df


def prepare_data(filepath: str, vix_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(filepath)
    if DATE_COL not in df.columns:
        raise ValueError(f"Missing {DATE_COL} in {filepath}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID

    df["date"] = pd.to_datetime(df[DATE_COL]).dt.normalize()

    close_col = "Close" if "Close" in df.columns else "close"
    if close_col not in df.columns:
        raise ValueError(f"Missing Close/close in {filepath}")

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))

    df[TARGET_COL] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df[TARGET_COL].shift(1)

    v = vix_df[["date", "vix", "vix_lag_1"]].copy()
    df = df.merge(v, on="date", how="left")

    # critical: fill VIX gaps after merge
    df = df.sort_values("date").reset_index(drop=True)
    df["vix"] = df["vix"].ffill()
    df["vix_lag_1"] = df["vix_lag_1"].ffill()

    required = [DATE_COL, TARGET_COL] + FEATURE_COLS
    df = df.dropna(subset=required).reset_index(drop=True)
    df = df.drop(columns=["date"], errors="ignore")
    return df


# =========================================================
# PREDICTION (NEXT TRADING DAY)
# =========================================================
def predict_next_trading_day(pipeline: Chronos2Pipeline, df: pd.DataFrame) -> dict:
    """
    Predict for NEXT BUSINESS DAY (BDay(1)), while keeping Chronos timestamps DAILY.

    Why:
    - context is resampled to daily ("D")
    - Chronos validates future timestamps based on expected daily continuation
    - so future_df must also be daily
    - we then SELECT the step whose timestamp == next business day
    """
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # last TRADING day from original data
    last_trading_ts = pd.to_datetime(df[DATE_COL].iloc[-1]).normalize()
    next_bd = last_trading_ts + BDay(1)

    # daily context continuity for Chronos
    context_df = df.iloc[-600:].copy()
    context_df = context_df.set_index(DATE_COL).resample("D").ffill().reset_index()

    # anchor for daily future grid (must be the context's last timestamp)
    last_ctx = context_df.iloc[-1]
    last_ctx_date = pd.to_datetime(last_ctx[DATE_COL]).normalize()

    # build DAILY future timestamps (this makes Chronos validation pass)
    future_rows = []
    for i in range(1, INTERNAL_PRED_LEN_FOR_FREQ + 1):
        future_rows.append({
            ID_COL: last_ctx[ID_COL],
            DATE_COL: last_ctx_date + pd.Timedelta(days=i),  # DAILY, not BDay
            "ewma_vol_lag_1": float(last_ctx[TARGET_COL]),
            "vix_lag_1": float(last_ctx["vix"]),
        })
    future_df = pd.DataFrame(future_rows)

    context_cols = [ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS
    future_cols = [ID_COL, DATE_COL] + FEATURE_COLS

    pred_df = pipeline.predict_df(
        context_df[context_cols],
        future_df=future_df[future_cols],
        prediction_length=INTERNAL_PRED_LEN_FOR_FREQ,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=ID_COL,
        timestamp_column=DATE_COL,
        target=TARGET_COL,
        # keep validation ON (we fixed timestamps properly)
    )

    # select the row matching next business day
    pred_ts = pd.to_datetime(pred_df[DATE_COL]).dt.normalize()
    idx = pred_ts[pred_ts == next_bd].index

    if len(idx) == 0:
        # as a safety net (e.g. around holidays), extend horizon by picking first date > next_bd
        idx = pred_ts[pred_ts > next_bd].index
        if len(idx) == 0:
            raise RuntimeError("Could not locate next business day prediction in prediction window.")
        use_i = idx[0]
        chosen_date = pred_ts.loc[use_i]
    else:
        use_i = idx[0]
        chosen_date = next_bd

    return {
        "target_date": to_datestr(chosen_date),
        "forecast_value": float(pred_df["0.5"].loc[use_i]),
        "confidence_low": float(pred_df["0.1"].loc[use_i]),
        "confidence_high": float(pred_df["0.9"].loc[use_i]),
        "last_known_date": to_datestr(last_trading_ts),
        "last_known_vol": float(df[TARGET_COL].iloc[-1]),
    }


# =========================================================
# MAIN
# =========================================================
def main():
    print("🚀 STARTING DAILY FORECAST JOB (T+1 + T-5 HISTORY)")
    print("=" * 60)

    if not acquire_lock(LOCK_FILE):
        print("⚠️ Forecast job already running (lock exists). Exiting.")
        return

    try:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto")
        print(f"✅ Model loaded (preferred device: {DEVICE_PREF})")

        vix_df = load_vix()

        if os.path.exists(LATEST_FILE):
            with open(LATEST_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {"data": {}}

        history["last_updated"] = datetime.now().isoformat()
        now_utc = datetime.utcnow().isoformat() + "Z"

        csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "**", "data_*.csv"), recursive=True))
        if not csv_files:
            print(f"⚠️ No CSV files found in {CSV_DIR}. Expected pattern: **/data_*.csv")
            return

        for fp in tqdm(csv_files, desc="Forecasting Stocks"):
            symbol = os.path.basename(fp).replace("data_", "").replace(".csv", "")

            try:
                full_df = prepare_data(fp, vix_df)
                if len(full_df) < 80:
                    continue

                if symbol not in history["data"]:
                    history["data"][symbol] = []

                # map existing by (target_date, type)
                existing = {}
                for r in history["data"][symbol]:
                    t = r.get("type")
                    if t not in {"forecast_next_day", "backtest_next_day"}:
                        continue
                    td = to_datestr(r.get("target_date"))
                    if not td:
                        continue

                    r2 = dict(r)
                    r2["target_date"] = td
                    r2["last_known_date"] = to_datestr(r.get("last_known_date"))

                    if t == "backtest_next_day" and r2.get("actual_value") is None:
                        continue

                    existing[(td, t)] = r2

                # 1) live forecast (next business day)
                pred = predict_next_trading_day(pipeline, full_df)
                if is_weekend_datestr(pred["target_date"]):
                    # should not happen, but guard
                    pred["target_date"] = to_datestr(pd.to_datetime(pred["target_date"]) + BDay(1))

                actual_row = full_df[pd.to_datetime(full_df[DATE_COL]).dt.normalize() == pd.to_datetime(pred["target_date"])]
                actual_val = float(actual_row.iloc[0][TARGET_COL]) if not actual_row.empty else None

                live_rec = {
                    "target_date": pred["target_date"],
                    "forecast_value": pred["forecast_value"],
                    "confidence_low": pred["confidence_low"],
                    "confidence_high": pred["confidence_high"],
                    "actual_value": actual_val,
                    "last_known_date": pred["last_known_date"],
                    "last_known_vol": pred["last_known_vol"],
                    "generated_at": now_utc,
                    "type": "forecast_next_day",
                }
                existing[(live_rec["target_date"], "forecast_next_day")] = live_rec

                # 2) backtests (last 5)
                for back in range(1, DAYS_BACK + 1):
                    df_slice = full_df.iloc[:-back].copy()
                    if len(df_slice) < 80:
                        continue

                    bt = predict_next_trading_day(pipeline, df_slice)
                    if is_weekend_datestr(bt["target_date"]):
                        continue

                    actual_row = full_df[pd.to_datetime(full_df[DATE_COL]).dt.normalize() == pd.to_datetime(bt["target_date"])]
                    actual_val = float(actual_row.iloc[0][TARGET_COL]) if not actual_row.empty else None
                    if actual_val is None:
                        continue

                    bt_rec = {
                        "target_date": bt["target_date"],
                        "forecast_value": bt["forecast_value"],
                        "confidence_low": bt["confidence_low"],
                        "confidence_high": bt["confidence_high"],
                        "actual_value": actual_val,
                        "last_known_date": bt["last_known_date"],
                        "last_known_vol": bt["last_known_vol"],
                        "generated_at": now_utc,
                        "type": "backtest_next_day",
                    }
                    existing[(bt_rec["target_date"], "backtest_next_day")] = bt_rec

                # cleanup + sort
                cleaned = []
                for r in existing.values():
                    if r.get("type") == "forecast_next_day" and is_weekend_datestr(r.get("target_date", "")):
                        continue
                    cleaned.append(r)

                history["data"][symbol] = sorted(cleaned, key=lambda x: (x.get("target_date", ""), x.get("type", "")))

            except Exception as e:
                print(f"⚠️ Skipping {symbol}: {e}")

        # snapshot + latest
        today = datetime.now().strftime("%Y-%m-%d")
        snapshot_file = os.path.join(SNAPSHOT_DIR, f"forecast_history_{today}.json")
        atomic_write_json(snapshot_file, history)
        atomic_write_json(LATEST_FILE, history)

        print(f"\n📸 Snapshot written: {snapshot_file}")
        print(f"✅ Latest updated:   {LATEST_FILE}")

    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()
