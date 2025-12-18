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
  STOCK_CSV_DIR=jobs/data_csv

CSV format (new):
timestamp,Open,High,Low,Close,Volume,Return,VIX,Symbol,Company

Keying (company-only):
- history["data"] keyed ONLY by company_name
- records contain company_name but NO ticker/symbol fields

Safety:
- A meta mapping is stored once:
    history["meta"]["company_to_symbol"][company_name] = ticker_symbol
  so the backend can still find the correct CSV file data_<ticker>.csv

Migration:
- On load, old histories keyed by ticker are migrated to company_name keys if possible.
- Duplicate records merged (dedup by (target_date, type)).
"""

import os
import json
import glob
import time
import tempfile
from datetime import datetime, timedelta

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
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

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

print(f"ROOT_DIR: {ROOT_DIR}")
print(f".env: {ENV_PATH} (exists={os.path.exists(ENV_PATH)})")
print(f"CSV INPUT DIR: {CSV_DIR} (exists={os.path.exists(CSV_DIR)})")
print(f"FORECAST OUTPUT DIR: {FORECAST_DIR}")
print(f"HISTORY FILE: {LATEST_FILE}")
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

DAYS_BACK = 5
INTERNAL_PRED_LEN_FOR_FREQ = 7


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


def normalize_company_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize Company safely:
    - Keep inner spaces (e.g. "Swisscom AG" stays as-is)
    - Strip only leading/trailing whitespace
    - Convert non-breaking spaces to normal spaces
    - Treat "", "nan", "none" as missing
    """
    if "Company" not in df.columns:
        return df

    s = df["Company"]
    s = s.apply(lambda x: x.replace("\xa0", " ") if isinstance(x, str) else x)
    s = s.apply(lambda x: x.strip() if isinstance(x, str) else x)

    def _clean(x):
        if x is None:
            return None
        if isinstance(x, float) and np.isnan(x):
            return None
        if isinstance(x, str):
            low = x.lower()
            if x == "" or low in {"nan", "none", "null"}:
                return None
        return x

    df["Company"] = s.apply(_clean)
    return df


def parse_symbol_and_company_from_csv(df: pd.DataFrame, filepath: str) -> tuple[str, str | None]:
    symbol_from_fn = os.path.basename(filepath).replace("data_", "").replace(".csv", "")

    symbol = None
    if "Symbol" in df.columns:
        s = df["Symbol"].dropna()
        if not s.empty:
            symbol = str(s.iloc[-1]).strip() or None

    company = None
    if "Company" in df.columns:
        c = df["Company"].dropna()
        if not c.empty:
            company = str(c.iloc[-1]).strip() or None

    symbol = symbol or symbol_from_fn
    return symbol, company


def ensure_company_name(symbol: str, company: str | None) -> str:
    if company and str(company).strip():
        return str(company).strip()
    return str(symbol).strip()


def ensure_meta_mapping(history: dict) -> dict:
    if "meta" not in history or not isinstance(history["meta"], dict):
        history["meta"] = {}
    if "company_to_symbol" not in history["meta"] or not isinstance(history["meta"]["company_to_symbol"], dict):
        history["meta"]["company_to_symbol"] = {}
    return history


def normalize_record(r: dict) -> dict | None:
    if not isinstance(r, dict):
        return None
    td = to_datestr(r.get("target_date"))
    if not td:
        return None
    out = dict(r)
    out["target_date"] = td
    out["last_known_date"] = to_datestr(out.get("last_known_date"))
    t = out.get("type")
    if t not in {"forecast_next_day", "backtest_next_day"}:
        return None
    if t == "backtest_next_day" and out.get("actual_value") is None:
        return None
    return out


def merge_records(records: list[dict]) -> list[dict]:
    best = {}
    for r in records:
        rr = normalize_record(r)
        if rr is None:
            continue
        key = (rr.get("target_date"), rr.get("type"))
        if key not in best:
            best[key] = rr
            continue

        a = best[key]
        ga = a.get("generated_at") or ""
        gb = rr.get("generated_at") or ""
        if gb > ga:
            best[key] = rr

    merged = list(best.values())
    merged.sort(key=lambda x: (x.get("target_date", ""), x.get("type", "")))
    return merged


def migrate_history_company_keys(history: dict) -> dict:
    """
    Migrates old history structures to:
      history["data"][company_name] = [records...]
    and ensures company-only records (removes symbol-ish fields).
    Also preserves/creates meta mapping if it can infer it.
    """
    if not isinstance(history, dict):
        return {"data": {}, "meta": {"company_to_symbol": {}}}

    data = history.get("data")
    if not isinstance(data, dict):
        history["data"] = {}
        history = ensure_meta_mapping(history)
        return history

    history = ensure_meta_mapping(history)
    new_data: dict[str, list[dict]] = {}

    for old_key, recs in data.items():
        if not isinstance(recs, list):
            continue

        preferred_company = None
        inferred_symbol = None

        for r in recs:
            if not isinstance(r, dict):
                continue
            c = r.get("company_name") or r.get("company") or r.get("Company")
            if c and str(c).strip():
                preferred_company = str(c).strip()
            s = r.get("symbol") or r.get("Symbol") or r.get("ticker")
            if s and str(s).strip():
                inferred_symbol = str(s).strip()
            if preferred_company:
                break

        # best effort: company key, else keep old key
        target_key = preferred_company or str(old_key)

        if inferred_symbol and preferred_company:
            history["meta"]["company_to_symbol"][preferred_company] = inferred_symbol

        normalized = []
        for r in recs:
            rr = normalize_record(r)
            if rr is None:
                continue

            # enforce company-only record fields
            rr_company = rr.get("company_name") or rr.get("company") or rr.get("Company") or target_key
            rr_company = str(rr_company).strip() if rr_company else target_key

            rr2 = {
                "company_name": rr_company,
                "target_date": rr["target_date"],
                "forecast_value": rr.get("forecast_value"),
                "confidence_low": rr.get("confidence_low"),
                "confidence_high": rr.get("confidence_high"),
                "actual_value": rr.get("actual_value"),
                "last_known_date": rr.get("last_known_date"),
                "last_known_vol": rr.get("last_known_vol"),
                "generated_at": rr.get("generated_at"),
                "type": rr.get("type"),
            }
            normalized.append(rr2)

        if not normalized:
            continue

        new_data.setdefault(target_key, []).extend(normalized)

    for k in list(new_data.keys()):
        new_data[k] = merge_records(new_data[k])

    history["data"] = new_data
    return history


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


def load_vix_fallback() -> pd.DataFrame:
    print("Fetching VIX from Yahoo Finance (fallback)...")
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


def prepare_data(filepath: str, vix_df_external: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str, str]:
    df = pd.read_csv(filepath)
    df = normalize_company_column(df)

    if DATE_COL not in df.columns:
        raise ValueError(f"Missing {DATE_COL} in {filepath}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    symbol, company_raw = parse_symbol_and_company_from_csv(df, filepath)
    company_name = ensure_company_name(symbol, company_raw)

    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID

    close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    if not close_col:
        raise ValueError(f"Missing Close/close in {filepath}")

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))

    df[TARGET_COL] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df[TARGET_COL].shift(1)

    if "VIX" in df.columns:
        df["vix"] = pd.to_numeric(df["VIX"], errors="coerce") / 100.0
        df["vix_lag_1"] = df["vix"].shift(1)
        df["vix"] = df["vix"].ffill()
        df["vix_lag_1"] = df["vix_lag_1"].ffill()
    else:
        if vix_df_external is None:
            raise ValueError("No VIX column in CSV and no external VIX provided.")

        df["date"] = pd.to_datetime(df[DATE_COL]).dt.normalize()
        v = vix_df_external[["date", "vix", "vix_lag_1"]].copy()
        df = df.merge(v, on="date", how="left")
        df = df.sort_values(DATE_COL).reset_index(drop=True)
        df["vix"] = df["vix"].ffill()
        df["vix_lag_1"] = df["vix_lag_1"].ffill()
        df = df.drop(columns=["date"], errors="ignore")

    required = [DATE_COL, TARGET_COL] + FEATURE_COLS
    df = df.dropna(subset=required).reset_index(drop=True)

    return df, symbol, company_name


# =========================================================
# PREDICTION (NEXT TRADING DAY)
# =========================================================
def predict_next_trading_day(pipeline: Chronos2Pipeline, df: pd.DataFrame) -> dict:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    last_trading_ts = pd.to_datetime(df[DATE_COL].iloc[-1]).normalize()
    next_bd = last_trading_ts + BDay(1)

    context_df = df.iloc[-600:].copy()
    context_df = context_df.set_index(DATE_COL).resample("D").ffill().reset_index()

    last_ctx = context_df.iloc[-1]
    last_ctx_date = pd.to_datetime(last_ctx[DATE_COL]).normalize()

    future_rows = []
    for i in range(1, INTERNAL_PRED_LEN_FOR_FREQ + 1):
        future_rows.append({
            ID_COL: last_ctx[ID_COL],
            DATE_COL: last_ctx_date + pd.Timedelta(days=i),
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
    )

    pred_ts = pd.to_datetime(pred_df[DATE_COL]).dt.normalize()
    idx = pred_ts[pred_ts == next_bd].index

    if len(idx) == 0:
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
    print("STARTING DAILY FORECAST JOB (T+1 + T-5 HISTORY)")
    print("=" * 60)

    if not acquire_lock(LOCK_FILE):
        print("Forecast job already running (lock exists). Exiting.")
        return

    try:
        pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map="auto")
        print(f"Model loaded (preferred device: {DEVICE_PREF})")

        vix_df_fallback = load_vix_fallback()

        if os.path.exists(LATEST_FILE):
            with open(LATEST_FILE, "r", encoding="utf-8") as f:
                history = json.load(f)
        else:
            history = {"data": {}}

        history = migrate_history_company_keys(history)
        history = ensure_meta_mapping(history)

        history["last_updated"] = datetime.now().isoformat()
        now_utc = datetime.utcnow().isoformat() + "Z"

        csv_files = sorted(glob.glob(os.path.join(CSV_DIR, "**", "data_*.csv"), recursive=True))
        if not csv_files:
            print(f"No CSV files found in {CSV_DIR}. Expected pattern: **/data_*.csv")
            return

        for fp in tqdm(csv_files, desc="Forecasting Stocks"):
            try:
                full_df, symbol, company_name = prepare_data(fp, vix_df_external=vix_df_fallback)
                if len(full_df) < 80:
                    continue

                # company-only key
                series_key = company_name

                # store mapping for backend / CSV lookups
                history["meta"]["company_to_symbol"][company_name] = symbol

                if series_key not in history["data"]:
                    history["data"][series_key] = []

                existing = merge_records(history["data"][series_key])
                existing_map = {(r["target_date"], r["type"]): r for r in existing}

                pred = predict_next_trading_day(pipeline, full_df)
                if is_weekend_datestr(pred["target_date"]):
                    pred["target_date"] = to_datestr(pd.to_datetime(pred["target_date"]) + BDay(1))

                actual_row = full_df[
                    pd.to_datetime(full_df[DATE_COL]).dt.normalize() == pd.to_datetime(pred["target_date"])
                ]
                actual_val = float(actual_row.iloc[0][TARGET_COL]) if not actual_row.empty else None

                live_rec = {
                    "company_name": company_name,
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
                existing_map[(live_rec["target_date"], "forecast_next_day")] = live_rec

                for back in range(1, DAYS_BACK + 1):
                    df_slice = full_df.iloc[:-back].copy()
                    if len(df_slice) < 80:
                        continue

                    bt = predict_next_trading_day(pipeline, df_slice)
                    if is_weekend_datestr(bt["target_date"]):
                        continue

                    actual_row = full_df[
                        pd.to_datetime(full_df[DATE_COL]).dt.normalize() == pd.to_datetime(bt["target_date"])
                    ]
                    actual_val = float(actual_row.iloc[0][TARGET_COL]) if not actual_row.empty else None
                    if actual_val is None:
                        continue

                    bt_rec = {
                        "company_name": company_name,
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
                    existing_map[(bt_rec["target_date"], "backtest_next_day")] = bt_rec

                history["data"][series_key] = merge_records(list(existing_map.values()))

            except Exception as e:
                sym_log = os.path.basename(fp).replace("data_", "").replace(".csv", "")
                print(f"Skipping {sym_log}: {e}")

        today_dt = datetime.now()
        yesterday = (today_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        snapshot_file = os.path.join(SNAPSHOT_DIR, f"forecast_history_{yesterday}.json")

        atomic_write_json(snapshot_file, history)
        atomic_write_json(LATEST_FILE, history)

        print(f"Snapshot written: {snapshot_file}")
        print(f"Latest updated:   {LATEST_FILE}")

    finally:
        release_lock(LOCK_FILE)


if __name__ == "__main__":
    main()
