import os
import json
import glob
import platform
import warnings
import argparse
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np
from tqdm import tqdm
from chronos import Chronos2Pipeline
import yfinance as yf

from dotenv import load_dotenv

# =========================================================
# Repo-root aware .env loading + robust path resolving
# =========================================================

SCRIPT_DIR = Path(__file__).resolve().parent          # .../stonksai/jobs
REPO_ROOT = SCRIPT_DIR.parent                         # .../stonksai
ENV_PATH = REPO_ROOT / ".env"

if ENV_PATH.exists():
    load_dotenv(dotenv_path=ENV_PATH)
else:
    # Fallback: still allow running if user exported env vars already
    print(f"[WARN] .env not found at: {ENV_PATH} (continuing with process env)")

def resolve_env_path(var_name: str, default_rel: str) -> Path:
    """
    Read a path from env. If it's relative, resolve it relative to REPO_ROOT.
    This avoids cwd-dependent bugs (uvicorn started from backend/, etc.).
    """
    raw = os.getenv(var_name, default_rel)
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


# =========================================================
# Auth helper (optional)
# =========================================================

def require_job_auth(passed_token: str | None):
    expected = os.getenv("JOB_TRIGGER_TOKEN", "").strip()
    if not expected:
        return
    got = (passed_token or os.getenv("JOB_RUN_TOKEN", "")).strip()
    if not got or got != expected:
        raise SystemExit("[FATAL] Invalid or missing job token. Provide --token or set JOB_RUN_TOKEN.")


# =========================================================
# Device auto-selection
# =========================================================

def pick_best_device() -> str:
    override = os.getenv("CHRONOS_DEVICE", "").strip().lower()
    if override in {"cpu", "cuda", "mps"}:
        print(f"[device] Using override CHRONOS_DEVICE={override}")
        return override

    try:
        import torch
    except Exception as e:
        print(f"[device] torch not available ({e}). Using cpu.")
        return "cpu"

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "CUDA GPU"
        print(f"[device] CUDA available: {name}")
        return "cuda"

    is_mac = platform.system().lower() == "darwin"
    mps_ok = False
    if is_mac and hasattr(torch.backends, "mps"):
        try:
            mps_ok = torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            mps_ok = False

    if mps_ok:
        warnings.filterwarnings(
            "ignore",
            message=".*'pin_memory' argument is set as true but not supported on MPS.*",
            category=UserWarning,
        )
        print("[device] MPS available. Using mps.")
        return "mps"

    print("[device] Using cpu.")
    return "cpu"


DEVICE = pick_best_device()

# =========================================================
# Paths (from .env, resolved against repo root)
# =========================================================

# CSV input directory (should point to latest folder)
# Recommended in .env: STOCK_CSV_DIR=jobs/data_csv/latest
CSV_DIR = resolve_env_path("STOCK_CSV_DIR", "jobs/data_csv/latest")

# Forecast output dir
# Recommended in .env: FORECAST_STORE_DIR=jobs/data_out
OUTPUT_DIR = resolve_env_path("FORECAST_STORE_DIR", "jobs/data_out")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATE_STR = datetime.utcnow().strftime("%Y-%m-%d")
VERSIONED_FILE = OUTPUT_DIR / f"forecast_history-{DATE_STR}.json"
LATEST_FILE = OUTPUT_DIR / "forecast_history.json"

# =========================================================
# Model configuration
# =========================================================

DATE_COL = "timestamp"
TARGET_COL = "ewma_vol"
ID_COL = "id"
DEFAULT_ID = "series_1"
FREQ = "B"
LAMBDA = 0.94
FEATURE_COLS = ["ewma_vol_lag_1", "vix_lag_1"]


# =========================================================
# Helpers
# =========================================================

def calculate_ewma_volatility(returns: pd.Series, lambda_=0.94) -> pd.Series:
    ewma_var = np.zeros(len(returns))
    returns_filled = returns.fillna(0).values
    ewma_var[0] = returns_filled[0] ** 2
    for t in range(1, len(returns)):
        ewma_var[t] = lambda_ * ewma_var[t - 1] + (1 - lambda_) * (returns_filled[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


def load_vix() -> pd.DataFrame:
    """
    VIX is already in your stock CSVs, but we fetch ^VIX again to ensure we can
    build consistent lag features even if CSV gaps exist.
    """
    print("[vix] Fetching VIX from Yahoo Finance (^VIX) for lags...")
    vix = yf.Ticker("^VIX")
    hist = vix.history(period="2y")
    if hist.empty:
        raise RuntimeError("VIX history returned empty from yfinance.")
    hist.index = hist.index.tz_localize(None)

    vix_df = hist[["Close"]].rename(columns={"Close": "vix"})
    vix_df["date"] = vix_df.index
    vix_df = vix_df.reset_index(drop=True)
    vix_df["vix"] = vix_df["vix"] / 100.0
    vix_df["vix_lag_1"] = vix_df["vix"].shift(1)
    return vix_df


def prepare_data(filepath: Path, vix_df: pd.DataFrame) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    if DATE_COL not in df.columns:
        raise ValueError(f"Missing '{DATE_COL}' column in {filepath}")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    if ID_COL not in df.columns:
        df[ID_COL] = DEFAULT_ID

    close_col = "Close" if "Close" in df.columns else ("close" if "close" in df.columns else None)
    if close_col is None:
        raise ValueError(f"Missing Close/close column in {filepath}")

    df["log_return"] = np.log(df[close_col] / df[close_col].shift(1))
    df["ewma_vol"] = calculate_ewma_volatility(df["log_return"], LAMBDA)
    df["ewma_vol_lag_1"] = df["ewma_vol"].shift(1)

    # merge vix for lag features
    df = df.merge(
        vix_df[["date", "vix", "vix_lag_1"]],
        left_on=DATE_COL,
        right_on="date",
        how="left",
    ).drop(columns=["date"], errors="ignore")

    # business day reindex
    pieces = []
    for sid, g in df.groupby(ID_COL):
        g = g.set_index(DATE_COL)
        idx = pd.date_range(g.index.min(), g.index.max(), freq=FREQ)
        g = g.reindex(idx).ffill()
        g[ID_COL] = sid
        g.index.name = DATE_COL
        pieces.append(g.reset_index())

    out = pd.concat(pieces, ignore_index=True)

    # Ensure required cols exist
    needed = [TARGET_COL, "ewma_vol_lag_1", "vix", "vix_lag_1"]
    out = out.dropna(subset=needed)
    return out


def predict_next_day(pipeline: Chronos2Pipeline, df: pd.DataFrame) -> dict:
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    context_df = df.iloc[-600:].copy()
    context_df = context_df.set_index(DATE_COL).resample("D").ffill().reset_index()

    last_row = context_df.iloc[-1]
    next_date = last_row[DATE_COL] + pd.Timedelta(days=1)

    future_df = pd.DataFrame([{
        ID_COL: last_row[ID_COL],
        DATE_COL: next_date,
        "ewma_vol_lag_1": last_row["ewma_vol"],
        "vix_lag_1": last_row["vix"],
    }])

    pred = pipeline.predict_df(
        context_df[[ID_COL, DATE_COL, TARGET_COL] + FEATURE_COLS],
        future_df=future_df[[ID_COL, DATE_COL] + FEATURE_COLS],
        prediction_length=1,
        quantile_levels=[0.1, 0.5, 0.9],
        id_column=ID_COL,
        timestamp_column=DATE_COL,
        target=TARGET_COL,
    )

    return {
        "target_date": next_date,
        "forecast_value": float(pred["0.5"].iloc[0]),
        "confidence_low": float(pred["0.1"].iloc[0]),
        "confidence_high": float(pred["0.9"].iloc[0]),
        "last_known_date": last_row[DATE_COL],
        "last_known_vol": float(last_row["ewma_vol"]),
    }


def atomic_write_json(path: Path, payload: dict):
    tmp = Path(str(path) + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def load_history() -> dict:
    if LATEST_FILE.exists():
        try:
            with open(LATEST_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[history] Warning reading latest history ({e}). Starting fresh.")
    return {"data": {}}


def date_key(dt) -> str:
    return str(pd.to_datetime(dt).date())


def iso_date(dt) -> str:
    return (
        pd.to_datetime(dt)
        .to_pydatetime()
        .replace(hour=0, minute=0, second=0, microsecond=0)
        .isoformat()
    )


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", default=None, help="Job trigger token (optional; required if JOB_TRIGGER_TOKEN is set).")
    args = parser.parse_args()

    require_job_auth(args.token)

    print("=" * 60)
    print("DAILY VOLATILITY FORECAST GENERATOR")
    print("=" * 60)
    print(f"[repo] REPO_ROOT: {REPO_ROOT}")
    print(f"[in]  CSV_DIR:    {CSV_DIR}")
    print(f"[out] OUTPUT_DIR:{OUTPUT_DIR}")
    print(f"[out] VERSIONED: {VERSIONED_FILE}")
    print(f"[out] LATEST:    {LATEST_FILE}")
    print(f"[device] {DEVICE}")

    if not CSV_DIR.exists():
        raise SystemExit(f"[FATAL] CSV_DIR not found: {CSV_DIR}")

    csv_files = sorted(CSV_DIR.glob("data_*.csv"))
    print(f"[in] Found {len(csv_files)} CSV files.")
    if not csv_files:
        raise SystemExit("[FATAL] No CSVs found (pattern data_*.csv). Run data_final.py first.")

    print("[model] Loading Chronos...")
    pipeline = Chronos2Pipeline.from_pretrained("amazon/chronos-2", device_map=DEVICE)

    vix_df = load_vix()

    history = load_history()
    history.setdefault("data", {})

    for fp in tqdm(csv_files, desc="Generating Forecasts"):
        stock = fp.stem.replace("data_", "")
        try:
            df = prepare_data(fp, vix_df)
            if len(df) < 100:
                print(f"[SKIP] {stock}: not enough rows ({len(df)})")
                continue

            res = predict_next_day(pipeline, df)

            actual_row = df[df[DATE_COL] == res["target_date"]]
            actual_value = float(actual_row.iloc[0][TARGET_COL]) if not actual_row.empty else None

            record = {
                "target_date": iso_date(res["target_date"]),
                "forecast_value": res["forecast_value"],
                "confidence_low": res["confidence_low"],
                "confidence_high": res["confidence_high"],
                "actual_value": actual_value,
                "last_known_date": iso_date(res["last_known_date"]),
                "last_known_vol": res["last_known_vol"],
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "type": "forecast_next_day",
            }

            history["data"].setdefault(stock, [])
            merged = {date_key(x["target_date"]): x for x in history["data"][stock]}
            merged[date_key(record["target_date"])] = record
            history["data"][stock] = sorted(merged.values(), key=lambda x: x["target_date"])

            print(f"[OK] {stock}: {record['forecast_value']:.4f}")

        except Exception as e:
            print(f"[ERR] {stock}: {e}")

    history["last_updated"] = datetime.utcnow().isoformat() + "Z"
    atomic_write_json(VERSIONED_FILE, history)
    atomic_write_json(LATEST_FILE, history)

    print(f"[ok] Saved snapshot: {VERSIONED_FILE}")
    print(f"[ok] Updated latest: {LATEST_FILE}")
    print("DONE.")


if __name__ == "__main__":
    main()
