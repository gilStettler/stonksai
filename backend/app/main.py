from __future__ import annotations

import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from backend.api.lock import JobLock
from backend.api.derive import derive_metrics

# =============================================================================
# Load .env (repo root)
# =============================================================================
REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(REPO_ROOT / ".env")

# =============================================================================
# Config
# =============================================================================
VT_API_KEYS_RAW = os.getenv("VT_API_KEYS", "devkey:admin")

FORECAST_STORE_DIR = REPO_ROOT / os.getenv("FORECAST_STORE_DIR", "jobs/data_out")
STOCK_CSV_DIR = REPO_ROOT / os.getenv("STOCK_CSV_DIR", "jobs/data_csv")
JOBS_DIR = REPO_ROOT / os.getenv("JOBS_DIR", "jobs")
LOCK_DIR = REPO_ROOT / os.getenv("LOCK_DIR", ".locks")

PYTHON_BIN = os.getenv("PYTHON_BIN", "python")

ENABLE_JOB_TRIGGERS = os.getenv("ENABLE_JOB_TRIGGERS", "1").strip() == "1"
JOB_TRIGGER_TOKEN = os.getenv("JOB_TRIGGER_TOKEN", "").strip()

FORECAST_STORE_DIR.mkdir(parents=True, exist_ok=True)
LOCK_DIR.mkdir(parents=True, exist_ok=True)

FORECAST_FILE = FORECAST_STORE_DIR / "forecast_history.json"

# =============================================================================
# Auth
# =============================================================================
def parse_api_keys(raw: str) -> Dict[str, str]:
    """
    Parses VT_API_KEYS like:
      vt_live_admin_123:admin,vt_live_free_abc:free
    Returns dict: {key: role}
    """
    out: Dict[str, str] = {}
    for part in (raw or "").split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, role = part.split(":", 1)
        else:
            k, role = part, "free"
        out[k.strip()] = role.strip()
    return out


API_KEYS = parse_api_keys(VT_API_KEYS_RAW)


def require_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, str]:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing Bearer token")

    token = authorization.split(" ", 1)[1].strip()
    role = API_KEYS.get(token)
    if not role:
        raise HTTPException(status_code=403, detail="Invalid API key")

    return {"token": token, "role": role}


def require_admin(user: Dict[str, str] = Depends(require_user)) -> Dict[str, str]:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return user


# =============================================================================
# Helpers
# =============================================================================
def read_forecast_history() -> Dict[str, Any]:
    if not FORECAST_FILE.exists():
        # initial empty structure
        return {"last_updated": None, "data": {}}
    try:
        return json.loads(FORECAST_FILE.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not read forecast_history.json: {e}")


def run_job_sync(script: Path) -> Dict[str, Any]:
    """
    Runs a job synchronously and returns stdout/stderr. Useful for Streamlit buttons.
    """
    if not ENABLE_JOB_TRIGGERS:
        raise HTTPException(status_code=403, detail="Job triggers disabled (ENABLE_JOB_TRIGGERS=0)")

    if not script.exists():
        raise HTTPException(status_code=404, detail=f"Job script missing: {script}")

    p = subprocess.run(
        [PYTHON_BIN, str(script)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )
    return {
        "returncode": p.returncode,
        "stdout": (p.stdout or "")[-4000:],
        "stderr": (p.stderr or "")[-4000:],
    }


# =============================================================================
# App
# =============================================================================
app = FastAPI(title="StonksAI Backend", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health
# =============================================================================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "time": datetime.utcnow().isoformat(),
        "repo_root": str(REPO_ROOT),
        "forecast_file_exists": FORECAST_FILE.exists(),
    }


# =============================================================================
# Stocks
# =============================================================================
@app.get("/v1/stocks")
def list_stocks(_: Dict[str, str] = Depends(require_user)):
    """
    Preferred: use forecast_history.json as source of truth (symbols actually available).
    Fallback: list CSVs if forecast file doesn't exist yet.
    """
    data = read_forecast_history()
    symbols = sorted((data.get("data") or {}).keys())

    if symbols:
        return {"symbols": symbols, "last_updated": data.get("last_updated")}

    # fallback to CSVs if no forecasts yet
    if STOCK_CSV_DIR.exists():
        csv_symbols = sorted(p.stem for p in STOCK_CSV_DIR.glob("*.csv"))
    else:
        csv_symbols = []

    return {"symbols": csv_symbols, "last_updated": data.get("last_updated")}


# =============================================================================
# Forecast - latest
# =============================================================================
@app.get("/v1/forecast/latest")
def forecast_latest(symbol: str, _: Dict[str, str] = Depends(require_user)):
    data = read_forecast_history()
    rows = (data.get("data") or {}).get(symbol)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No forecast for symbol: {symbol}")

    latest = rows[-1]

    # optional derived metrics (safe)
    try:
        derived = derive_metrics(
            symbol=symbol,
            history_records=rows,
            csv_latest_dir=STOCK_CSV_DIR / "latest",
            ewma_lambda=0.94,
            range_days=5,
        )
    except Exception:
        derived = {}

    return {
        "symbol": symbol,
        "record": latest,
        "derived": derived,
        "last_updated": data.get("last_updated"),
    }


# =============================================================================
# Forecast - history
# =============================================================================
@app.get("/v1/forecast/history")
def forecast_history(symbol: str, _: Dict[str, str] = Depends(require_user)):
    data = read_forecast_history()
    rows = (data.get("data") or {}).get(symbol)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history for symbol: {symbol}")

    return {"symbol": symbol, "records": rows, "last_updated": data.get("last_updated")}


# =============================================================================
# Forecast - backtests (frontend uses this)
# =============================================================================
@app.get("/v1/forecast/backtests")
def forecast_backtests(symbol: str, days: int = 5, _: Dict[str, str] = Depends(require_user)):
    data = read_forecast_history()
    rows = (data.get("data") or {}).get(symbol)
    if not rows:
        raise HTTPException(status_code=404, detail=f"No history for symbol: {symbol}")

    days = max(1, min(int(days), 120))
    return {
        "symbol": symbol,
        "days": days,
        "records": rows[-days:],
        "last_updated": data.get("last_updated"),
    }


# =============================================================================
# Admin - jobs (frontend buttons expect these endpoints)
# =============================================================================
@app.post("/v1/admin/jobs/ingest")
def admin_ingest(_: Dict[str, str] = Depends(require_admin)):
    """
    Runs jobs/data.py synchronously.
    """
    script = JOBS_DIR / "data.py"

    lock = JobLock(str(LOCK_DIR / "ingest.lock"))
    with lock:
        result = run_job_sync(script)

    return {"job": "ingest", **result}


@app.post("/v1/admin/jobs/forecast")
def admin_forecast(_: Dict[str, str] = Depends(require_admin)):
    """
    Runs jobs/generate_daily_forecasts.py synchronously.
    """
    script = JOBS_DIR / "generate_daily_forecasts.py"

    lock = JobLock(str(LOCK_DIR / "forecast.lock"))
    with lock:
        result = run_job_sync(script)

    return {"job": "forecast", **result}
