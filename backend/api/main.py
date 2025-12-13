import os
import json
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from api.lock import JobLock
from api.derive import derive_metrics

API_DIR = Path(__file__).resolve().parent
REPO_ROOT = API_DIR.parent.parent
ENV_PATH = REPO_ROOT / ".env"
load_dotenv(ENV_PATH)

def resolve_from_root(env_key: str, default: str) -> Path:
    raw = os.getenv(env_key, default)
    p = Path(raw)
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()

FORECAST_STORE_DIR = resolve_from_root("FORECAST_STORE_DIR", "jobs/data_out")
FORECAST_LATEST = FORECAST_STORE_DIR / "forecast_history.json"

CSV_BASE_DIR = resolve_from_root("STOCK_CSV_DIR", "jobs/data_csv")
CSV_LATEST_DIR = CSV_BASE_DIR / "latest"

JOBS_DIR = resolve_from_root("JOBS_DIR", "jobs")

LOCK_DIR = resolve_from_root("LOCK_DIR", ".locks")
INGEST_LOCK = LOCK_DIR / "ingest.lock"
FORECAST_LOCK = LOCK_DIR / "forecast.lock"
LOCK_DIR.mkdir(parents=True, exist_ok=True)

PYTHON_BIN = os.getenv("PYTHON_BIN", str(REPO_ROOT / "venv" / "bin" / "python"))

DATA_SCRIPT = JOBS_DIR / "data.py"
FORECAST_SCRIPT = JOBS_DIR / "generate_daily_forecasts.py"

VT_API_KEYS_RAW = os.getenv("VT_API_KEYS", "").strip()

def parse_api_keys(raw: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not raw:
        return out
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            k, plan = part.split(":", 1)
            out[k.strip()] = plan.strip()
        else:
            out[part] = "free"
    return out

API_KEYS = parse_api_keys(VT_API_KEYS_RAW)

def require_user(authorization: Optional[str] = Header(default=None)) -> Dict[str, Any]:
    if not API_KEYS:
        raise HTTPException(500, "Server misconfigured: VT_API_KEYS not set")

    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(401, "Missing Authorization: Bearer <api_key>")

    key = authorization.split(" ", 1)[1].strip()
    plan = API_KEYS.get(key)
    if not plan:
        raise HTTPException(403, "Invalid API key")

    return {"api_key": key, "plan": plan}

def require_admin(user=Depends(require_user)) -> Dict[str, Any]:
    if user["plan"] != "admin":
        raise HTTPException(403, "Admin privileges required")
    return user

ENABLE_JOB_TRIGGERS = os.getenv("ENABLE_JOB_TRIGGERS", "1").strip().lower() in {"1", "true", "yes"}
JOB_TRIGGER_TOKEN = os.getenv("JOB_TRIGGER_TOKEN", "").strip()

def safe_read_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise HTTPException(404, f"File not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        raise HTTPException(500, f"Failed to read JSON: {e}")

def list_symbols_from_forecast(history: Dict[str, Any]) -> List[str]:
    data = history.get("data", {})
    return sorted(list(data.keys()))

def get_history_records(history: Dict[str, Any], symbol: str) -> List[Dict[str, Any]]:
    rows = history.get("data", {}).get(symbol)
    if not rows:
        raise HTTPException(404, f"No forecast data for symbol {symbol}")
    return rows

def run_job(script_path: Path) -> Dict[str, Any]:
    if not script_path.exists():
        raise HTTPException(500, f"Job script not found: {script_path}")

    cmd = [PYTHON_BIN, str(script_path)]
    if JOB_TRIGGER_TOKEN:
        cmd += ["--token", JOB_TRIGGER_TOKEN]

    try:
        p = subprocess.run(
            cmd,
            cwd=str(JOBS_DIR),
            capture_output=True,
            text=True,
            check=False,
        )
        return {
            "returncode": int(p.returncode),
            "stdout": (p.stdout or "")[-8000:],
            "stderr": (p.stderr or "")[-8000:],
        }
    except Exception as e:
        raise HTTPException(500, f"Failed to run job: {e}")

app = FastAPI(title="VolaTrade API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class JobResult(BaseModel):
    returncode: int
    stdout: str
    stderr: str

@app.get("/health")
def health():
    return {"status": "ok", "ts": time.time()}

@app.get("/v1/stocks")
def stocks(user=Depends(require_user)):
    history = safe_read_json(FORECAST_LATEST)
    return {
        "last_updated": history.get("last_updated"),
        "symbols": list_symbols_from_forecast(history),
    }

@app.get("/v1/forecast/latest")
def forecast_latest(symbol: str, user=Depends(require_user)):
    history = safe_read_json(FORECAST_LATEST)
    rows = get_history_records(history, symbol)
    rec = rows[-1]

    derived = derive_metrics(
        symbol=symbol,
        history_records=rows,
        csv_latest_dir=CSV_LATEST_DIR,
        ewma_lambda=0.94,
        range_days=5,
    )

    return {
        "last_updated": history.get("last_updated"),
        "symbol": symbol,
        "record": rec,
        "derived": derived,
    }

@app.get("/v1/forecast/history")
def forecast_history(symbol: str, user=Depends(require_user)):
    history = safe_read_json(FORECAST_LATEST)
    rows = get_history_records(history, symbol)
    return {
        "last_updated": history.get("last_updated"),
        "symbol": symbol,
        "records": rows,
    }

@app.post("/v1/admin/jobs/ingest", response_model=JobResult)
def admin_job_ingest(admin=Depends(require_admin)):
    if not ENABLE_JOB_TRIGGERS:
        raise HTTPException(403, "Job triggers disabled")
    lock = JobLock(INGEST_LOCK, stale_after_seconds=3600)
    with lock as ok:
        if not ok:
            raise HTTPException(409, "Ingest job already running")
        return run_job(DATA_SCRIPT)

@app.post("/v1/admin/jobs/forecast", response_model=JobResult)
def admin_job_forecast(admin=Depends(require_admin)):
    if not ENABLE_JOB_TRIGGERS:
        raise HTTPException(403, "Job triggers disabled")
    lock = JobLock(FORECAST_LOCK, stale_after_seconds=3600)
    with lock as ok:
        if not ok:
            raise HTTPException(409, "Forecast job already running")
        return run_job(FORECAST_SCRIPT)
