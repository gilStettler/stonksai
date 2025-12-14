from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd


def calculate_ewma_volatility(returns: pd.Series, lambda_: float = 0.94) -> pd.Series:
    n = len(returns)
    ewma_var = np.zeros(n, dtype=float)
    r = returns.fillna(0.0).values
    ewma_var[0] = r[0] ** 2
    for t in range(1, n):
        ewma_var[t] = lambda_ * ewma_var[t - 1] + (1 - lambda_) * (r[t] ** 2)
    return pd.Series(np.sqrt(ewma_var), index=returns.index)


def _close_col(df: pd.DataFrame) -> str:
    if "Close" in df.columns:
        return "Close"
    if "close" in df.columns:
        return "close"
    raise ValueError("CSV missing Close/close column.")


def last_n_realized(csv_latest_dir: Path, symbol: str, n: int = 5, lambda_: float = 0.94) -> Tuple[List[Dict[str, Any]], Optional[float]]:
    """
    Returns:
      points: [{date, realized}] for last n days from CSV (realized EWMA vol)
      last_val: last realized EWMA vol
    """
    fp = Path(csv_latest_dir) / f"data_{symbol}.csv"
    if not fp.exists():
        return [], None

    df = pd.read_csv(fp)
    if "timestamp" not in df.columns:
        return [], None

    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    c = _close_col(df)
    df["log_return"] = np.log(df[c] / df[c].shift(1))
    df["realized_ewma"] = calculate_ewma_volatility(df["log_return"], lambda_=lambda_)

    tail = df.dropna(subset=["realized_ewma"]).tail(n)
    if tail.empty:
        return [], None

    points = []
    for _, r in tail.iterrows():
        points.append({"date": str(pd.to_datetime(r["timestamp"]).date()), "realized": float(r["realized_ewma"])})

    last_val = float(tail["realized_ewma"].iloc[-1])
    return points, last_val


def range_from_points(points: List[Dict[str, Any]]) -> Dict[str, Any]:
    vals = [p["realized"] for p in points if p.get("realized") is not None]
    if not vals:
        return {"n": 0, "min": None, "max": None, "pos": None}

    vmin = float(np.min(vals))
    vmax = float(np.max(vals))
    return {"n": len(vals), "min": vmin, "max": vmax, "pos": None}


def position_in_range(value: Optional[float], vmin: Optional[float], vmax: Optional[float]) -> Optional[float]:
    if value is None or vmin is None or vmax is None:
        return None
    if vmax == vmin:
        return 0.5
    return float((value - vmin) / (vmax - vmin))


def delta_arrow(delta: Optional[float], eps: float = 1e-6) -> Optional[str]:
    if delta is None:
        return None
    if delta > eps:
        return "↑"
    if delta < -eps:
        return "↓"
    return "→"


def risk_from_pos(pos: Optional[float]) -> Dict[str, Any]:
    """
    Simple beginner-friendly risk label based on where forecast lies inside last-n realized range.
    """
    if pos is None:
        return {"label": "Medium", "reason": "missing_pos"}

    if pos <= 0.33:
        return {"label": "Low", "reason": "forecast_near_recent_low"}
    if pos >= 0.66:
        return {"label": "High", "reason": "forecast_near_recent_high"}
    return {"label": "Medium", "reason": "forecast_mid_range"}


def _pick_latest_forecast_record(history_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prefer newest forecast_next_day by target_date; fallback to last record.
    Assumes records already normalized (YYYY-MM-DD target_date).
    """
    if not history_records:
        return {}
    forecasts = [r for r in history_records if r.get("type") == "forecast_next_day" and r.get("target_date")]
    if not forecasts:
        return history_records[-1]
    return max(forecasts, key=lambda r: r["target_date"])


def derive_metrics(
    symbol: str,
    history_records: List[Dict[str, Any]],
    csv_latest_dir: Path,
    ewma_lambda: float = 0.94,
    range_days: int = 5,
) -> Dict[str, Any]:
    """
    Produces:
      - confidence_band from latest FORECAST record
      - 5d realized range from CSV (min/max/n + pos)
      - delta_vol comparing forecast vs last realized from CSV
      - risk label based on pos
    """
    if not history_records:
        return {}

    latest = _pick_latest_forecast_record(history_records)

    try:
        forecast_val = float(latest.get("forecast_value"))
    except Exception:
        forecast_val = None

    band = {}
    try:
        band = {"low": float(latest.get("confidence_low")), "high": float(latest.get("confidence_high"))}
    except Exception:
        band = {"low": None, "high": None}

    points, last_realized = last_n_realized(csv_latest_dir, symbol, n=range_days, lambda_=ewma_lambda)
    r = range_from_points(points)

    pos = position_in_range(forecast_val, r["min"], r["max"])
    r["pos"] = pos
    r["points"] = points

    dvol = None
    if forecast_val is not None and last_realized is not None:
        dvol = float(forecast_val - last_realized)

    return {
        "confidence_band": band,
        "range_5d": r,
        "delta_vol": dvol,
        "delta_arrow": delta_arrow(dvol),
        "risk": risk_from_pos(pos),
        "last_realized": last_realized,
    }
